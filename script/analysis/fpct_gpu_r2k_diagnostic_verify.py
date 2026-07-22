#!/usr/bin/env python3
"""Independent verifier for sealed R2k DIAGNOSTIC_ONLY artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


ORDER = (
    ("c_post", "f"),
    ("f", "c_post"),
    ("f", "c_post"),
    ("c_post", "f"),
    ("c_post", "f"),
    ("f", "c_post"),
    ("f", "c_post"),
    ("c_post", "f"),
)
SEEDS = tuple(range(104729, 104737))
SYNC_NAMES = {
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "aten::item",
    "aten::_local_scalar_dense",
}
HOT_SCOPES = {"fpct.pack", "fpct.attention", "fpct.project_candidates"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path: Path) -> Any:
    return json.loads(path.read_text())


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty percentile")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def bootstrap_ucb(values: list[float]) -> float:
    generator = random.Random(20260722)
    replicates = []
    for _ in range(50000):
        sample = [values[generator.randrange(len(values))] for _ in values]
        replicates.append(statistics.median(sample))
    return sorted(replicates)[int(0.95 * (len(replicates) - 1))]


def parse_gpu(line: str) -> dict[str, Any]:
    fields = [field.strip() for field in line.split(",")]
    if len(fields) != 9 or fields[0].startswith("unavailable"):
        raise ValueError(f"GPU telemetry is not parseable: {line!r}")
    return {
        "uuid": fields[0],
        "name": fields[1],
        "pstate": fields[2],
        "sm_clock_mhz": float(fields[3]),
        "memory_clock_mhz": float(fields[4]),
        "temperature_c": float(fields[5]),
        "power_w": float(fields[6]),
        "power_limit_w": float(fields[7]),
        "throttle": fields[8],
    }


def trace_summary(path: Path) -> dict[str, Any]:
    payload = load(path)
    events = payload["traceEvents"]
    scientific = next(
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name") == "scientific_forward"
        and event.get("ph") == "X"
    )
    start = float(scientific["ts"])
    end = start + float(scientific["dur"])
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and start <= float(event.get("ts", -1)) < end
    ]
    by_kernel: dict[str, float] = defaultdict(float)
    for event in kernels:
        by_kernel[str(event.get("name", "<unnamed>"))] += float(event["dur"])
    scopes: dict[str, list[tuple[float, float, int]]] = defaultdict(list)
    for event in events:
        if event.get("cat") != "user_annotation" or event.get("ph") != "X":
            continue
        name = str(event.get("name"))
        event_start = float(event["ts"])
        scopes[name].append(
            (event_start, event_start + float(event["dur"]), int(event.get("tid", -1)))
        )
    hot_intervals = [interval for name in HOT_SCOPES for interval in scopes.get(name, [])]
    hot_sync = []
    for event in events:
        if event.get("name") not in SYNC_NAMES or event.get("ph") != "X":
            continue
        timestamp = float(event.get("ts", -1))
        thread = int(event.get("tid", -2))
        if any(left <= timestamp <= right and thread == scope_thread for left, right, scope_thread in hot_intervals):
            hot_sync.append(str(event["name"]))
    scope_seconds = {
        name: sum(right - left for left, right, _thread in intervals) / 1e6
        for name, intervals in scopes.items()
        if name in HOT_SCOPES | {"receiver_attention", "scientific_forward"}
    }
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "scientific_scope_seconds": float(scientific["dur"]) / 1e6,
        "summed_kernel_seconds": sum(float(event["dur"]) for event in kernels) / 1e6,
        "kernel_event_count": len(kernels),
        "scope_seconds": scope_seconds,
        "hot_path_sync_events": hot_sync,
        "top_kernel_seconds": [
            {"name": name, "seconds": duration / 1e6}
            for name, duration in sorted(
                by_kernel.items(), key=lambda item: (-item[1], item[0])
            )[:20]
        ],
        "kernel_totals": by_kernel,
    }


def verify(root: Path) -> dict[str, Any]:
    block_root = root / "results/blocks"
    attestation_root = root / "attestations"
    blocks = [load(block_root / f"block_{index:02d}.json") for index in range(8)]
    canary_ratios: dict[str, dict[str, list[float]]] = {
        label: {"cuda": [], "wall": []}
        for label in ("checkpoint_native", "forced_on")
    }
    hardware = []
    peak_hbm = []
    expansion = []
    artifacts = []
    for index, block in enumerate(blocks):
        if block["classification"] != "DIAGNOSTIC_ONLY":
            raise ValueError("block classification changed")
        if block["block_index"] != index or block["process_seed"] != SEEDS[index]:
            raise ValueError(f"block identity mismatch: {index}")
        if tuple(block["order"]) != ORDER[index]:
            raise ValueError(f"block order mismatch: {index}")
        if block["accuracy_or_correctness_accessed"]:
            raise ValueError("accuracy firewall violation")
        attestation = load(attestation_root / f"block_{index:02d}.json")
        if attestation.get("target_exit_code") != 0:
            raise ValueError(f"unsealed block: {index}")
        artifacts.append(
            {
                "path": str(block_root / f"block_{index:02d}.json"),
                "sha256": sha256_file(block_root / f"block_{index:02d}.json"),
                "bytes": (block_root / f"block_{index:02d}.json").stat().st_size,
                "attestation_sha256": sha256_file(attestation_root / f"block_{index:02d}.json"),
            }
        )
        for label in canary_ratios:
            arms = block["canaries"][label]["arms"]
            if tuple(block["canaries"][label]["order"]) != ORDER[index]:
                raise ValueError("canary order mismatch")
            by_operator = {arm["operator"]: arm for arm in arms}
            if set(by_operator) != {"c_post", "f"}:
                raise ValueError("operator set mismatch")
            for arm in arms:
                timing = arm["timing"]
                cuda = [float(value) for value in timing["cuda_event_seconds"]]
                wall = [float(value) for value in timing["synchronized_wall_seconds"]]
                if timing["warmups"] != 20 or timing["measurements"] != 50:
                    raise ValueError("timing count contract mismatch")
                if len(cuda) != 50 or len(wall) != 50:
                    raise ValueError("raw sample count mismatch")
                if not all(math.isfinite(value) and value > 0 for value in cuda + wall):
                    raise ValueError("nonfinite/nonpositive latency")
                if statistics.median(cuda) != timing["cuda_median_seconds"]:
                    raise ValueError("CUDA median mismatch")
                if statistics.median(wall) != timing["wall_median_seconds"]:
                    raise ValueError("wall median mismatch")
                if arm["accuracy_or_correctness_accessed"]:
                    raise ValueError("arm accuracy firewall violation")
                peak_hbm.append(float(arm["peak_hbm_gib"]))
                if arm["expanded_slot_ratios"]:
                    expansion.extend(float(value) for value in arm["expanded_slot_ratios"])
                for key in ("telemetry_before", "telemetry_after"):
                    telemetry = arm[key]
                    parsed = parse_gpu(telemetry["gpu"])
                    parsed.update(
                        {
                            "node_name": telemetry["node_name"],
                            "pod_name": telemetry["pod_name"],
                            "compute_process_count": len(
                                [line for line in telemetry["compute_processes"].splitlines() if line]
                            ),
                            "scheduler": telemetry["cpu_scheduler"],
                            "cpu_affinity_count": len(telemetry["cpu_affinity"]),
                        }
                    )
                    hardware.append(parsed)
            canary_ratios[label]["cuda"].append(
                by_operator["f"]["timing"]["cuda_median_seconds"]
                / by_operator["c_post"]["timing"]["cuda_median_seconds"]
            )
            canary_ratios[label]["wall"].append(
                by_operator["f"]["timing"]["wall_median_seconds"]
                / by_operator["c_post"]["timing"]["wall_median_seconds"]
            )

    aggregate_path = root / "results/diagnostic_aggregate.json"
    aggregate = load(aggregate_path)
    if aggregate["classification"] != "DIAGNOSTIC_ONLY" or aggregate["may_produce_r2k_go"]:
        raise ValueError("aggregate claim boundary violation")
    recomputed = {}
    for label, values in canary_ratios.items():
        median = statistics.median(values["cuda"])
        ucb = bootstrap_ucb(values["cuda"])
        row = aggregate["canaries"][label]
        if abs(median - row["cuda_balanced_median_ratio"]) > 1e-15:
            raise ValueError("aggregate median mismatch")
        if abs(ucb - row["cuda_block_bootstrap_one_sided_95_ucb"]) > 1e-15:
            raise ValueError("aggregate UCB mismatch")
        recomputed[label] = {
            "cuda_block_ratios": values["cuda"],
            "wall_block_ratios": values["wall"],
            "cuda_balanced_median_ratio": median,
            "cuda_block_bootstrap_one_sided_95_ucb": ucb,
            "wall_balanced_median_ratio": statistics.median(values["wall"]),
            "qualified": median <= 1.35 and ucb <= 1.50,
        }
    qualified = all(row["qualified"] for row in recomputed.values())
    if qualified != (aggregate["status"] == "DIAGNOSTIC_QUALIFIED"):
        raise ValueError("qualification mismatch")

    geometry_path = root / "results/geometry/geometry.json"
    geometry = load(geometry_path)
    if geometry["accuracy_or_correctness_accessed"]:
        raise ValueError("geometry accuracy firewall violation")
    geometry_rows = [
        {
            "panel_id": row["panel_id"],
            "cuda_median_ratio": float(row["cuda_median_ratio"]),
            "wall_median_ratio": float(row["wall_median_ratio"]),
        }
        for row in geometry["rows"]
    ]
    traces = {
        name: trace_summary(root / f"results/traces/{name}.chrome.json")
        for name in ("c_post", "f")
    }
    f_kernels = traces["f"].pop("kernel_totals")
    cpost_kernels = traces["c_post"].pop("kernel_totals")
    kernel_delta = sorted(
        (
            (name, duration - cpost_kernels.get(name, 0.0))
            for name, duration in f_kernels.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )[:20]
    if traces["c_post"]["hot_path_sync_events"] or traces["f"]["hot_path_sync_events"]:
        raise ValueError("hot-path host synchronization detected")
    uuids = sorted({row["uuid"] for row in hardware})
    if len(uuids) != 1:
        raise ValueError("multiple GPU UUIDs in paired blocks")
    if any(row["pstate"] != "P0" for row in hardware):
        raise ValueError("non-P0 diagnostic sample")
    if any(row["throttle"] != "0x0000000000000000" for row in hardware):
        raise ValueError("active throttle reason in diagnostic sample")
    if any(row["compute_process_count"] != 1 for row in hardware):
        raise ValueError("foreign compute process observed")

    return {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2k_diagnostic_independent_verify_v1",
        "status": "DIAGNOSTIC_QUALIFIED" if qualified else "DIAGNOSTIC_NOT_QUALIFIED",
        "classification": "DIAGNOSTIC_ONLY",
        "canaries": recomputed,
        "hardware": {
            "gpu_uuid": uuids[0],
            "gpu_name": hardware[0]["name"],
            "pstate": "P0",
            "sm_clock_mhz_min_max": [min(row["sm_clock_mhz"] for row in hardware), max(row["sm_clock_mhz"] for row in hardware)],
            "memory_clock_mhz_min_max": [min(row["memory_clock_mhz"] for row in hardware), max(row["memory_clock_mhz"] for row in hardware)],
            "temperature_c_min_max": [min(row["temperature_c"] for row in hardware), max(row["temperature_c"] for row in hardware)],
            "power_w_min_max": [min(row["power_w"] for row in hardware), max(row["power_w"] for row in hardware)],
            "throttle_reasons": "none",
            "foreign_compute_processes": 0,
            "cpu_affinity_count": sorted({row["cpu_affinity_count"] for row in hardware}),
            "schedulers": sorted({row["scheduler"] for row in hardware}),
        },
        "resources": {
            "peak_hbm_gib_max": max(peak_hbm),
            "expansion_mean": statistics.mean(expansion),
            "expansion_p95": percentile(expansion, 0.95),
        },
        "geometry": geometry_rows,
        "traces": traces,
        "f_minus_cpost_top_kernel_delta_seconds": [
            {"name": name, "seconds": duration / 1e6}
            for name, duration in kernel_delta
        ],
        "artifacts": artifacts
        + [
            {"path": str(aggregate_path), "sha256": sha256_file(aggregate_path), "bytes": aggregate_path.stat().st_size},
            {"path": str(geometry_path), "sha256": sha256_file(geometry_path), "bytes": geometry_path.stat().st_size},
        ],
        "accuracy_or_correctness_accessed": False,
        "may_produce_r2k_go": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = verify(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": payload["status"], "classification": payload["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
