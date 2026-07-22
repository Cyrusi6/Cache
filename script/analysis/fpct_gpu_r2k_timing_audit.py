#!/usr/bin/env python3
"""Read-only descriptive audit of historical FPCT R2 P2/P3 timing artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNS = {
    "R2h": Path("/netdisk/lijunsi/fpct-confirmatory/fpct-r2h-39af03d-v1"),
    "R2i-v1": Path("/netdisk/lijunsi/fpct-confirmatory/fpct-r2i-8d21c72-v1"),
    "R2i-v2": Path("/netdisk/lijunsi/fpct-confirmatory/fpct-r2i-8d21c72-v2"),
    "R2j": Path("/netdisk/lijunsi/fpct-confirmatory/fpct-r2j-efa02fb-v1"),
}
PROFILES = ("P2_CPOST_OFF", "P3_F_OFF")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _median_absolute_deviation(values: list[float]) -> float:
    center = statistics.median(values)
    return statistics.median(abs(value - center) for value in values)


def _linear_slope(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean_x = (len(values) - 1) / 2
    mean_y = statistics.mean(values)
    numerator = sum((index - mean_x) * (value - mean_y) for index, value in enumerate(values))
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    return numerator / denominator if denominator else 0.0


def latency_summary(values: list[float]) -> dict[str, Any]:
    mean = statistics.mean(values)
    standard_deviation = statistics.pstdev(values)
    median = statistics.median(values)
    return {
        "samples_seconds": values,
        "count": len(values),
        "mean_seconds": mean,
        "median_seconds": median,
        "max_seconds": max(values),
        "mad_seconds": _median_absolute_deviation(values),
        "cv": standard_deviation / mean if mean else None,
        "linear_iteration_slope_seconds": _linear_slope(values),
        "tail_max_over_median": max(values) / median if median else None,
        "first_over_median": values[0] / median if median else None,
    }


def trace_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    events = payload.get("traceEvents", [])
    scientific = next(
        (
            event
            for event in events
            if event.get("cat") == "user_annotation"
            and event.get("name") == "scientific_forward"
            and event.get("ph") == "X"
        ),
        None,
    )
    if scientific is None:
        raise ValueError(f"scientific_forward scope missing: {path}")
    start = float(scientific["ts"])
    end = start + float(scientific["dur"])
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and start <= float(event.get("ts", -math.inf)) < end
    ]
    by_name: dict[str, float] = defaultdict(float)
    for event in kernels:
        by_name[str(event.get("name", "<unnamed>"))] += float(event.get("dur", 0.0))
    top = sorted(by_name.items(), key=lambda item: (-item[1], item[0]))[:20]
    device_properties = payload.get("deviceProperties", [])
    return {
        "scientific_scope_wall_seconds_with_profiler": float(scientific["dur"]) / 1e6,
        "kernel_event_count": len(kernels),
        "summed_kernel_seconds": sum(float(event.get("dur", 0.0)) for event in kernels) / 1e6,
        "top_kernel_seconds": [{"name": name, "seconds": duration / 1e6} for name, duration in top],
        "device_properties": device_properties,
        "trace_sha256": sha256_file(path),
        "hardware_telemetry": {
            "gpu_uuid": "not_captured",
            "sm_clock_mhz": "not_captured",
            "memory_clock_mhz": "not_captured",
            "temperature_c": "not_captured",
            "power_w": "not_captured",
            "p_state": "not_captured",
            "clocks_throttle_reasons": "not_captured",
            "foreign_gpu_processes": "not_captured",
            "node_pod_inventory": "not_captured",
            "cpu_affinity_and_scheduler": "not_captured",
        },
    }


def audit_run(label: str, root: Path) -> dict[str, Any]:
    profile_rows: dict[str, Any] = {}
    for profile_id in PROFILES:
        directory = root / "results/pretrained_matrix/profiles" / profile_id
        manifest_path = directory / "profile_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        trace_path = directory / f"{profile_id}.chrome.json"
        trace = trace_summary(trace_path)
        profile_rows[profile_id] = {
            "manifest_sha256": sha256_file(manifest_path),
            "manifest_mtime_utc": datetime.fromtimestamp(
                manifest_path.stat().st_mtime, tz=timezone.utc
            ).isoformat(),
            "latency": latency_summary([float(value) for value in manifest["latency_samples_seconds"]]),
            "trace": trace,
            "peak_hbm_gib": float(manifest["peak_hbm_gib"]),
            "expanded_slot_ratios": manifest["expanded_slot_ratios"],
            "hot_path_sync_event_count": int(manifest["hot_path_sync_event_count"]),
            "scientific_sync_event_count": int(manifest["scientific_sync_event_count"]),
        }
    cpost = profile_rows["P2_CPOST_OFF"]["latency"]
    factorized = profile_rows["P3_F_OFF"]["latency"]
    return {
        "run_label": label,
        "root": str(root),
        "profiles": profile_rows,
        "ratios": {
            "median_wall_ratio": factorized["median_seconds"] / cpost["median_seconds"],
            "max_wall_ratio": factorized["max_seconds"] / cpost["max_seconds"],
            "trace_summed_kernel_ratio": (
                profile_rows["P3_F_OFF"]["trace"]["summed_kernel_seconds"]
                / profile_rows["P2_CPOST_OFF"]["trace"]["summed_kernel_seconds"]
            ),
            "trace_scientific_scope_ratio": (
                profile_rows["P3_F_OFF"]["trace"]["scientific_scope_wall_seconds_with_profiler"]
                / profile_rows["P2_CPOST_OFF"]["trace"]["scientific_scope_wall_seconds_with_profiler"]
            ),
        },
        "profile_order": ["P2_CPOST_OFF", "P3_F_OFF", "P4_F_REPLICATED", "P5_F_ON", "P6_DECODE4"],
        "legacy_warmups": 1,
        "legacy_measurements": 7,
        "accuracy_or_correctness_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2k_historical_timing_audit_v1",
        "status": "DESCRIPTIVE_ONLY",
        "runs": {label: audit_run(label, root) for label, root in RUNS.items()},
        "hot_path_blob_identity": {
            "r2i_scientific_commit": "8d21c72",
            "r2j_scientific_commit": "efa02fba98adff2a891445c4908a8dc9ac8c7fff",
            "rosetta/model/fpct_attention.py": "be395083cacfcfddf635b823e34a1020c1286659",
            "rosetta/model/wrapper.py": "5890df52d2c882d37467fa54f977102b9e12c282",
            "script/experiment/fpct_gpu_r2_runner.py": "41dc22dd08d5b49f78340ef25b4f65051f9ee234",
        },
        "interpretation": "Historical profiles lack telemetry needed for an infrastructure attribution. The audit cannot alter the permanent R2j failure.",
        "r2j_terminal_unchanged": "GPU_ENGINEERING_BLOCKED_R2",
        "accuracy_or_correctness_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
