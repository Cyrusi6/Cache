#!/usr/bin/env python3
"""R2k diagnostic-only paired latency runner and immutable-input aggregator."""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, record_function

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    build_fpct_packed_layout,
    fpct_qwen_eager_attention_forward,
    fpct_qwen_hierarchical_attention_forward,
    pack_fpct_memory,
)
from script.experiment import fpct_gpu_r2_runner as r2


BLOCK_ORDER = (
    ("c_post", "f"),
    ("f", "c_post"),
    ("f", "c_post"),
    ("c_post", "f"),
    ("c_post", "f"),
    ("f", "c_post"),
    ("f", "c_post"),
    ("c_post", "f"),
)
PROCESS_SEEDS = tuple(range(104729, 104737))
WARMUPS = 20
MEASUREMENTS = 50


class R2kDiagnosticError(RuntimeError):
    pass


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _command_output(command: list[str]) -> str:
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True, timeout=10
    )
    return completed.stdout.strip() if completed.returncode == 0 else f"unavailable:{completed.stderr.strip()}"


def telemetry() -> dict[str, Any]:
    query = (
        "uuid,name,pstate,clocks.current.sm,clocks.current.memory,temperature.gpu,"
        "power.draw,power.limit,clocks_throttle_reasons.active"
    )
    affinity = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    return {
        "gpu": _command_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
        ),
        "compute_processes": _command_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        ),
        "cpu_affinity": affinity,
        "cpu_scheduler": _command_output(["chrt", "-p", str(os.getpid())]),
        "node_name": os.environ.get("NODE_NAME", "not_provided"),
        "pod_name": os.environ.get("POD_NAME", "not_provided"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "not_set"),
        "timestamp_ns": time.time_ns(),
    }


def measure_cuda(forward: Callable[[], Any], *, steps: int = 1) -> dict[str, Any]:
    for _ in range(WARMUPS):
        for _step in range(steps):
            forward()
    torch.cuda.synchronize()
    cuda_seconds: list[float] = []
    wall_seconds: list[float] = []
    for _ in range(MEASUREMENTS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        wall_start = time.perf_counter()
        start.record()
        for _step in range(steps):
            forward()
        end.record()
        end.synchronize()
        wall_seconds.append(time.perf_counter() - wall_start)
        cuda_seconds.append(start.elapsed_time(end) / 1000.0)
    return {
        "warmups": WARMUPS,
        "measurements": MEASUREMENTS,
        "cuda_event_seconds": cuda_seconds,
        "synchronized_wall_seconds": wall_seconds,
        "cuda_median_seconds": statistics.median(cuda_seconds),
        "wall_median_seconds": statistics.median(wall_seconds),
    }


def _actual_arm(
    lock_path: Path,
    panel_path: Path,
    data_root: Path,
    operator: str,
    forced_on: bool,
    seed: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise R2kDiagnosticError("CUDA unavailable")
    lock = r2.load_json(lock_path)
    panel = r2.load_json(panel_path)
    row = r2._panel_records(panel, data_root)[0]
    condition = {"operator": operator, "forced": forced_on}
    config = r2._model_config(lock, condition)
    config.update(
        {
            "fpct_instrumentation": False,
            "fpct_profile_scopes": False,
            "fpct_trace": False,
        }
    )
    from script.train import SFT_train as sft

    device = torch.device("cuda:0")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model, receiver_tokenizer, aligner, sender_tokenizer = sft.setup_models(
        config, "rosetta", str(device), torch.bfloat16
    )
    model.eval()
    eager = r2._assert_eager(model)
    if forced_on:
        for projector in model.projector_list:
            projector.set_alignment_confidence_eval_mode("forced_on")
    collator = sft.RosettaDataCollator(
        slm_tokenizer=receiver_tokenizer,
        llm_tokenizer=sender_tokenizer,
        max_length=1024,
        aligner=aligner,
        do_alignment=True,
    )
    item = sft.AlignedChatDataset(
        [row["messages"]],
        aligner,
        max_length=1024,
        soft_alignment_top_k=4,
        fpct_alignment_sanitizer="certified_slot0_v1",
    )[0]
    batch = r2._device_batch(collator([item]), device)

    def forward():
        with torch.no_grad():
            return r2._forward(model, batch)

    before = telemetry()
    torch.cuda.reset_peak_memory_stats()
    timing = measure_cuda(forward)
    after = telemetry()
    layout = model._fpct_packed_layout
    expansion = None
    if layout is not None:
        expansion = (
            layout.expanded_slots.detach().float() / float(layout.source_length)
        ).cpu().tolist()
    return {
        "operator": operator,
        "forced_on": forced_on,
        "seed": seed,
        "timing": timing,
        "peak_hbm_gib": torch.cuda.max_memory_allocated() / 2**30,
        "expanded_slot_ratios": expansion,
        "eager_attestation": eager,
        "telemetry_before": before,
        "telemetry_after": after,
        "accuracy_or_correctness_accessed": False,
    }


def run_block(args: argparse.Namespace) -> dict[str, Any]:
    block_index = int(args.block_index)
    if block_index < 0 or block_index >= len(BLOCK_ORDER):
        raise R2kDiagnosticError("block index must be in [0,7]")
    output = Path(args.output)
    if output.exists():
        raise R2kDiagnosticError(f"diagnostic block already exists: {output}")
    seed = PROCESS_SEEDS[block_index]
    order = BLOCK_ORDER[block_index]
    canaries: dict[str, Any] = {}
    for forced_on in (False, True):
        label = "forced_on" if forced_on else "checkpoint_native"
        arms = []
        for operator in order:
            arms.append(
                _actual_arm(
                    Path(args.run_lock),
                    Path(args.panel),
                    Path(args.data_root),
                    operator,
                    forced_on,
                    seed,
                )
            )
            torch.cuda.empty_cache()
        canaries[label] = {"order": list(order), "arms": arms}
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2k_diagnostic_block_v1",
        "classification": "DIAGNOSTIC_ONLY",
        "block_index": block_index,
        "process_seed": seed,
        "order": list(order),
        "canaries": canaries,
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output, payload)
    return payload


def _geometry_memory(row: dict[str, Any], device: torch.device):
    b, hq, hkv, q_length, parents, dimension = (
        int(row["batch"]),
        int(row["heads"]),
        int(row["kv_heads"]),
        int(row["query"]),
        int(row["parents"]),
        int(row["head_dim"]),
    )
    cardinality = int(row["cardinality"])
    generator = torch.Generator(device=device).manual_seed(104729 + parents + cardinality)
    query = torch.randn(b, hq, q_length, dimension, device=device, dtype=torch.bfloat16, generator=generator)
    parent_key = torch.randn(b, hkv, parents, dimension, device=device, dtype=torch.bfloat16, generator=generator)
    parent_value = torch.randn(
        parent_key.shape,
        device=device,
        dtype=torch.bfloat16,
        generator=generator,
    )
    candidate_key = parent_key[:, :, :, None, :].expand(-1, -1, -1, 4, -1).clone()
    candidate_value = parent_value[:, :, :, None, :].expand_as(candidate_key).clone()
    prior = torch.zeros(b, parents, 4, device=device, dtype=torch.float32)
    prior[..., 0] = 1.0
    remaining_extra = int(row["expanded_slots"]) - parents
    parent_index = 0
    while remaining_extra > 0:
        parent_cardinality = min(cardinality, remaining_extra + 1)
        prior[:, parent_index] = 0
        prior[:, parent_index, :parent_cardinality] = 1.0 / parent_cardinality
        candidate_key[:, :, parent_index, :parent_cardinality] += torch.randn(
            b,
            hkv,
            parent_cardinality,
            dimension,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ) * 0.05
        candidate_value[:, :, parent_index, :parent_cardinality] += torch.randn(
            b,
            hkv,
            parent_cardinality,
            dimension,
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        ) * 0.05
        remaining_extra -= parent_cardinality - 1
        parent_index += 1
    valid = prior > 0
    weights = prior[:, None, :, :, None]
    collapsed_key = (candidate_key.float() * weights).sum(dim=3).to(torch.bfloat16)
    collapsed_value = (candidate_value.float() * weights).sum(dim=3).to(torch.bfloat16)
    sidecar = FPCTSidecarSegment(0, candidate_key, candidate_value, prior, valid)
    layout = build_fpct_packed_layout(parents, [sidecar], max_slots_hint=int(row["expanded_slots"]))
    packed = pack_fpct_memory(
        collapsed_key,
        collapsed_value,
        None,
        [sidecar],
        query_length=q_length,
        layout=layout,
    )
    module = SimpleNamespace(num_key_value_groups=hq // hkv, training=False)
    return query, collapsed_key, collapsed_value, packed, module


def run_geometry(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise R2kDiagnosticError("CUDA unavailable")
    panel = json.loads(Path(args.panel).read_text())
    output = Path(args.output)
    if output.exists():
        raise R2kDiagnosticError(f"geometry output already exists: {output}")
    rows = []
    for row in panel["rows"]:
        query, parent_key, parent_value, packed, module = _geometry_memory(
            row, torch.device("cuda:0")
        )
        scaling = float(query.shape[-1] ** -0.5)

        def cpost():
            with torch.no_grad():
                fpct_qwen_eager_attention_forward(
                    module, query, parent_key, parent_value, None, scaling=scaling
                )

        def factorized():
            with torch.no_grad():
                fpct_qwen_hierarchical_attention_forward(
                    module,
                    query,
                    packed,
                    parent_key,
                    parent_value,
                    None,
                    scaling=scaling,
                )

        cpost_timing = measure_cuda(cpost, steps=int(row["decode_steps"]))
        factorized_timing = measure_cuda(factorized, steps=int(row["decode_steps"]))
        rows.append(
            {
                **row,
                "c_post": cpost_timing,
                "f": factorized_timing,
                "cuda_median_ratio": factorized_timing["cuda_median_seconds"]
                / cpost_timing["cuda_median_seconds"],
                "wall_median_ratio": factorized_timing["wall_median_seconds"]
                / cpost_timing["wall_median_seconds"],
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2k_geometry_result_v1",
        "classification": "DIAGNOSTIC_ONLY",
        "rows": rows,
        "telemetry": telemetry(),
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output, payload)
    return payload


def _block_bootstrap_ucb(values: list[float]) -> float:
    generator = random.Random(20260722)
    replicates = []
    for _ in range(50000):
        sample = [values[generator.randrange(len(values))] for _ in values]
        replicates.append(statistics.median(sample))
    replicates.sort()
    return replicates[int(0.95 * (len(replicates) - 1))]


def run_aggregate(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root)
    blocks = [json.loads((root / f"block_{index:02d}.json").read_text()) for index in range(8)]
    canary_results = {}
    for label in ("checkpoint_native", "forced_on"):
        cuda_ratios = []
        wall_ratios = []
        for index, block in enumerate(blocks):
            if block["block_index"] != index or tuple(block["order"]) != BLOCK_ORDER[index]:
                raise R2kDiagnosticError(f"block provenance mismatch: {index}")
            by_operator = {row["operator"]: row for row in block["canaries"][label]["arms"]}
            cuda_ratios.append(
                by_operator["f"]["timing"]["cuda_median_seconds"]
                / by_operator["c_post"]["timing"]["cuda_median_seconds"]
            )
            wall_ratios.append(
                by_operator["f"]["timing"]["wall_median_seconds"]
                / by_operator["c_post"]["timing"]["wall_median_seconds"]
            )
        median_ratio = statistics.median(cuda_ratios)
        ucb = _block_bootstrap_ucb(cuda_ratios)
        canary_results[label] = {
            "cuda_block_ratios": cuda_ratios,
            "wall_block_ratios": wall_ratios,
            "cuda_balanced_median_ratio": median_ratio,
            "cuda_block_bootstrap_one_sided_95_ucb": ucb,
            "wall_balanced_median_ratio": statistics.median(wall_ratios),
            "qualified": median_ratio <= 1.35 and ucb <= 1.50,
        }
    qualified = all(row["qualified"] for row in canary_results.values())
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2k_diagnostic_aggregate_v1",
        "classification": "DIAGNOSTIC_ONLY",
        "status": "DIAGNOSTIC_QUALIFIED" if qualified else "DIAGNOSTIC_NOT_QUALIFIED",
        "canaries": canary_results,
        "may_produce_r2k_go": False,
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(Path(args.output), payload)
    return payload


def run_trace(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise R2kDiagnosticError("CUDA unavailable")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for operator in ("c_post", "f"):
        lock = r2.load_json(Path(args.run_lock))
        panel = r2.load_json(Path(args.panel))
        row = r2._panel_records(panel, Path(args.data_root))[0]
        condition = {"operator": operator, "forced": True}
        config = r2._model_config(lock, condition)
        config.update({"fpct_instrumentation": False, "fpct_profile_scopes": True, "fpct_trace": False})
        from script.train import SFT_train as sft

        device = torch.device("cuda:0")
        torch.manual_seed(104729)
        torch.cuda.manual_seed_all(104729)
        model, receiver_tokenizer, aligner, sender_tokenizer = sft.setup_models(
            config, "rosetta", str(device), torch.bfloat16
        )
        model.eval()
        for projector in model.projector_list:
            projector.set_alignment_confidence_eval_mode("forced_on")
        collator = sft.RosettaDataCollator(
            slm_tokenizer=receiver_tokenizer,
            llm_tokenizer=sender_tokenizer,
            max_length=1024,
            aligner=aligner,
            do_alignment=True,
        )
        item = sft.AlignedChatDataset(
            [row["messages"]], aligner, max_length=1024, soft_alignment_top_k=4,
            fpct_alignment_sanitizer="certified_slot0_v1",
        )[0]
        batch = r2._device_batch(collator([item]), device)
        with torch.no_grad():
            r2._forward(model, batch)
        trace_path = output_root / f"{operator}.chrome.json"
        with torch.profiler.profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_stack=True,
            record_shapes=True,
        ) as profile:
            with torch.no_grad(), record_function("scientific_forward"):
                r2._forward(model, batch)
        profile.export_chrome_trace(str(trace_path))
        rows.append(
            {
                "operator": operator,
                "trace": {
                    "path": str(trace_path),
                    "sha256": r2.sha256_file(trace_path),
                    "bytes": trace_path.stat().st_size,
                },
            }
        )
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2k_separate_trace_v1",
        "classification": "DIAGNOSTIC_ONLY",
        "rows": rows,
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output_root / "trace_manifest.json", payload)
    return payload


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    block = commands.add_parser("block")
    block.add_argument("--run-lock", required=True)
    block.add_argument("--panel", required=True)
    block.add_argument("--data-root", required=True)
    block.add_argument("--block-index", type=int, required=True)
    block.add_argument("--output", required=True)
    geometry = commands.add_parser("geometry")
    geometry.add_argument("--panel", required=True)
    geometry.add_argument("--output", required=True)
    aggregate = commands.add_parser("aggregate")
    aggregate.add_argument("--root", required=True)
    aggregate.add_argument("--output", required=True)
    trace = commands.add_parser("trace")
    trace.add_argument("--run-lock", required=True)
    trace.add_argument("--panel", required=True)
    trace.add_argument("--data-root", required=True)
    trace.add_argument("--output-root", required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    payload = {
        "block": run_block,
        "geometry": run_geometry,
        "aggregate": run_aggregate,
        "trace": run_trace,
    }[args.command](args)
    print(json.dumps({"status": payload.get("status", "COMPLETE"), "classification": "DIAGNOSTIC_ONLY"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
