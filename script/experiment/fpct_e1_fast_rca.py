#!/usr/bin/env python3
"""Compact fixed-checkpoint FPCT-E1 root-cause audit.

This executor deliberately never constructs the historical 30M-row Cartesian
table.  It consumes the frozen 326-group compact sidecar, runs full-response
teacher forcing, writes one row per sample/intervention, and reduces mechanism
statistics online at sample/layer/head granularity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


PROTOCOL_ID = "fpct_e1_fast_rca_v1"
TASKS = ("ai2-arc", "openbookqa", "mmlu-redux")
TASK_COUNTS = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
VARIANTS = (
    "c_post",
    "f",
    "lambda_0",
    "lambda_025",
    "lambda_05",
    "lambda_2",
    "k_only_v_collapse",
    "parent_mass_preserving",
    "partition_composition",
)
LAMBDA_BY_VARIANT = {
    "f": 1.0,
    "lambda_0": 0.0,
    "lambda_025": 0.25,
    "lambda_05": 0.5,
    "lambda_2": 2.0,
    "k_only_v_collapse": 1.0,
    "parent_mass_preserving": 1.0,
}
INTERVENTION_BY_VARIANT = {
    "k_only_v_collapse": "k_only_v_collapse",
    "parent_mass_preserving": "parent_mass_preserving",
}
MANIFEST_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1_fast_rca/fast_rca_manifest.json"
)


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_head(repo: Path) -> str:
    injected = os.environ.get("FPCT_EXECUTION_SHA")
    if injected is not None and re.fullmatch(r"[0-9a-f]{40}", injected) is None:
        raise ValueError("FPCT_EXECUTION_SHA is not a full lowercase Git SHA")
    try:
        observed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        if injected is None:
            raise RuntimeError(
                "runtime has no usable Git binary and no injected execution SHA"
            )
        return injected
    if injected is not None and observed != injected:
        raise ValueError("Git HEAD differs from injected execution SHA")
    return observed


def read_manifest(repo: Path) -> dict[str, Any]:
    value = json.loads((repo / MANIFEST_RELATIVE).read_text(encoding="utf-8"))
    if value.get("protocol_id") != PROTOCOL_ID or value.get("schema_version") != 1:
        raise ValueError("E1-FAST-RCA manifest identity mismatch")
    if tuple(value.get("tasks", {}).get("order", ())) != TASKS:
        raise ValueError("task order differs from the prospective contract")
    if value.get("tasks", {}).get("counts") != TASK_COUNTS:
        raise ValueError("task counts differ from the 326-group contract")
    if tuple(value.get("interventions", {}).get("execution_order", ())) != VARIANTS:
        raise ValueError("intervention order differs from the prospective contract")
    if value.get("firewall", {}).get("e1_pilot") != "SEALED_NOT_RUN_NOT_READ":
        raise ValueError("E1-pilot firewall is not sealed")
    return value


def compact_sidecar_items(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    import torch

    record = manifest["compact_input"]
    path = Path(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("compact sidecar SHA256 mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("split_role") != "e0_design":
        raise ValueError("compact sidecar is not E0-design")
    if payload.get("e1_pilot_consumed") is not False:
        raise ValueError("compact sidecar reports E1-pilot consumption")
    if payload.get("model_or_checkpoint_loaded") is not False:
        raise ValueError("compact sidecar is not model-output-free")
    items = payload.get("items")
    if not isinstance(items, list) or len(items) != sum(TASK_COUNTS.values()):
        raise ValueError("compact sidecar item count mismatch")
    counts = Counter(str(item.get("task")) for item in items)
    if dict(counts) != TASK_COUNTS:
        raise ValueError("compact sidecar task counts mismatch")
    if len({item["content_group_sha256"] for item in items}) != len(items):
        raise ValueError("compact sidecar contains duplicate content groups")
    for item in items:
        if not item.get("answer_queries") or not item.get("certified_parents"):
            raise ValueError("compact sidecar item lacks response/support geometry")
        if any(
            parent.get("topology") != "partition_compositional"
            for parent in item["certified_parents"]
        ):
            raise ValueError("unexpected non-partition certified topology")
    return sorted(items, key=lambda row: (TASKS.index(row["task"]), row["sample_sha256"]))


def overlap_composition_weights(item: Mapping[str, Any]):
    """Return a copied feature with partition rows weighted by covered length."""

    import torch

    feature = copy.deepcopy(item["feature"])
    weights = feature["soft_alignment"]["source_weights"].clone()
    indices = feature["soft_alignment"]["source_indices"]
    for row in item["raw_topology_ledger"]:
        if row.get("taxonomy") != "partition_compositional" or not row.get("certified"):
            continue
        parent = int(row["parent_position"])
        lengths = {
            int(candidate["source_index"]): int(candidate["intersection_length"])
            for candidate in row["candidates"]
            if candidate.get("runtime_retained") is True
        }
        total = sum(lengths.values())
        if total <= 0:
            raise ValueError("partition composition has no positive covered length")
        for slot in range(indices.shape[1]):
            source = int(indices[parent, slot])
            weights[parent, slot] = float(lengths.get(source, 0)) / float(total)
        if not torch.isclose(weights[parent].sum(), torch.tensor(1.0)):
            raise ValueError("partition composition weights do not sum to one")
    feature["soft_alignment"]["source_weights"] = weights
    return feature


def _checkpoint_by_id(manifest: Mapping[str, Any], checkpoint_id: str) -> Mapping[str, Any]:
    values = {
        value["checkpoint_id"]: value for value in manifest["checkpoints"]
    }
    if checkpoint_id not in values:
        raise KeyError(f"unknown checkpoint: {checkpoint_id}")
    return values[checkpoint_id]


def prepare_plan(repo: Path, output_root: Path) -> dict[str, Any]:
    from script.experiment.fpct_e1_capture_runner import _tree_manifest

    manifest = read_manifest(repo)
    items = compact_sidecar_items(manifest)
    execution_sha = git_head(repo)
    checkpoints = []
    for frozen in manifest["checkpoints"]:
        observed = _tree_manifest(Path(frozen["path"]))
        if (
            observed["tree_sha256"] != frozen["tree_sha256"]
            or observed["file_count"] != frozen["file_count"]
            or observed["bytes"] != frozen["bytes"]
        ):
            raise ValueError(f"checkpoint tree mismatch: {frozen['checkpoint_id']}")
        checkpoints.append({**dict(frozen), "verified": True})
    shards = []
    for checkpoint in checkpoints:
        for task in TASKS:
            shard_id = f"{checkpoint['checkpoint_id']}--{task}"
            shards.append(
                {
                    "shard_id": shard_id,
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "task": task,
                    "sample_count": TASK_COUNTS[task],
                    "variants": list(VARIANTS),
                    "output_relative": f"shards/{shard_id}",
                }
            )
    plan = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "execution_sha": execution_sha,
        "manifest_sha256": sha256_file(repo / MANIFEST_RELATIVE),
        "compact_input_sha256": manifest["compact_input"]["sha256"],
        "population_sha256": hashlib.sha256(
            canonical_json_bytes(
                [
                    (item["task"], item["sample_sha256"], item["content_group_sha256"])
                    for item in items
                ]
            )
        ).hexdigest(),
        "output_root": str(output_root.resolve()),
        "checkpoints": checkpoints,
        "shards": shards,
        "firewall": dict(manifest["firewall"]),
    }
    plan["plan_sha256"] = hashlib.sha256(canonical_json_bytes(plan)).hexdigest()
    output_root.mkdir(parents=True, exist_ok=False)
    atomic_write(output_root / "plan.json", canonical_json_bytes(plan))
    return plan


def _set_variant(model: Any, variant: str) -> None:
    model.set_fpct_diagnostic_intervention("none")
    if variant == "c_post":
        model.set_fpct_centered_lambda(1.0)
        model.fpct_operator = "c_post"
        return
    if variant == "partition_composition":
        model.set_fpct_centered_lambda(1.0)
        model.fpct_operator = "c_pre"
        return
    model.fpct_operator = "f"
    model.set_fpct_centered_lambda(LAMBDA_BY_VARIANT[variant])
    model.set_fpct_diagnostic_intervention(
        INTERVENTION_BY_VARIANT.get(variant, "none")
    )


def _gold_logp(outputs: Any, labels: Any) -> dict[str, Any]:
    import torch

    shifted = labels[:, 1:]
    eligible = shifted != -100
    batch, query = torch.where(eligible)
    token = shifted[batch, query]
    values = torch.log_softmax(outputs.logits[:, :-1].float(), dim=-1)[
        batch, query, token
    ]
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise RuntimeError("teacher-forced gold log-probability is empty/nonfinite")
    return {
        "answer_token_count": int(values.numel()),
        "gold_logp_sum": float(values.sum().cpu()),
        "gold_logp_mean": float(values.mean().cpu()),
    }


def _compact_capture(report: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "metrics": report.get("metrics", {}),
        "metric_statistics": report.get("metric_statistics", {}),
        "long_form_row_count": report.get("long_form_row_count"),
        "expanded_rows_materialized": "long_form_primitives" in report,
    }


def _merge_moment(
    bucket: dict[tuple[str, int, int, str], dict[str, float]],
    key: tuple[str, int, int, str],
    statistics: Mapping[str, Any],
) -> None:
    count = int(statistics.get("count", 0))
    if count <= 0:
        return
    value = bucket.setdefault(
        key,
        {
            "count": 0.0,
            "sum": 0.0,
            "square_sum": 0.0,
            "min": math.inf,
            "max": -math.inf,
        },
    )
    observed_mean = float(statistics["mean"])
    observed_variance = float(statistics.get("variance", 0.0))
    value["count"] += count
    value["sum"] += count * observed_mean
    value["square_sum"] += count * (
        observed_variance + observed_mean * observed_mean
    )
    value["min"] = min(value["min"], float(statistics.get("min", observed_mean)))
    value["max"] = max(value["max"], float(statistics.get("max", observed_mean)))


def _accumulate_layer_head(
    bucket: dict[tuple[str, int, int, str], dict[str, float]],
    report: Mapping[str, Any],
    variant: str,
) -> None:
    for layer_text, layer in report.get("layers", {}).items():
        layer_index = int(layer_text)
        for metric, statistics in layer.get("metric_statistics", {}).items():
            _merge_moment(bucket, (variant, layer_index, -1, metric), statistics)
        for head in layer.get("heads", []):
            head_index = int(head["head_index"])
            for metric, statistics in head.get("metric_statistics", {}).items():
                _merge_moment(
                    bucket,
                    (variant, layer_index, head_index, metric),
                    statistics,
                )
            _merge_moment(
                bucket,
                (variant, layer_index, head_index, "gamma_query_variance"),
                {
                    "count": 1,
                    "mean": float(head.get("gamma_query_variance", 0.0)),
                    "variance": 0.0,
                },
            )


def _finalize_layer_head(
    bucket: Mapping[tuple[str, int, int, str], Mapping[str, float]],
) -> list[dict[str, Any]]:
    rows = []
    for (variant, layer, head, metric), value in sorted(bucket.items()):
        count = int(value["count"])
        average = value["sum"] / count
        variance = max(value["square_sum"] / count - average * average, 0.0)
        rows.append(
            {
                "variant": variant,
                "layer": layer,
                "head": head,
                "metric": metric,
                "count": count,
                "mean": average,
                "variance": variance,
                "min": value["min"],
                "max": value["max"],
            }
        )
    return rows


def run_shard(
    repo: Path,
    plan_path: Path,
    shard_id: str,
    device_name: str,
    *,
    smoke_samples: int | None = None,
) -> dict[str, Any]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch
    import yaml
    from transformers import AutoTokenizer
    from rosetta.train.dataset_adapters import RosettaDataCollator
    from rosetta.utils.evaluate import load_rosetta_model, set_default_chat_template
    from rosetta.utils.model_loading import resolve_model_path

    manifest = read_manifest(repo)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("execution_sha") != git_head(repo):
        raise ValueError("runtime source HEAD differs from frozen execution SHA")
    shard = next(value for value in plan["shards"] if value["shard_id"] == shard_id)
    checkpoint = _checkpoint_by_id(manifest, shard["checkpoint_id"])
    task = shard["task"]
    items = [item for item in compact_sidecar_items(manifest) if item["task"] == task]
    smoke = smoke_samples is not None
    if smoke:
        if smoke_samples != 1:
            raise ValueError("pre-registered smoke must contain exactly one sample")
        items = items[:1]
        output = Path(plan["output_root"]) / "smoke" / f"{shard_id}--n1"
    else:
        output = Path(plan["output_root"]) / shard["output_relative"]
    if (output / "receipt.json").exists():
        return json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(f"incomplete shard output requires forensic review: {output}")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    temporary.mkdir(parents=True, exist_ok=False)

    config_path = repo / checkpoint["config_by_task"][task]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["model"]["rosetta_config"]["checkpoints_dir"] = checkpoint["path"]
    config["model"]["rosetta_config"]["fpct_operator"] = "f"
    config["model"]["rosetta_config"]["fpct_instrumentation"] = True
    config["model"]["rosetta_config"]["projector_load_mode"] = "strict_attested"
    config["eval"]["gpu_ids"] = [0]
    device = torch.device(device_name)
    model, receiver_tokenizer = load_rosetta_model(
        config["model"], config["eval"], device,
        config["model"].get("generation_config", {}),
    )
    model.eval()
    sender_name = config["model"]["rosetta_config"]["teacher_model"]
    sender_tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(sender_name))
    set_default_chat_template(sender_tokenizer, sender_name)
    collator = RosettaDataCollator(
        receiver_tokenizer,
        sender_tokenizer,
        max_length=32768,
        aligner=None,
        do_alignment=False,
    )
    rows_path = temporary / "sample_metrics.jsonl"
    layer_head_moments: dict[tuple[str, int, int, str], dict[str, float]] = {}
    with rows_path.open("x", encoding="utf-8") as handle:
        for item in items:
            for variant in VARIANTS:
                _set_variant(model, variant)
                feature = (
                    overlap_composition_weights(item)
                    if variant == "partition_composition"
                    else item["feature"]
                )
                batch = collator([feature])
                labels = batch.pop("labels").to(device)
                batch = {
                    key: (
                        [child.to(device) for child in value]
                        if isinstance(value, list)
                        else value.to(device) if hasattr(value, "to") else value
                    )
                    for key, value in batch.items()
                }
                capture = variant != "partition_composition"
                if capture:
                    query_mask = model.fpct_teacher_forced_query_mask(labels)
                    model.begin_fpct_capture(
                        mode="teacher_forced_response",
                        metadata={
                            "protocol_id": PROTOCOL_ID,
                            "variant": variant,
                            "sample_sha256": item["sample_sha256"],
                        },
                        query_mask=query_mask,
                        detail_mode="summary_only",
                    )
                with torch.no_grad():
                    outputs = model(**batch, labels=labels, use_cache=True)
                report = model.end_fpct_capture() if capture else None
                if report is not None:
                    _accumulate_layer_head(layer_head_moments, report, variant)
                row = {
                    "schema_version": 1,
                    "protocol_id": PROTOCOL_ID,
                    "execution_sha": plan["execution_sha"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "seed": checkpoint["seed"],
                    "checkpoint_arm": checkpoint["checkpoint_arm"],
                    "task": task,
                    "sample_sha256": item["sample_sha256"],
                    "content_group_sha256": item["content_group_sha256"],
                    "variant": variant,
                    "topology": "partition_compositional",
                    **_gold_logp(outputs, labels),
                    "capture": _compact_capture(report) if report is not None else None,
                    "mechanism_metric_semantics": (
                        "counterfactual_global_f_diagnostics"
                        if variant == "parent_mass_preserving"
                        else "actual_operator"
                    ),
                }
                handle.write(canonical_json_bytes(row).decode("utf-8"))
                handle.flush()
                del outputs, labels, batch
    layer_head_path = temporary / "layer_head_summary.json"
    atomic_write(
        layer_head_path,
        canonical_json_bytes(
            {
                "schema_version": 1,
                "protocol_id": PROTOCOL_ID,
                "rows": _finalize_layer_head(layer_head_moments),
            }
        ),
    )
    receipt = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "SMOKE_COMPLETE_NOT_SCIENTIFIC" if smoke else "COMPLETE",
        "execution_sha": plan["execution_sha"],
        "plan_sha256": plan["plan_sha256"],
        "shard_id": shard_id,
        "checkpoint_id": checkpoint["checkpoint_id"],
        "task": task,
        "sample_count": len(items),
        "row_count": len(items) * len(VARIANTS),
        "sample_metrics_sha256": sha256_file(rows_path),
        "layer_head_summary_sha256": sha256_file(layer_head_path),
        "expanded_rows_materialized": False,
        "e1_pilot_consumed": False,
        "model_selection_consumed": False,
        "test_consumed": False,
        "scientific_analysis_eligible": not smoke,
    }
    atomic_write(temporary / "receipt.json", canonical_json_bytes(receipt))
    os.replace(temporary, output)
    return receipt


def verify_outputs(plan_path: Path) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    receipts = []
    for shard in plan["shards"]:
        root = Path(plan["output_root"]) / shard["output_relative"]
        receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        rows = root / "sample_metrics.jsonl"
        if sha256_file(rows) != receipt["sample_metrics_sha256"]:
            raise ValueError(f"shard hash mismatch: {shard['shard_id']}")
        layer_head = root / "layer_head_summary.json"
        if sha256_file(layer_head) != receipt["layer_head_summary_sha256"]:
            raise ValueError(f"layer/head hash mismatch: {shard['shard_id']}")
        count = sum(1 for _ in rows.open("r", encoding="utf-8"))
        if count != receipt["row_count"] or receipt["expanded_rows_materialized"]:
            raise ValueError(f"shard row/representation mismatch: {shard['shard_id']}")
        receipts.append(receipt)
    return {
        "status": "GO_ALL_SHARDS_COMPLETE",
        "shard_count": len(receipts),
        "row_count": sum(value["row_count"] for value in receipts),
        "expanded_rows_materialized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output-root", type=Path, required=True)
    run = sub.add_parser("run-shard")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--shard-id", required=True)
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--smoke-samples", type=int)
    verify = sub.add_parser("verify")
    verify.add_argument("--plan", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    if args.command == "prepare":
        result = prepare_plan(repo, args.output_root)
    elif args.command == "run-shard":
        result = run_shard(
            repo,
            args.plan,
            args.shard_id,
            args.device,
            smoke_samples=args.smoke_samples,
        )
    else:
        result = verify_outputs(args.plan)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
