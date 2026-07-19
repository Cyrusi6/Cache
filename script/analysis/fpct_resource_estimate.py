from __future__ import annotations

"""CPU-only FPCT cache/attention resource estimator from frozen FPCT-1B rows."""

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
EXECUTION_SHA = "7f8af71968a39bc6cba2e4e34de762b291cda834"
DEFAULT_AUDIT_ROOT = REPO_ROOT / (
    "local/final_results/fpct_factorized_transport/"
    f"fpct_1b_ambiguity_support/rev_{EXECUTION_SHA}"
)
RECEIVER_DIR = Path(
    "/netdisk/lijunsi/c2c-route1-identifiability/models/Qwen3-0.6B"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.quantile(array, 0.50, method="linear")),
        "p90": float(np.quantile(array, 0.90, method="linear")),
        "p95": float(np.quantile(array, 0.95, method="linear")),
        "max": float(array.max()),
    }


def receiver_token_counts() -> dict[str, int]:
    from transformers import AutoTokenizer
    from script.analysis.fpct_1b_structural_support_audit import (
        DEFAULT_SHARED_ROOT,
        load_projected_samples,
        prompt_for_sample,
        validate_canonical_samples,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        RECEIVER_DIR, local_files_only=True, use_fast=True
    )
    samples = load_projected_samples(DEFAULT_SHARED_ROOT)
    validate_canonical_samples(samples)
    counts: dict[str, int] = {}
    for sample in samples:
        prompt = prompt_for_sample(sample)
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        count = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        counts[sample.sample_key_sha256] = count
    if len(counts) != 7265:
        raise ValueError(f"expected 7265 receiver token counts, got {len(counts)}")
    return counts


def load_parent_counts(path: Path) -> dict[tuple[str, str, str], list[int]]:
    counts: dict[tuple[str, str, str], list[int]] = defaultdict(
        lambda: [0, 0, 0, 0, 0]
    )
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            m = int(row["legal_candidate_count"])
            if not 0 <= m <= 4:
                raise ValueError(f"invalid candidate count: {m}")
            counts[(row["pair"], row["task"], row["sample_key_sha256"])][m] += 1
    if len(counts) != 4 * 7265:
        raise ValueError(f"expected 29060 pair/task/sample rows, got {len(counts)}")
    return counts


def estimate(audit_root: Path) -> dict[str, Any]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError('CUDA_VISIBLE_DEVICES must be explicitly set to ""')
    config_path = RECEIVER_DIR / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    layers = int(config["num_hidden_layers"])
    hkv = int(config["num_key_value_heads"])
    hq = int(config["num_attention_heads"])
    head_dim = int(config.get("head_dim", config["hidden_size"] // hq))
    dtype_bytes = 2  # frozen receiver config is bfloat16
    kv_atom_bytes = 2 * hkv * head_dim * dtype_bytes

    token_counts = receiver_token_counts()
    parent_counts = load_parent_counts(audit_root / "parent_support.csv")
    by_pair_task: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for (pair, task, sample_hash), counts in parent_counts.items():
        native_slots = token_counts[sample_hash]
        eligible = sum(counts)
        extra_slots = sum(max(m - 1, 0) * counts[m] for m in range(5))
        child_atoms = sum(m * counts[m] for m in range(2, 5))
        expanded_slots = native_slots + extra_slots
        dense_topk4_slots = native_slots + 3 * eligible
        by_pair_task[(pair, task)].append(
            {
                "native_slots": float(native_slots),
                "eligible_parents": float(eligible),
                "extra_slots": float(extra_slots),
                "expanded_slots": float(expanded_slots),
                "expansion_ratio": expanded_slots / native_slots,
                "dense_topk4_ratio": dense_topk4_slots / native_slots,
                "ambiguous_child_atoms": float(child_atoms),
                "dense_sidecar_bytes_per_layer": float(native_slots * 4 * kv_atom_bytes),
                "ambiguous_sidecar_bytes_per_layer": float(child_atoms * kv_atom_bytes),
                "expanded_cache_increment_bytes_per_layer": float(extra_slots * kv_atom_bytes),
            }
        )

    rows = []
    for pair in ("tinyllama", "qwen25_0p5b", "llama32_1b", "qwen3_1p7b"):
        for task in ("ai2-arc", "openbookqa", "mmlu-redux"):
            samples = by_pair_task[(pair, task)]
            expansion = quantiles([row["expansion_ratio"] for row in samples])
            dense = quantiles([row["dense_topk4_ratio"] for row in samples])
            mean_dense_sidecar = float(
                np.mean([row["dense_sidecar_bytes_per_layer"] for row in samples])
            )
            mean_ambiguous_sidecar = float(
                np.mean([row["ambiguous_sidecar_bytes_per_layer"] for row in samples])
            )
            mean_increment = float(
                np.mean(
                    [row["expanded_cache_increment_bytes_per_layer"] for row in samples]
                )
            )
            rows.append(
                {
                    "pair": pair,
                    "task": task,
                    "sample_count": len(samples),
                    "expansion_ratio": expansion,
                    "attention_score_flop_ratio": expansion,
                    "dense_topk4_attention_ratio": dense,
                    "mean_dense_topk4_sidecar_bytes_per_layer": mean_dense_sidecar,
                    "mean_ambiguous_only_sidecar_bytes_per_layer": mean_ambiguous_sidecar,
                    "mean_expanded_cache_increment_bytes_per_layer": mean_increment,
                    "mean_dense_topk4_sidecar_bytes_all_layers": mean_dense_sidecar
                    * layers,
                    "mean_ambiguous_only_sidecar_bytes_all_layers": mean_ambiguous_sidecar
                    * layers,
                    "mean_expanded_cache_increment_bytes_all_layers": mean_increment
                    * layers,
                }
            )
    return {
        "schema_version": 1,
        "execution_sha": EXECUTION_SHA,
        "audit_parent_support_sha256": sha256_file(audit_root / "parent_support.csv"),
        "receiver_config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "num_hidden_layers": layers,
            "num_attention_heads": hq,
            "num_key_value_heads": hkv,
            "head_dim": head_dim,
            "dtype": config.get("torch_dtype"),
            "dtype_bytes": dtype_bytes,
            "kv_atom_bytes_per_layer": kv_atom_bytes,
        },
        "definitions": {
            "extra_slots": "sum_i max(m_i-1,0)",
            "expanded_slots": "receiver_native_slots + extra_slots",
            "dense_topk4_slots": "receiver_native_slots + 3 * eligible_parent_count",
            "ambiguous_only_sidecar_atoms": "sum_i m_i * 1[m_i>=2]",
        },
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = estimate(args.audit_root)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "COMPLETE", "output": str(args.output), "rows": len(result["rows"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
