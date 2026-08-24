#!/usr/bin/env python3
"""Materialize the frozen FPCT math-direct two-seed replication configs."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml


BASE_SEED = 2026082101
SEEDS = (2026082102, 2026082103)
ARMS = ("c_post", "f")
TASKS = ("ai2-arc", "openbookqa", "mmlu-redux")
TASKS_AND_GROUPS = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
CELLS = ("Y_CC", "Y_CF", "Y_FC", "Y_FF")
RUN_NAME = "math-direct-replication-20260824-v1"
CONTAINER_ROOT = Path("/workspace/Cache/local/fpct_math_direct_replication") / RUN_NAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_new(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"existing generated config differs: {path}")
        return
    path.write_text(payload, encoding="utf-8")


def generate(repo: Path, output: Path) -> list[Path]:
    source = repo / "recipe/eval_recipe/fpct_math_direct"
    subject_source = (
        repo
        / "recipe/eval_recipe/fpct_e0/rendered/"
        "eval_2026072201_Y_CC_mmlu-redux.yaml"
    )
    subjects = yaml.safe_load(subject_source.read_text(encoding="utf-8"))["eval"][
        "subjects"
    ]
    if len(subjects) != 49 or len(set(subjects)) != 49:
        raise ValueError("frozen MMLU subject allowlist must contain 49 unique entries")

    generated: list[Path] = []
    for seed in SEEDS:
        for arm in ARMS:
            base = json.loads(
                (source / f"train_{BASE_SEED}_{arm}.json").read_text(encoding="utf-8")
            )
            config = copy.deepcopy(base)
            config["training"]["seed"] = seed
            config["model"]["projector_init_seed"] = seed
            config["output"]["output_dir"] = str(CONTAINER_ROOT / "seeds" / str(seed) / arm)
            config["output"]["wandb_config"]["run_name"] = (
                f"fpct-math-direct-r2-{seed}-{arm}"
            )
            path = output / f"train_{seed}_{arm}.json"
            _write_new(path, json.dumps(config, indent=2, sort_keys=True) + "\n")
            generated.append(path)

        for cell in CELLS:
            trained = "c_post" if cell in {"Y_CC", "Y_CF"} else "f"
            for task in TASKS:
                base = yaml.safe_load(
                    (source / f"eval_{BASE_SEED}_{cell}_{task}.yaml").read_text(
                        encoding="utf-8"
                    )
                )
                config = copy.deepcopy(base)
                config["model"]["rosetta_config"]["checkpoints_dir"] = str(
                    CONTAINER_ROOT / "seeds" / str(seed) / trained / "final"
                )
                config["output"]["output_dir"] = str(
                    CONTAINER_ROOT / "seeds" / str(seed) / "eval" / cell / task
                )
                if task == "mmlu-redux":
                    config["eval"]["subjects"] = list(subjects)
                path = output / f"eval_{seed}_{cell}_{task}.yaml"
                _write_new(path, yaml.safe_dump(config, sort_keys=True))
                generated.append(path)
    return generated


def build_manifest(repo: Path, output: Path, configs: list[Path]) -> Path:
    manifest = {
        "schema_version": 1,
        "protocol_id": "fpct_math_direct_replication_v1",
        "status": "PRE_OUTPUT_LOCK",
        "classification": "EXPLORATORY_THREE_SEED_REPLICATION_NOT_CONFIRMATORY",
        "execution_commit": "CLEAN_PUSHED_HEAD_CAPTURED_AT_RUNTIME",
        "parent_result_commit": "1e46734eaf433fd61c6cb036544a9766fd6e36c8",
        "immutable_existing_seed": 2026082101,
        "new_seeds": list(SEEDS),
        "pair": "TinyLlama-1.1B-Chat-v1.0__to__Qwen3-0.6B",
        "operator": {
            "position_mode": "math",
            "arms": list(ARMS),
            "headline": "F-C_post",
            "a": 1,
            "g": 1,
            "native_null": False,
            "selector": False,
            "attention_backend": "eager",
        },
        "training": {
            "examples": 2048,
            "epochs": 1,
            "optimizer_steps": 64,
            "processes": 2,
            "gpus_per_seed": 2,
            "per_device_batch_size": 1,
            "gradient_accumulation": 16,
            "effective_global_batch": 32,
            "learning_rate": 0.0001,
            "weight_decay": 0.01,
            "scheduler": "linear",
            "warmup_ratio": 0.10,
            "dropout": 0.1,
            "precision": "bf16",
            "arm_order": list(ARMS),
            "same_node_serial_within_seed": True,
            "parallel_across_seeds_allowed": True,
        },
        "evaluation": {
            "cells": ["Y_CC", "Y_CF", "Y_FC", "Y_FF"],
            "tasks_and_groups": TASKS_AND_GROUPS,
            "population": "E0-design_already_open",
            "primary": "accuracy_task_macro_equal_weight",
            "teacher_forced_diagnostic": "answer_token_mean_logp_task_macro",
        },
        "decision": {
            "system": {
                "mean_T_pp_min": 1.0,
                "positive_T_seeds_min": 2,
                "per_task_mean_T_pp_min": -2.0,
                "integrity_all_go": True,
            },
            "mechanism": {"mean_O_strictly_positive": True, "positive_O_seeds_min": 2},
            "classifications": [
                "QUERY_TIME_FPCT_CONTINUE",
                "FACTORIZED_TRAINING_COLLAPSED_INFERENCE",
                "STOP_CURRENT_OPERATOR",
            ],
        },
        "diagnostics": {
            "step0_gradient": {
                "seed": 2026082102,
                "sampler_rank": 0,
                "world_size_contract": 2,
                "optimizer_steps": 0,
                "groups": ["key_path", "value_path", "nuisance_gate_confidence"],
            },
            "teacher_forced": {"all_seeds": True, "all_four_cells": True},
        },
        "assets": {
            "training_sidecar": {
                "path": "/netdisk/lijunsi/fpct-math-direct-r5/assets/mmlu_auxiliary_train_2048_certified.pt",
                "sha256": "48caee80b31925a6074c9c5304bd861163f4e2e21adb55ebec9bf00237e2d990",
            },
            "teacher_forced_input": {
                "path": "/netdisk/lijunsi/fpct-e1/fpct-e1-a5r2-1c64b606-v1/input_lock/e0_design_input_lock.pt",
                "sha256": "d843512b7e446efd229fe3097030407e59acb20b4de591a84b88176bd8b0eec9",
            },
            "dev_manifest": {
                "path": "recipe/eval_recipe/fpct_e0/exploratory_dev_manifest.json",
                "sha256": "25fe8c4dceeaa1174e1433a02ec86f312d909d7d58c3c9a8c7f2caa9d908216a",
            },
        },
        "normative_files": {
            path: _sha256(repo / path)
            for path in (
                "FPCT_MATH_DIRECT_REPLICATION_PROTOCOL.md",
                "script/analysis/fpct_math_direct_replication_diagnostics.py",
                "script/analysis/fpct_math_direct_replication_results.py",
                "script/experiment/fpct_math_direct_replication_configs.py",
                "test/test_fpct_math_direct_replication.py",
            )
        },
        "configs": [
            {"path": str(path.relative_to(repo)), "sha256": _sha256(path)}
            for path in sorted(configs)
        ],
        "output_root": "/netdisk/lijunsi/fpct-math-direct-r2/math-direct-replication-20260824-v1",
        "firewall": {
            "e1_pilot": "SEALED_NOT_RUN_NOT_READ",
            "model_selection": "SEALED_NOT_RUN_NOT_READ",
            "test": "SEALED_NOT_RUN_NOT_READ",
            "confirmatory": "NOT_AUTHORIZED",
        },
    }
    path = output / "replication_manifest.json"
    _write_new(path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("recipe/eval_recipe/fpct_math_direct_r2"),
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    paths = generate(repo, output)
    manifest = build_manifest(repo, output, paths)
    print(
        json.dumps(
            {"generated": [str(path) for path in paths], "manifest": str(manifest)},
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
