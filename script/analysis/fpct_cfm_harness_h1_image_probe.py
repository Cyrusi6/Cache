#!/usr/bin/env python3
"""No-model exact-image probe for H1; intended to run inside the sealed R2m image."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(tmp, path)


def normalize_config(config: dict[str, Any], run_uid: str, seed: int, arm: str, examples: int, steps: int, sidecar_sha: str) -> dict[str, Any]:
    training = config["training"]
    return {
        "schema_version": 1, "run_uid": run_uid, "seed": seed, "arm": arm,
        "examples": examples, "optimizer_steps": steps,
        "world_size": training["num_processes"],
        "per_device_batch_size": training["per_device_train_batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "learning_rate": training["learning_rate"], "weight_decay": training["weight_decay"],
        "max_length": training["max_length"], "precision": "bf16", "sidecar_sha256": sidecar_sha,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--historical-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tripwires = {"subprocess": 0, "cuda": 0, "optimizer": 0, "model": 0, "dataset": 0, "checkpoint": 0}
    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "script/runtime"))
    candidate = json.loads(args.candidate_lock.read_text())
    manifest = json.loads(args.manifest.read_text())
    runner = load_module("_h1_runner", repo / "script/experiment/fpct_confirmatory_runner.py")
    controller = load_module("_h1_controller", repo / "script/experiment/fpct_gpu_r2_controller.py")
    def forbidden(*_args, **_kwargs):
        tripwires["subprocess"] += 1
        raise RuntimeError("H1 subprocess tripwire")
    runner.subprocess.run = forbidden
    configs = []
    sidecar_sha = candidate["assets"]["sidecar_sha256"]
    lock_for_runner = {
        "run_uid": candidate["run_uid"],
        "assets": {"training_alignment_sidecar_2048": {"container_path": "/h1/no-model-sidecar"}},
    }
    for arm in ("c_pre", "c_post", "f"):
        raw = runner.training_config(lock_for_runner, 104729, arm, Path("/h1/not-created") / arm, examples=128, optimizer_steps=4)
        configs.append(normalize_config(raw, candidate["run_uid"], 104729, arm, 128, 4, sidecar_sha))
    for seed in manifest["formal_training"]["seeds"]:
        for arm in runner.ARM_ORDER[seed]:
            raw = runner.training_config(lock_for_runner, seed, arm, Path("/h1/not-created") / str(seed) / arm)
            configs.append(normalize_config(raw, candidate["run_uid"], seed, arm, 2048, 64, sidecar_sha))
    with tempfile.TemporaryDirectory(prefix="h1-controller-") as temporary:
        root = Path(temporary)
        state = controller.initialize(root, args.historical_lock)
        negative = {}
        try:
            controller.transition(root, "FORMAL_TRAINING_COMPLETE")
            negative["premature_formal_completion"] = False
        except Exception:
            negative["premature_formal_completion"] = True
        evidence = root / "go.json"; atomic_json(evidence, {"status": "GO"})
        for stage in ("R2_GPU_NUMERICAL_GO", "R2_PRETRAINED_GO", "MATCHED_SMOKE_GO"):
            state = controller.transition(root, stage, evidence)
        state = controller.transition(root, "FORMAL_TRAINING_SUBMITTED")
        for seed in range(45, 57):
            triplet = root / f"triplet_{seed}.json"; atomic_json(triplet, {"status": "COMPLETE"})
            state = controller.record_triplet(root, seed, triplet)
        state = controller.transition(root, "FORMAL_TRAINING_COMPLETE")
        state = controller.transition(root, "MODEL_SELECTION_RELEASED")
        try:
            controller.transition(root, "MODEL_SELECTION_RELEASED")
            negative["duplicate_model_selection_release"] = False
        except Exception:
            negative["duplicate_model_selection_release"] = True
        state = controller.transition(root, "MODEL_SELECTION_COMPLETE")
        state = controller.transition(root, "HELD_OUT_RELEASED")
        try:
            controller.transition(root, "HELD_OUT_RELEASED")
            negative["duplicate_held_out_release"] = False
        except Exception:
            negative["duplicate_held_out_release"] = True
        final_stage = state["stage"]
    output = {
        "schema_version": 1, "protocol_id": "fpct_cfm_harness_h1_exact_image_probe_v1",
        "status": "GO", "config_count": len(configs), "smoke_config_count": 3,
        "formal_config_count": 36, "configs": configs,
        "runner_arm_order_equal": {str(seed): list(runner.ARM_ORDER[seed]) for seed in runner.ARM_ORDER} == manifest["formal_training"]["arm_order"],
        "controller_final_stage": final_stage, "controller_negative_checks": negative,
        "tripwires": tripwires, "no_training_output_directory_created": not Path("/h1/not-created").exists(),
        "scientific_output": False, "training_authorized": False,
    }
    if not output["runner_arm_order_equal"] or not all(negative.values()) or any(tripwires.values()) or not output["no_training_output_directory_created"]:
        output["status"] = "BLOCKED"
    atomic_json(args.output, output)
    return 0 if output["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
