from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


NEW_IMAGE = "docker.io/library/fpct-e0-decode@sha256:19c7a81568b4701fa60f11c682423d9eb812fe6e8f95bcfc36aa21eb98e82683"
NEW_COMMIT = "6a51ad4ed1d66067c0ac2d3f2c8c3b5de0f5d2ba"
NEW_TREE = "1534f7fe2010ebc51b160b3bcb74e58d9eb44b3fd6cb8aad7674356dc91f4b7c"
ORIGINAL_RUNNER = Path("/opt/fpct-e0/fpct_e0_runner.py")
PRESEALED_LAUNCHER = "/opt/fpct/script/experiment/fpct_e0_bootstrap_launcher.py"
BOOTSTRAP = "/opt/fpct/script/runtime/fpct_bootstrap.py"


def load_original() -> Any:
    spec = importlib.util.spec_from_file_location(
        "fpct_e0_decode_original_runner", ORIGINAL_RUNNER
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ORIGINAL_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def verify_image() -> dict[str, Any]:
    provenance = json.loads(
        Path("/opt/fpct/.fpct_image_provenance.json").read_text()
    )
    expected = {
        "head": NEW_COMMIT,
        "upstream": NEW_COMMIT,
        "branch": "research/fpct-e0-exploratory",
        "tree_sha256": NEW_TREE,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"decode recovery image mismatch: {provenance}")
    return provenance


def install_presealed_training(original: Any) -> None:
    original_command = original._training_command

    def command(seed: int, arm: str, config: Path, attestation: Path):
        values = original_command(seed, arm, config, attestation)
        positions = [
            index for index, value in enumerate(values) if value == BOOTSTRAP
        ]
        if positions != [8]:
            raise RuntimeError(f"unexpected bootstrap position: {positions}")
        values.insert(positions[0], PRESEALED_LAUNCHER)
        return values

    original._training_command = command


def prepare_common(original: Any, config_root: Path, run_root: Path):
    manifest = json.loads((config_root / "e0_manifest.json").read_text())
    index = json.loads((config_root / "config_index.json").read_text())
    if manifest["run_uid"] != original.E0_UID:
        raise RuntimeError("E0 run UID mismatch")
    assets = original._verify_runtime_inputs(
        manifest, config_root, run_root
    )
    original._copy_configs(config_root, run_root, index)
    parity = json.loads(
        (run_root / "parity/formula_production_parity.json").read_text()
    )
    if parity.get("status") != "GO":
        raise RuntimeError("E0 production parity is not GO")
    return manifest, index, assets


def run_evaluation(original: Any, seed: int, run_root: Path, config_root: Path):
    for cell in ("Y_CC", "Y_CF", "Y_FC", "Y_FF"):
        for task in original.TASK_LIMITS:
            config = run_root / "configs" / f"eval_{seed}_{cell}_{task}.yaml"
            original._run([
                "/opt/conda/bin/python3.11",
                "/opt/fpct/script/evaluation/unified_evaluator.py",
                "--config",
                str(config),
            ], cwd=Path("/opt/fpct"))
    return original.mechanism_probe(
        seed,
        run_root,
        run_root / "configs",
        config_root / "exploratory_dev_manifest.json",
    )


def set_active(seed_root: Path, attempt: Path) -> Path:
    active = seed_root / "active"
    if active.is_symlink():
        active.unlink()
    elif active.exists():
        raise RuntimeError("active seed path is not a symlink")
    active.symlink_to(attempt.name)
    return active


def resume_evaluation(
    original: Any,
    seed: int,
    source_attempt_number: int,
    attempt_number: int,
    config_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    _manifest, _index, assets = prepare_common(
        original, config_root, run_root
    )
    image_provenance = verify_image()
    seed_root = run_root / "seeds" / str(seed)
    source_attempt = seed_root / f"attempt_{source_attempt_number}"
    integrity = json.loads((source_attempt / "matched_integrity.json").read_text())
    if integrity.get("status") != "GO":
        raise RuntimeError("source matched training integrity is not GO")
    for arm in original.SEED_ORDER[seed]:
        record = json.loads(
            (source_attempt / arm / "fpct_formal_integrity.json").read_text()
        )
        if record.get("optimizer_steps") != 64 or not record.get(
            "checkpoint_reload_equal"
        ):
            raise RuntimeError(f"source training arm is incomplete: {arm}")

    attempt = seed_root / f"attempt_{attempt_number}"
    if attempt.exists():
        raise FileExistsError(attempt)
    attempt.mkdir()
    for name in ("c_post", "f", "matched_integrity.json"):
        (attempt / name).symlink_to(
            Path("..") / source_attempt.name / name
        )
    set_active(seed_root, attempt)
    original.atomic_json(attempt / "runtime_provenance.json", {
        "schema_version": 1,
        "run_uid": original.E0_UID,
        "seed": seed,
        "attempt": attempt_number,
        "mode": "EVALUATION_RECOVERY",
        "training_source_attempt": source_attempt_number,
        "image": NEW_IMAGE,
        "image_provenance": image_provenance,
        "mounted_asset_sha256": assets,
        "invalid_evaluation_attempt_quarantined": source_attempt_number,
    })
    mechanism = run_evaluation(original, seed, run_root, config_root)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "attempt": attempt_number,
        "status": "COMPLETE",
        "integrity": "GO",
        "training_source_attempt": source_attempt_number,
        "mechanism_nonzero": mechanism["nonzero_activation"],
        "evaluation_recovery": True,
    }
    original.atomic_json(attempt / "seed_complete.json", payload)
    return payload


def run_fresh(
    original: Any,
    seed: int,
    attempt_number: int,
    config_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    _manifest, _index, assets = prepare_common(
        original, config_root, run_root
    )
    image_provenance = verify_image()
    seed_root = run_root / "seeds" / str(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    attempt = seed_root / f"attempt_{attempt_number}"
    if attempt.exists():
        raise FileExistsError(attempt)
    attempt.mkdir()
    set_active(seed_root, attempt)
    original.atomic_json(attempt / "runtime_provenance.json", {
        "schema_version": 1,
        "run_uid": original.E0_UID,
        "seed": seed,
        "attempt": attempt_number,
        "mode": "FRESH_TRAIN_AND_EVALUATE",
        "pod": os.environ.get("E0_POD_NAME"),
        "node": os.environ.get("E0_NODE_NAME"),
        "image": NEW_IMAGE,
        "image_provenance": image_provenance,
        "mounted_asset_sha256": assets,
    })
    order = original.SEED_ORDER[seed]
    for arm in order:
        config = run_root / "configs" / f"train_{seed}_{arm}.json"
        attestation = attempt / "attestations" / f"{seed}-{arm}-rank_{{rank}}.json"
        attestation.parent.mkdir(parents=True, exist_ok=True)
        original._run(
            original._training_command(seed, arm, config, attestation),
            cwd=Path("/opt/fpct"),
        )
    integrity = original._verify_matched(seed, attempt, order)
    mechanism = run_evaluation(original, seed, run_root, config_root)
    payload = {
        "schema_version": 1,
        "seed": seed,
        "attempt": attempt_number,
        "status": "COMPLETE",
        "integrity": integrity["status"],
        "mechanism_nonzero": mechanism["nonzero_activation"],
    }
    original.atomic_json(attempt / "seed_complete.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    resume = sub.add_parser("resume-evaluation")
    resume.add_argument("--seed", type=int, required=True)
    resume.add_argument("--source-attempt", type=int, required=True)
    resume.add_argument("--attempt", type=int, required=True)
    fresh = sub.add_parser("run-seed")
    fresh.add_argument("--seed", type=int, required=True)
    fresh.add_argument("--attempt", type=int, required=True)
    for child in (resume, fresh):
        child.add_argument("--config-root", type=Path, default=Path("/opt/fpct-e0"))
        child.add_argument("--run-root", type=Path, default=Path("/fpct-e0"))
    args = parser.parse_args()
    original = load_original()
    install_presealed_training(original)
    if args.command == "resume-evaluation":
        payload = resume_evaluation(
            original, args.seed, args.source_attempt, args.attempt,
            args.config_root.resolve(), args.run_root.resolve(),
        )
    else:
        payload = run_fresh(
            original, args.seed, args.attempt,
            args.config_root.resolve(), args.run_root.resolve(),
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
