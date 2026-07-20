from __future__ import annotations

"""Recoverable state machine for the prospective FPCT GPU R2 execution."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any


STAGES = (
    "R2_RUN_LOCKED",
    "R2_GPU_NUMERICAL_GO",
    "R2_PRETRAINED_GO",
    "MATCHED_SMOKE_GO",
    "FORMAL_TRAINING_SUBMITTED",
    "FORMAL_TRAINING_COMPLETE",
    "MODEL_SELECTION_RELEASED",
    "MODEL_SELECTION_COMPLETE",
    "HELD_OUT_RELEASED",
    "TERMINAL",
)
TERMINAL = {
    "GPU_ENGINEERING_BLOCKED_R2",
    "MATCHED_SMOKE_INTEGRITY_FAILURE",
    "FORMAL_TRAINING_INCONCLUSIVE",
    "MODEL_SELECTION_FUTILITY",
    "PERFORMANCE_GO_MECHANISM_UNRESOLVED",
    "MECHANISM_SUPPORTED_GO",
    "FUTILITY_NO_GO",
    "HARM_NO_GO",
    "INCONCLUSIVE",
}


class R2StateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise R2StateError(f"expected JSON object: {path}")
    return value


def append_event(root: Path, payload: dict[str, Any]) -> None:
    ledger = root / "event_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def initialize(root: Path, run_lock: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "controller_state.json"
    lock_sha = sha256_file(run_lock)
    if state_path.is_file():
        state = load_json(state_path)
        if state["run_lock_sha256"] != lock_sha:
            raise R2StateError("controller root is bound to another run lock")
        return state
    lock = load_json(run_lock)
    if lock.get("status") != "PRE_OUTPUT_LOCKED_R2":
        raise R2StateError("R2 controller requires PRE_OUTPUT_LOCKED_R2")
    state = {
        "schema_version": 1,
        "run_uid": lock["run_uid"],
        "run_lock": str(run_lock.resolve()),
        "run_lock_sha256": lock_sha,
        "stage": "R2_RUN_LOCKED",
        "completed_triplets": [],
        "failed_triplets": {},
        "model_selection_release_count": 0,
        "held_out_release_count": 0,
        "updated_unix": time.time(),
    }
    atomic_json(state_path, state)
    append_event(root, {"event": "INITIALIZED", "unix": state["updated_unix"]})
    return state


def _validate_evidence(target: str, evidence: Path | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    payload = load_json(evidence)
    expected = {
        "R2_GPU_NUMERICAL_GO": "GO",
        "R2_PRETRAINED_GO": "GO",
        "MATCHED_SMOKE_GO": "GO",
    }.get(target)
    if expected is not None and payload.get("status") != expected:
        raise R2StateError(
            f"{target} requires evidence status {expected}, got {payload.get('status')}"
        )
    return payload


def transition(root: Path, target: str, evidence: Path | None = None) -> dict[str, Any]:
    state_path = root / "controller_state.json"
    state = load_json(state_path)
    _validate_evidence(target, evidence)
    if target in TERMINAL:
        next_stage = "TERMINAL"
    else:
        if target not in STAGES:
            raise R2StateError(f"unknown R2 stage: {target}")
        current_index = STAGES.index(state["stage"])
        target_index = STAGES.index(target)
        if target_index != current_index + 1:
            raise R2StateError(f"non-sequential transition {state['stage']} -> {target}")
        next_stage = target
    if target == "FORMAL_TRAINING_COMPLETE" and state["completed_triplets"] != list(range(45, 57)):
        raise R2StateError("formal completion requires all 12 matched triplets")
    if target == "MODEL_SELECTION_RELEASED":
        if state["model_selection_release_count"]:
            raise R2StateError("model-selection was already released")
        state["model_selection_release_count"] = 1
    if target == "HELD_OUT_RELEASED":
        if state["stage"] != "MODEL_SELECTION_COMPLETE":
            raise R2StateError("held-out release requires completed model-selection")
        if state["held_out_release_count"]:
            raise R2StateError("held-out was already released")
        state["held_out_release_count"] = 1
    state["stage"] = next_stage
    if target in TERMINAL:
        state["terminal_classification"] = target
    if evidence is not None:
        state.setdefault("evidence", {})[target] = {
            "path": str(evidence.resolve()),
            "sha256": sha256_file(evidence),
        }
    state["updated_unix"] = time.time()
    atomic_json(state_path, state)
    append_event(root, {"event": "TRANSITION", "target": target, "unix": state["updated_unix"]})
    return state


def record_triplet(root: Path, seed: int, manifest: Path) -> dict[str, Any]:
    if seed not in range(45, 57):
        raise R2StateError("formal seed must be 45..56")
    state_path = root / "controller_state.json"
    state = load_json(state_path)
    if state["stage"] != "FORMAL_TRAINING_SUBMITTED":
        raise R2StateError("triplets can only be recorded during formal training")
    payload = load_json(manifest)
    status = str(payload.get("status"))
    record = {
        "path": str(manifest.resolve()),
        "sha256": sha256_file(manifest),
        "status": status,
    }
    key = str(seed)
    prior = state.setdefault("triplets", {}).get(key)
    if prior is not None and prior != record:
        raise R2StateError("refusing to replace triplet evidence")
    state["triplets"][key] = record
    if status == "COMPLETE":
        if seed not in state["completed_triplets"]:
            state["completed_triplets"].append(seed)
            state["completed_triplets"].sort()
    else:
        state["failed_triplets"][key] = status
    state["updated_unix"] = time.time()
    atomic_json(state_path, state)
    append_event(root, {"event": "TRIPLET", "seed": seed, "status": status, "unix": state["updated_unix"]})
    return state


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--run-lock", type=Path, required=True)
    advance = sub.add_parser("transition")
    advance.add_argument("stage")
    advance.add_argument("--evidence", type=Path)
    triplet = sub.add_parser("record-triplet")
    triplet.add_argument("--seed", type=int, required=True)
    triplet.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "init":
        payload = initialize(args.root, args.run_lock)
    elif args.command == "transition":
        payload = transition(args.root, args.stage, args.evidence)
    elif args.command == "record-triplet":
        payload = record_triplet(args.root, args.seed, args.manifest)
    else:
        payload = load_json(args.root / "controller_state.json")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
