from __future__ import annotations

"""Recoverable, result-firewalled controller for the FPCT confirmatory run."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any


STAGES = (
    "RUN_LOCKED", "GPU_NUMERICAL_GO", "PRETRAINED_SMOKE_GO",
    "MATCHED_SMOKE_GO", "FORMAL_TRAINING_SUBMITTED", "FORMAL_TRAINING_COMPLETE",
    "MODEL_SELECTION_RELEASED", "MODEL_SELECTION_COMPLETE",
    "HELD_OUT_RELEASED", "TERMINAL",
)
TERMINAL = {
    "IMPORT_PROVENANCE_BLOCKED", "FORENSIC_REPLAY_MISMATCH",
    "NO_GO_GPU_CURRENT_CERTIFIER", "GPU_ENGINEERING_BLOCKED",
    "MODEL_SELECTION_FUTILITY", "PERFORMANCE_GO_MECHANISM_UNRESOLVED",
    "MECHANISM_SUPPORTED_GO", "INCONCLUSIVE",
}


class StateError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise StateError(f"controller state missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def append_event(root: Path, event: dict[str, Any]) -> None:
    ledger = root / "event_ledger.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def initialize(root: Path, run_lock: Path) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "controller_state.json"
    if state_path.exists():
        state = load(state_path)
        if state["run_lock_sha256"] != sha256(run_lock):
            raise StateError("existing controller belongs to a different run lock")
        return state
    lock = json.loads(run_lock.read_text(encoding="utf-8"))
    state = {
        "schema_version": 1,
        "run_uid": lock["run_uid"],
        "run_lock": str(run_lock.resolve()),
        "run_lock_sha256": sha256(run_lock),
        "stage": "RUN_LOCKED",
        "held_out_release_count": 0,
        "model_selection_release_count": 0,
        "completed_triplets": [],
        "failed_triplets": {},
        "updated_unix": time.time(),
    }
    atomic_json(state_path, state)
    append_event(root, {"event": "INITIALIZED", "stage": "RUN_LOCKED", "unix": state["updated_unix"]})
    return state


def transition(root: Path, target: str, evidence: Path | None = None) -> dict[str, Any]:
    state_path = root / "controller_state.json"
    state = load(state_path)
    if target in TERMINAL:
        next_stage = "TERMINAL"
    else:
        next_stage = target
        if target not in STAGES:
            raise StateError(f"unknown stage: {target}")
        current_index = STAGES.index(state["stage"])
        target_index = STAGES.index(target)
        if target_index != current_index + 1:
            raise StateError(f"non-sequential transition {state['stage']} -> {target}")
    if target == "FORMAL_TRAINING_COMPLETE" and len(state["completed_triplets"]) != 12:
        raise StateError("formal completion requires 12 complete matched triplets")
    if target == "MODEL_SELECTION_RELEASED":
        if state["model_selection_release_count"]:
            raise StateError("model-selection has already been released")
        state["model_selection_release_count"] = 1
    if target == "HELD_OUT_RELEASED":
        if state["stage"] != "MODEL_SELECTION_COMPLETE":
            raise StateError("held-out release requires completed model-selection gate")
        if state["held_out_release_count"]:
            raise StateError("held-out has already been released")
        state["held_out_release_count"] = 1
    state["stage"] = next_stage
    if target in TERMINAL:
        state["terminal_classification"] = target
    if evidence:
        state.setdefault("evidence", {})[target] = {
            "path": str(evidence.resolve()), "sha256": sha256(evidence)
        }
    state["updated_unix"] = time.time()
    atomic_json(state_path, state)
    append_event(root, {"event": "TRANSITION", "target": target, "unix": state["updated_unix"]})
    return state


def record_triplet(root: Path, seed: int, manifest: Path, status: str) -> dict[str, Any]:
    if seed not in range(45, 57):
        raise StateError("formal seed outside 45..56")
    state_path = root / "controller_state.json"
    state = load(state_path)
    if state["stage"] != "FORMAL_TRAINING_SUBMITTED":
        raise StateError("triplets can only be recorded during formal training")
    key = str(seed)
    record = {"path": str(manifest.resolve()), "sha256": sha256(manifest), "status": status}
    if key in state.get("triplets", {}) and state["triplets"][key] != record:
        raise StateError("refusing to overwrite a triplet with different evidence")
    state.setdefault("triplets", {})[key] = record
    if status == "COMPLETE" and seed not in state["completed_triplets"]:
        state["completed_triplets"].append(seed)
        state["completed_triplets"].sort()
    elif status != "COMPLETE":
        state["failed_triplets"][key] = status
    state["updated_unix"] = time.time()
    atomic_json(state_path, state)
    append_event(root, {"event": "TRIPLET", "seed": seed, "status": status, "unix": state["updated_unix"]})
    return state


def kubectl_submit(template: Path, namespace: str) -> None:
    subprocess.run(["kubectl", "apply", "-n", namespace, "-f", str(template)], check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init"); init.add_argument("--run-lock", type=Path, required=True)
    trans = sub.add_parser("transition"); trans.add_argument("stage"); trans.add_argument("--evidence", type=Path)
    trip = sub.add_parser("record-triplet"); trip.add_argument("--seed", type=int, required=True); trip.add_argument("--manifest", type=Path, required=True); trip.add_argument("--status", required=True)
    submit = sub.add_parser("submit"); submit.add_argument("--template", type=Path, required=True); submit.add_argument("--namespace", default="c2c-research")
    sub.add_parser("status")
    args = parser.parse_args()
    if args.command == "init": payload = initialize(args.root, args.run_lock)
    elif args.command == "transition": payload = transition(args.root, args.stage, args.evidence)
    elif args.command == "record-triplet": payload = record_triplet(args.root, args.seed, args.manifest, args.status)
    elif args.command == "submit": kubectl_submit(args.template, args.namespace); payload = {"submitted": str(args.template)}
    else: payload = load(args.root / "controller_state.json")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
