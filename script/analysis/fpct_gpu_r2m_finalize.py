#!/usr/bin/env python3
"""Config/provenance-only finalizer for the frozen R2m immutable gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


class R2mFinalizeError(RuntimeError):
    pass


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def finalize(
    original_path: Path,
    semantic_path: Path,
    active_path: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise R2mFinalizeError("refusing to overwrite immutable R2m final result")
    original = load(original_path)
    semantic = load(semantic_path)
    active = load(active_path)
    checks = {
        "original_23_checks": original.get("status") == "GO"
        and len(original.get("checks", {})) == 23
        and all(bool(value) for value in original.get("checks", {}).values()),
        "semantic_checks": semantic.get("status") == "GO"
        and all(bool(value) for value in semantic.get("checks", {}).values()),
        "balanced_checkpoint_native": bool(
            active.get("canaries", {}).get("checkpoint_native", {}).get("qualified")
        ),
        "balanced_forced_on": bool(
            active.get("canaries", {}).get("forced_on", {}).get("qualified")
        ),
        "accuracy_firewall": not bool(original.get("accuracy_or_correctness_accessed"))
        and not bool(semantic.get("accuracy_or_correctness_accessed"))
        and not bool(active.get("accuracy_or_correctness_accessed")),
    }
    go = all(checks.values())
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2m_immutable_final_v1",
        "classification": "R2M_IMMUTABLE_GO" if go else "GPU_ENGINEERING_BLOCKED_R2M",
        "status": "GO" if go else "GPU_ENGINEERING_BLOCKED_R2M",
        "checks": checks,
        "original_result": original,
        "semantic_result": semantic,
        "balanced_active_canary": active,
        "training_authorized": go,
        "accuracy_or_correctness_accessed": False,
    }
    atomic_json(output, payload)
    if not go:
        raise R2mFinalizeError(f"immutable R2m final gate failed: {checks}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original-result", type=Path, required=True)
    parser.add_argument("--semantic-result", type=Path, required=True)
    parser.add_argument("--active-aggregate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = finalize(
        args.original_result, args.semantic_result, args.active_aggregate, args.output
    )
    print(json.dumps({"status": payload["status"], "classification": payload["classification"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
