from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2026072201)
    parser.add_argument("--run-root", type=Path, default=Path("/fpct-e0"))
    parser.add_argument("--config-root", type=Path, default=Path("/opt/fpct-e0"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sys.path.insert(0, "/opt/fpct")
    runner = load_module(
        "fpct_e0_geometry_runner", args.config_root / "fpct_e0_runner.py"
    )
    import rosetta.model.wrapper as wrapper_module

    original = wrapper_module.build_fpct_packed_layout
    calls: list[dict[str, Any]] = []

    def diagnostic_layout(source_length: int, sidecars, **kwargs):
        segments = tuple(sidecars)
        record = {
            "source_length": int(source_length),
            "segments": [
                {
                    "parent_start": int(segment.parent_start),
                    "candidate_parent_count": int(segment.key.shape[2]),
                    "top_k": int(segment.key.shape[3]),
                    "parent_end": int(
                        segment.parent_start + segment.key.shape[2]
                    ),
                    "source_length_hint": int(
                        segment.source_length_hint
                        if segment.source_length_hint is not None else -1
                    ),
                    "max_slots_hint": int(
                        segment.max_slots_hint
                        if segment.max_slots_hint is not None else -1
                    ),
                }
                for segment in segments
            ],
        }
        calls.append(record)
        print(json.dumps({"fpct_e0_geometry": record}, sort_keys=True), flush=True)
        return original(source_length, segments, **kwargs)

    wrapper_module.build_fpct_packed_layout = diagnostic_layout
    error = None
    try:
        runner.mechanism_probe(
            args.seed,
            args.run_root,
            args.run_root / "configs",
            args.config_root / "exploratory_dev_manifest.json",
        )
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        wrapper_module.build_fpct_packed_layout = original

    payload = {
        "schema_version": 1,
        "seed": args.seed,
        "calls": calls,
        "error": error,
        "accuracy_access": False,
        "correctness_access": False,
        "checkpoint_modified": False,
        "status": "DIAGNOSED" if calls and error else "INCONCLUSIVE",
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "DIAGNOSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
