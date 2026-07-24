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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, default=Path("/fpct-e0"))
    parser.add_argument("--config-root", type=Path, default=Path("/opt/fpct-e0"))
    args = parser.parse_args()

    sys.path.insert(0, "/opt/fpct")
    runner = load_module(
        "fpct_e0_decode_smoke_runner", args.config_root / "fpct_e0_runner.py"
    )
    import rosetta.model.wrapper as wrapper_module

    original_layout = wrapper_module.build_fpct_packed_layout
    original_atomic = runner.atomic_json
    calls: list[dict[str, Any]] = []
    mechanism_payload: dict[str, Any] = {}

    def capture_layout(source_length: int, sidecars, **kwargs):
        segments = tuple(sidecars)
        calls.append({
            "source_length": int(source_length),
            "parent_ranges": [
                [
                    int(segment.parent_start),
                    int(segment.parent_start + segment.key.shape[2]),
                ]
                for segment in segments
            ],
            "source_length_hints": [
                int(segment.source_length_hint or 0) for segment in segments
            ],
        })
        return original_layout(source_length, segments, **kwargs)

    def capture_atomic(_path: Path, payload: Any, **_kwargs) -> None:
        mechanism_payload.update(payload)

    wrapper_module.build_fpct_packed_layout = capture_layout
    runner.atomic_json = capture_atomic
    error = None
    try:
        returned = runner.mechanism_probe(
            2026072201,
            args.run_root,
            args.run_root / "configs",
            args.config_root / "exploratory_dev_manifest.json",
        )
        mechanism_payload.update(returned)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        wrapper_module.build_fpct_packed_layout = original_layout
        runner.atomic_json = original_atomic

    geometry_ok = (
        len(calls) >= 2
        and calls[0]["source_length"] == 129
        and calls[1]["source_length"] == 130
        and calls[0]["parent_ranges"] == calls[1]["parent_ranges"]
        and max(end for _start, end in calls[1]["parent_ranges"])
        <= calls[1]["source_length"]
    )
    status = "GO" if error is None and geometry_ok else "BLOCKED"
    payload = {
        "schema_version": 1,
        "status": status,
        "calls": calls,
        "error": error,
        "geometry_ok": geometry_ok,
        "mechanism_nonzero": mechanism_payload.get("nonzero_activation"),
        "accuracy_access": False,
        "correctness_access": False,
        "checkpoint_modified": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
