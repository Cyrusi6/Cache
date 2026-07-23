from __future__ import annotations

import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path("/opt/fpct").resolve()
    sys.path.insert(0, str(repo / "script/runtime"))
    import fpct_bootstrap

    provenance_path = repo / ".fpct_image_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    expected = provenance["tree_sha256"]
    before = fpct_bootstrap._source_tree_sha(
        repo, exclude={provenance_path.name}
    )
    if before != expected:
        raise RuntimeError("exact image source tree is dirty before preflight")

    runtime_paths = {
        key: Path(os.environ[key]).resolve()
        for key in (
            "WANDB_DIR", "WANDB_CACHE_DIR", "WANDB_CONFIG_DIR",
            "WANDB_DATA_DIR", "HF_HOME", "HF_DATASETS_CACHE",
            "XDG_CACHE_HOME", "TORCH_EXTENSIONS_DIR", "TMPDIR",
        )
    }
    if any(path == repo or repo in path.parents for path in runtime_paths.values()):
        raise RuntimeError("a runtime write path is inside /opt/fpct")
    for path in runtime_paths.values():
        path.mkdir(parents=True, exist_ok=True)

    import rosetta  # noqa: F401
    import wandb
    run = wandb.init(
        project="FPCT",
        name="fpct-e0-runtime-write-preflight",
        mode="offline",
        dir=str(runtime_paths["WANDB_DIR"]),
        reinit=True,
    )
    run.log({"runtime_write_preflight": 1})
    run.finish()

    after = fpct_bootstrap._source_tree_sha(
        repo, exclude={provenance_path.name}
    )
    payload = {
        "schema_version": 1,
        "status": "GO" if before == after == expected else "BLOCKED",
        "expected_tree_sha256": expected,
        "before_tree_sha256": before,
        "after_tree_sha256": after,
        "runtime_paths": {key: str(value) for key, value in runtime_paths.items()},
        "python_dont_write_bytecode": os.environ.get("PYTHONDONTWRITEBYTECODE") == "1",
        "accuracy_or_model_forward": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["status"] == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
