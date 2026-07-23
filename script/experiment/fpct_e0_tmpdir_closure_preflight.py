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
    target = (repo / "script/train/SFT_train.py").resolve()
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "script/runtime"))

    import fpct_bootstrap

    provenance_path = repo / ".fpct_image_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    expected_tree = provenance["tree_sha256"]
    tree_before = fpct_bootstrap._source_tree_sha(
        repo, exclude={provenance_path.name}
    )
    if tree_before != expected_tree:
        raise RuntimeError("exact image source tree is dirty before preflight")

    if Path(os.environ.get("TMPDIR", "")).resolve() != Path("/tmp"):
        raise RuntimeError("TMPDIR must equal /tmp for bootstrap normalization")

    fpct_bootstrap._ACTIVE_REPO = repo
    fpct_bootstrap._ACTIVE_TARGET = target
    fpct_bootstrap._ACTIVE_SENTINEL = fpct_bootstrap._SENTINEL
    sys.modules["fpct_bootstrap"] = fpct_bootstrap
    fpct_bootstrap._LOADED_BY_KEY = fpct_bootstrap._closure(repo, True)
    fpct_bootstrap._PRE_ATTESTATION = fpct_bootstrap._attest(
        repo, target, True
    )
    fpct_bootstrap._LOADED_BY_KEY["formal_target"] = (
        fpct_bootstrap._load_script_module("formal_target", target)
    )
    before = fpct_bootstrap._attest(repo, target, True)
    fpct_bootstrap._PRE_ATTESTATION = before

    import wandb

    run = wandb.init(
        project="FPCT",
        name="fpct-e0-tmpdir-closure-preflight",
        mode="offline",
        dir=os.environ["WANDB_DIR"],
        reinit=True,
    )
    run.log({"tmpdir_closure_preflight": 1})
    run.finish()

    after = fpct_bootstrap._attest(repo, target, True)
    tree_after = fpct_bootstrap._source_tree_sha(
        repo, exclude={provenance_path.name}
    )
    stable_before = before["process"]["stable_sys_path"]
    stable_after = after["process"]["stable_sys_path"]
    canonical_markers = [
        value for value in stable_before
        if value.startswith("/tmp/<torch-remote-module-sha256=")
    ]
    raw_torch_tmp = [
        value for value in stable_before
        if value.startswith("/tmp/tmp")
    ]
    status = "GO" if all((
        before["stable_fingerprint_sha256"]
        == after["stable_fingerprint_sha256"],
        stable_before == stable_after,
        bool(canonical_markers),
        not raw_torch_tmp,
        tree_before == tree_after == expected_tree,
    )) else "BLOCKED"
    payload = {
        "schema_version": 1,
        "status": status,
        "expected_tree_sha256": expected_tree,
        "tree_before_sha256": tree_before,
        "tree_after_sha256": tree_after,
        "fingerprint_before_sha256": before["stable_fingerprint_sha256"],
        "fingerprint_after_sha256": after["stable_fingerprint_sha256"],
        "stable_sys_path_before": stable_before,
        "stable_sys_path_after": stable_after,
        "canonical_torch_remote_module_markers": canonical_markers,
        "raw_torch_tmp_entries": raw_torch_tmp,
        "tmpdir": os.environ["TMPDIR"],
        "wandb_mode": "offline",
        "model_forward": False,
        "training": False,
        "checkpoint_access": False,
        "accuracy_access": False,
    }
    atomic_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "GO" else 2


if __name__ == "__main__":
    raise SystemExit(main())
