#!/usr/bin/env python3
"""Verify the prospective R2k boundary before any new GPU diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="HEAD")
    parser.add_argument("--phase", choices=("protocol", "implementation"), default="protocol")
    args = parser.parse_args()
    recipe = REPO / "recipe/eval_recipe/fpct_gpu_r2k"
    diff = json.loads((recipe / "protocol_diff.json").read_text())
    boundary = json.loads((recipe / "implementation_boundary.json").read_text())
    manifest = json.loads((recipe / "latency_audit_manifest.json").read_text())
    if manifest["immutable_predecessor"]["terminal"] != "GPU_ENGINEERING_BLOCKED_R2":
        raise SystemExit("R2j terminal changed")
    if manifest["accuracy_or_correctness_accessed"]:
        raise SystemExit("accuracy firewall violation")
    for relative, expected in diff["baseline_sha256"].items():
        path = REPO / relative
        actual = sha256_file(path)
        if args.phase == "implementation" and relative in {
            "rosetta/model/fpct_attention.py",
            "rosetta/model/wrapper.py",
            "script/experiment/fpct_gpu_r2_runner.py",
        }:
            continue
        if actual != expected:
            raise SystemExit(f"immutable hash mismatch: {relative}: {actual} != {expected}")
    for relative, expected_blob in diff["r2i_r2j_hot_path_blob_identity"].items():
        for commit in (diff["r2i_scientific_commit"], diff["baseline_scientific_commit"]):
            actual_blob = git("rev-parse", f"{commit}:{relative}")
            if actual_blob != expected_blob:
                raise SystemExit(f"historical hot-path blob mismatch: {commit}:{relative}")
    changed = set(
        filter(
            None,
            git("diff", "--name-only", diff["protocol_parent_commit"], args.target).splitlines(),
        )
    )
    allowed_exact = set(boundary["allowlist"]["scientific_hot_path"])
    allowed_prefixes = (
        "recipe/eval_recipe/fpct_gpu_r2k/",
        "recipe/k8s/fpct_gpu_r2k/",
        "test/test_fpct_gpu_r2k_",
        "script/analysis/fpct_gpu_r2k_",
        "script/experiment/fpct_gpu_r2k_",
        "FPCT_GPU_R2K_",
    )
    allowed_exact.update({"FPCT_STATUS.md", "FRAMEWORK_UPDATE.md", "EXPERIMENT.md"})
    unexpected = sorted(
        path
        for path in changed
        if path not in allowed_exact and not path.startswith(allowed_prefixes)
    )
    if unexpected:
        raise SystemExit(f"paths outside R2k allowlist: {unexpected}")
    forbidden_prefixes = tuple(boundary["forbidden"]["paths"])
    forbidden_changed = sorted(
        path for path in changed if path.startswith(forbidden_prefixes)
    )
    if forbidden_changed:
        raise SystemExit(f"forbidden paths changed: {forbidden_changed}")
    print(json.dumps({"status": "GO", "phase": args.phase, "target": args.target, "changed_paths": sorted(changed)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
