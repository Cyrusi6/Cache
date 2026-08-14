#!/usr/bin/env python3
"""Build and verify the natural-output-free FPCT-E1 A5R5 hard gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from script.experiment.fpct_e1_a5r5_controller import (
    OLD_TREE_SHA256,
    canonical_json_bytes,
    sha256_file,
    verify_old_forensic_root,
)


PROTOCOL_ID = "fpct_e1_mechanism_audit_v12_a5r5_fresh_root_recovery"
STATUS = "GO_PRE_NATURAL_A5R5_FRESH_ROOT_RECOVERY"
TRACKED = (
    "FPCT_E1_A5R5_INTERRUPTION_RECOVERY_AMENDMENT.md",
    "recipe/eval_recipe/fpct_e1/e1_a5r5_interruption_recovery_manifest.json",
    "recipe/eval_recipe/fpct_e1/executions/a47d52f8/input_lock_interruption_observation.json",
    "script/experiment/fpct_e1_a5r5_controller.py",
    "script/analysis/fpct_e1_a5r5_gate.py",
    "test/test_fpct_e1_a5r5_controller.py",
    "test/test_fpct_e1_a5r5_gate.py",
)
IMMUTABLE = {
    "math.md": "98d1b61f84d046548d5ba0070d6858c7080cb14fdef9169b08ad167461b809ad",
    "script/experiment/fpct_e1_prepare_input_lock.py": "164bc67da8e3b0acd84c0085e72520e5951a0575ff68f22599f9e08b9d752450",
    "recipe/eval_recipe/fpct_e1/e1_a5r4_setgid_mode_synthetic_gate.json": "0f3477575b8a75fb4b6028c0eeda19aa56885aca2fdb0addce0171a5e02c7ea8",
}


def _canonical_without_evidence(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes({key: item for key, item in value.items() if key != "evidence_sha256"})


def build_gate(repo_root: Path, test_output: Path) -> dict[str, Any]:
    repo_root = repo_root.absolute()
    output = test_output.read_text(encoding="utf-8")
    matches = re.findall(r"(\d+) passed", output)
    if not matches:
        raise RuntimeError("A5R5 test output has no passing summary")
    test_count = int(matches[-1])
    if " failed" in output or " error" in output.lower():
        raise RuntimeError("A5R5 test output contains a failure/error summary")
    manifest = json.loads((repo_root / TRACKED[1]).read_text(encoding="utf-8"))
    observation = json.loads((repo_root / TRACKED[2]).read_text(encoding="utf-8"))
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "APPROVED_PRE_SUCCESSOR_NATURAL_ACCESS"
        or observation.get("status")
        != "HUMAN_OBSERVED_EXTERNAL_INTERRUPTION_NO_TERMINAL_PRODUCER_RECEIPT"
        or observation.get("full_root_fingerprint", {}).get("tree_sha256") != OLD_TREE_SHA256
    ):
        raise RuntimeError("A5R5 manifest/observation contract changed")
    immutable = {path: sha256_file(repo_root / path) for path in IMMUTABLE}
    if immutable != IMMUTABLE:
        raise RuntimeError(f"A5R5 immutable predecessor changed: {immutable}")
    forensic = verify_old_forensic_root()
    controller_source = (repo_root / TRACKED[3]).read_text(encoding="utf-8")
    required_source = (
        "start_new_session=True",
        "materialization_claim.json",
        "publish_json_no_overwrite(state_root / \"launch_claim.json\"",
        "worker_start_claim.json",
        "proc_starttime_ticks",
        "_await_bound_launch_receipt",
        "_verify_completed_input_lock",
        '"CUDA_VISIBLE_DEVICES": ""',
        "INTERRUPTED_TERMINAL_NO_RELAUNCH",
    )
    if not all(item in controller_source for item in required_source):
        raise RuntimeError("A5R5 controller lost a frozen one-shot/offline invariant")
    forbidden_source = ("AutoModel", "AutoTokenizer", "torch.load", "kubectl", "gpu_job.sh")
    if any(item in controller_source for item in forbidden_source):
        raise RuntimeError("A5R5 controller acquired a forbidden model/GPU/K8s dependency")
    tracked = {path: sha256_file(repo_root / path) for path in TRACKED}
    gate: dict[str, Any] = {
        "schema_version": 12,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r5_pre_natural_synthetic_gate",
        "status": STATUS,
        "base_commit": "a47d52f8adf5ada79919d39ac42d7b9fe655595f",
        "test_count": test_count,
        "test_output_sha256": sha256_file(test_output),
        "tracked_files_sha256": tracked,
        "immutable_predecessor_sha256": immutable,
        "interrupted_root_fingerprint": forensic,
        "checks": {
            "fresh_root_only": True,
            "old_root_no_resume_repair_reuse_cleanup": True,
            "exclusive_launch_claim_before_detached_worker": True,
            "exclusive_materialization_claim_before_run_root": True,
            "worker_process_identity_bound_to_launch_receipt": True,
            "pid_reuse_detection": True,
            "snapshot_completed_producer_verifier_replayed": True,
            "controller_state_outside_scientific_root": True,
            "cpu_offline": True,
            "unchanged_prepare_a5r4_and_math": True,
            "natural_successor_accessed": False,
            "model_or_checkpoint_loaded": False,
            "model_forward_run": False,
            "gpu_cuda_or_kubernetes_used": False,
            "training": False,
            "e1_pilot_or_confirmatory_accessed": False,
        },
    }
    gate["evidence_sha256"] = hashlib.sha256(_canonical_without_evidence(gate)).hexdigest()
    return gate


def verify_gate(gate_path: Path, repo_root: Path) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("protocol_id") != PROTOCOL_ID or gate.get("status") != STATUS:
        raise RuntimeError("A5R5 gate identity/status changed")
    evidence = gate.get("evidence_sha256")
    if evidence != hashlib.sha256(_canonical_without_evidence(gate)).hexdigest():
        raise RuntimeError("A5R5 gate evidence hash changed")
    observed = {path: sha256_file(repo_root / path) for path in TRACKED}
    if observed != gate.get("tracked_files_sha256"):
        raise RuntimeError("A5R5 tracked source changed")
    if {path: sha256_file(repo_root / path) for path in IMMUTABLE} != IMMUTABLE:
        raise RuntimeError("A5R5 immutable predecessor changed")
    if verify_old_forensic_root() != gate.get("interrupted_root_fingerprint"):
        raise RuntimeError("A5R5 interrupted-root fingerprint changed")
    return gate


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--test-output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args(argv)
    if (args.test_output is None) == (args.verify is None):
        parser.error("choose exactly one of --test-output or --verify")
    value = (
        build_gate(args.repo_root, args.test_output)
        if args.test_output is not None
        else verify_gate(args.verify, args.repo_root)
    )
    print(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
