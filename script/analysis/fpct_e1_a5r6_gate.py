#!/usr/bin/env python3
"""Build and verify the pre-natural FPCT-E1 A5R6 recovery gate."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from script.experiment.fpct_e1_a5r6_controller import (
    BASE_COMMIT,
    GATE_EXPECTED_COLLECTION_SUMMARY,
    GATE_EXPECTED_TEST_COUNT,
    GATE_HISTORICAL_DESELECTED_NODES,
    GATE_REQUIRED_IMMUTABLE_SHA256,
    GATE_REQUIRED_CHECKS,
    GATE_REQUIRED_TEST_NODES,
    GATE_REQUIRED_TRACKED_PATHS,
    PROTOCOL_ID,
    canonical_json_bytes,
    expected_forensic_closure,
    expected_old_controller_evidence,
    sha256_file,
    verify_old_controller_evidence,
    verify_old_forensic_root,
)


STATUS = "GO_PRE_NATURAL_A5R6_SCHEMA_BINDING_RECOVERY"
TRACKED = GATE_REQUIRED_TRACKED_PATHS
IMMUTABLE = GATE_REQUIRED_IMMUTABLE_SHA256


def _canonical_without_evidence(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(
        {key: item for key, item in value.items() if key != "evidence_sha256"}
    )


def _choice_binding_source(repo_root: Path) -> str:
    path = repo_root / "script/experiment/fpct_e1_prepare_input_lock.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_choice_audit_binding"
    )
    return ast.get_source_segment(path.read_text(encoding="utf-8"), function) or ""


def build_gate(repo_root: Path, test_output: Path) -> dict[str, Any]:
    repo_root = repo_root.absolute()
    output = test_output.read_text(encoding="utf-8")
    matches = re.findall(r"(\d+) passed", output)
    if (
        not matches
        or int(matches[-1]) != GATE_EXPECTED_TEST_COUNT
        or GATE_EXPECTED_COLLECTION_SUMMARY not in output
        or re.search(r"\b(?:failed|errors?)\b", output.lower())
    ):
        raise RuntimeError("A5R6 test output is not an all-pass receipt")
    lines = output.splitlines()
    missing_nodes = [
        node for node in GATE_REQUIRED_TEST_NODES
        if not any(line.startswith(f"{node} ") and " PASSED " in f" {line} " for line in lines)
    ]
    if missing_nodes:
        raise RuntimeError(f"A5R6 test receipt lacks required nodes: {missing_nodes}")
    manifest = json.loads(
        (repo_root / TRACKED[1]).read_text(encoding="utf-8")
    )
    observation = json.loads(
        (repo_root / TRACKED[2]).read_text(encoding="utf-8")
    )
    if (
        manifest.get("protocol_id") != PROTOCOL_ID
        or manifest.get("status") != "APPROVED_PRE_SUCCESSOR_NATURAL_ACCESS"
        or observation.get("status")
        != "TERMINAL_ENGINEERING_FAILURE_NOT_SCIENTIFIC_RESULT"
        or observation.get("root_cause", {}).get("five_field_projection_validates")
        is not True
    ):
        raise RuntimeError("A5R6 manifest/failure observation changed")
    immutable = {path: sha256_file(repo_root / path) for path in IMMUTABLE}
    if immutable != IMMUTABLE:
        raise RuntimeError(f"A5R6 immutable predecessor changed: {immutable}")
    forensic = verify_old_forensic_root()
    if forensic != expected_forensic_closure():
        raise RuntimeError("A5R6 failed-root portable closure changed")
    controller_evidence = verify_old_controller_evidence()
    if controller_evidence != expected_old_controller_evidence():
        raise RuntimeError("A5R6 failed-controller terminal evidence changed")
    binding_source = _choice_binding_source(repo_root)
    if "publication" in binding_source:
        raise RuntimeError("A5R6 v9 choice binding still embeds publication")
    prepare_source = (
        repo_root / "script/experiment/fpct_e1_prepare_input_lock.py"
    ).read_text(encoding="utf-8")
    if (
        "_verify_completed_choice_audit_publication" not in prepare_source
        or "result[\"publication\"] = publication" not in prepare_source
        or prepare_source.count("_choice_audit_binding(") != 4
    ):
        raise RuntimeError("A5R6 lost independent A5R3 publication verification")
    controller_source = (
        repo_root / "script/experiment/fpct_e1_a5r6_controller.py"
    ).read_text(encoding="utf-8")
    portable_segment = controller_source[
        controller_source.index("def filesystem_inventory_fingerprint"):
        controller_source.index("def environment_local_owner_mode_safety")
    ]
    if (
        "st_uid" in portable_segment
        or "st_gid" in portable_segment
        or "numeric_gid_in_digest\": False" not in controller_source
        or "root_local_group_mismatch_count" not in controller_source
        or "fpct_e1_a5r6_controller.py" not in controller_source
    ):
        raise RuntimeError("A5R6 portable/local forensic separation changed")
    forbidden = ("AutoModel", "AutoTokenizer", "torch.load", "kubectl", "gpu_job.sh")
    if any(item in controller_source for item in forbidden):
        raise RuntimeError("A5R6 controller acquired model/GPU/K8s behavior")
    tracked = {path: sha256_file(repo_root / path) for path in TRACKED}
    gate: dict[str, Any] = {
        "schema_version": 13,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r6_pre_natural_synthetic_gate",
        "status": STATUS,
        "base_commit": BASE_COMMIT,
        "test_count": int(matches[-1]),
        "test_output_sha256": sha256_file(test_output),
        "test_contract": {
            "required_nodes": list(GATE_REQUIRED_TEST_NODES),
            "required_nodes_sha256": hashlib.sha256(
                ("\n".join(GATE_REQUIRED_TEST_NODES) + "\n").encode()
            ).hexdigest(),
            "historical_deselected_nodes": list(GATE_HISTORICAL_DESELECTED_NODES),
            "historical_deselected_nodes_sha256": hashlib.sha256(
                ("\n".join(GATE_HISTORICAL_DESELECTED_NODES) + "\n").encode()
            ).hexdigest(),
            "collection_summary": GATE_EXPECTED_COLLECTION_SUMMARY,
        },
        "tracked_files_sha256": tracked,
        "immutable_predecessor_sha256": immutable,
        "failed_root_portable_closure": forensic,
        "failed_controller_terminal_evidence": controller_evidence,
        "checks": dict(GATE_REQUIRED_CHECKS),
    }
    gate["evidence_sha256"] = hashlib.sha256(
        _canonical_without_evidence(gate)
    ).hexdigest()
    return gate


def verify_gate(gate_path: Path, repo_root: Path) -> dict[str, Any]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    expected_contract = {
        "required_nodes": list(GATE_REQUIRED_TEST_NODES),
        "required_nodes_sha256": hashlib.sha256(
            ("\n".join(GATE_REQUIRED_TEST_NODES) + "\n").encode()
        ).hexdigest(),
        "historical_deselected_nodes": list(GATE_HISTORICAL_DESELECTED_NODES),
        "historical_deselected_nodes_sha256": hashlib.sha256(
            ("\n".join(GATE_HISTORICAL_DESELECTED_NODES) + "\n").encode()
        ).hexdigest(),
        "collection_summary": GATE_EXPECTED_COLLECTION_SUMMARY,
    }
    if (
        gate.get("schema_version") != 13
        or gate.get("protocol_id") != PROTOCOL_ID
        or gate.get("artifact_type") != "a5r6_pre_natural_synthetic_gate"
        or gate.get("status") != STATUS
        or gate.get("base_commit") != BASE_COMMIT
        or gate.get("test_count") != GATE_EXPECTED_TEST_COUNT
        or gate.get("test_contract") != expected_contract
        or gate.get("checks") != GATE_REQUIRED_CHECKS
    ):
        raise RuntimeError("A5R6 gate identity/status changed")
    if gate.get("evidence_sha256") != hashlib.sha256(
        _canonical_without_evidence(gate)
    ).hexdigest():
        raise RuntimeError("A5R6 gate evidence hash changed")
    if {path: sha256_file(repo_root / path) for path in TRACKED} != gate.get(
        "tracked_files_sha256"
    ):
        raise RuntimeError("A5R6 tracked source changed")
    if {path: sha256_file(repo_root / path) for path in IMMUTABLE} != IMMUTABLE:
        raise RuntimeError("A5R6 immutable predecessor changed")
    if verify_old_forensic_root() != gate.get("failed_root_portable_closure"):
        raise RuntimeError("A5R6 failed-root portable closure changed")
    if verify_old_controller_evidence() != gate.get(
        "failed_controller_terminal_evidence"
    ):
        raise RuntimeError("A5R6 failed-controller terminal evidence changed")
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
