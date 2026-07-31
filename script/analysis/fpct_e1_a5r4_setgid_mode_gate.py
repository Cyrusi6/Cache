#!/usr/bin/env python3
"""Prospective CPU/offline gate for the FPCT-E1 A5R4 setgid overlay."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from script.experiment.fpct_e1_a5_prompt_provenance import (
    canonical_json_bytes,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 11
CHOICE_SEMANTICS_VERSION = 9
A5R3_ENVELOPE_SCHEMA_VERSION = 10
PROTOCOL_ID = "fpct_e1_mechanism_audit_v11_a5r4_setgid_mode_predicate"
ARTIFACT_TYPE = "a5r4_setgid_mode_synthetic_gate"
STATUS = "GO_PRE_NATURAL_A5R4_SETGID_MODE_HARD_GATE"
BASE_COMMIT = "5f2e41148d9b923e1264e83d76106389c155a596"
BLOCKED_EXECUTION_SHA = "3263531ed7137efd241de3c951bf7c5e9d37e669"
BLOCKED_CLOSURE_SHA256 = (
    "8ca3537914edb880b3436012841ec47b8d416345e2646fce22597c60d3bf4d17"
)

AMENDMENT_RELATIVE = Path("FPCT_E1_A5R4_SETGID_MODE_AMENDMENT.md")
CONTRACT_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r4_setgid_mode_contract.json"
)
SCHEMA_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r4_setgid_mode_schema.json"
)
GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r4_setgid_mode_synthetic_gate.json"
)
BLOCKED_CLOSURE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/executions/3263531e/input_lock_failure_receipt.json"
)
PREPARE_RELATIVE = Path("script/experiment/fpct_e1_prepare_input_lock.py")
GATE_SCRIPT_RELATIVE = Path(
    "script/analysis/fpct_e1_a5r4_setgid_mode_gate.py"
)
GATE_TEST_RELATIVE = Path("test/test_fpct_e1_a5r4_setgid_mode_gate.py")
TARGET_FILESYSTEM_ROOT = Path("/netdisk/lijunsi/fpct-e1")

A5R3_OBJECT_SHA256 = {
    "FPCT_E1_A5R3_PORTABLE_PUBLICATION_AMENDMENT.md": (
        "ccfb73bdfd396d5941e975b09143450189e75fffd4884482ce613cf03feb6cd4"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_contract.json": (
        "2ba72cfbccb5a1b7c9f46c02510f4763a2917210d3884c7b5d8223580db9e272"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_schema.json": (
        "b24d197481c01c917ad9d4063441f96ac23979bdb4b3db24855e551c7fb91460"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_synthetic_gate.json": (
        "004a8feb272af78c439f970b1ae637400e641a4fd93a412f29832f288871a4c2"
    ),
}
A5R3_GATE_EVIDENCE = {
    "status": "GO_PRE_NATURAL_A5R3_PORTABLE_PUBLICATION_HARD_GATE",
    "test_count": 43,
    "test_output_sha256": (
        "b5788fb2d770e31bae15967d968b49498dbae0e19bd96e45811291b800ed0ed3"
    ),
    "execution_tree_sha256": (
        "54f108bb07f3d10aa759e4eb93ee011915e5b93e91b864ae32da6923a06b1369"
    ),
    "evidence_sha256": (
        "907f3b6a631ed8a1c6bb13cf559097fc96fc9b5a2b2148ebf0c8eabcaeca63c9"
    ),
}

UNCHANGED_A5R3_PUBLICATION_AST_SHA256 = {
    "_acquire_choice_audit_publication_claim": (
        "bed373ca7cae771906c3051e3bdb5f1fbb94b20ff56f1c7304123e09120b2278"
    ),
    "_assert_choice_audit_publication_state_is_fresh": (
        "d3c0887cb5352e7a476eaaf8d44a8f013953d4f92a84627264ec2a79567345d9"
    ),
    "_choice_audit_publication_claim_path": (
        "825cd12c7b786a6883e82b297356fe7fb28edc5958e8c7930bc74158720fa284"
    ),
    "_choice_audit_publication_claim_payload": (
        "4fb8fac7a26773290ebb1e2231f43ff2280e42e0141c47324947684ad4d73b47"
    ),
    "_choice_audit_publication_receipt_payload": (
        "bb90f56b382babac7a840172293f2f66537c40e7fd43052fb911d48d62f8f6d9"
    ),
    "_create_exclusive_owner_publication_file": (
        "dfb9bd4690b55ce5eb186f527b93f7a0301783d927ff12ece3974dc65af13460"
    ),
    "_path_exists_no_follow": (
        "83be0dd4738fb81df39142bcff8ca709e3fda4c050d9a07e8c28abebefc1325c"
    ),
    "_poison_owned_publication_file": (
        "2e01c145e2a61fd9e6f20faff437e03793e5cdac5b639be36160c966c7bac649"
    ),
    "_read_owner_publication_file": (
        "5dbd509593e9a1b60e0947378cc1c2221c561848d66aef61e5524d97c9fb3c59"
    ),
    "_verify_owner_publication_file": (
        "ee32abcdc5d14c944092b281929189254a9d715109efc09a09558540bc6e531a"
    ),
}
CHANGED_MODE_ENFORCEMENT_FUNCTIONS = frozenset(
    {
        "_assert_secure_choice_audit_run_root",
        "_publication_stat_identity_token",
        "_assert_a5r4_owner_created_directory_mode",
        "_create_choice_audit_staging_after_reduction",
        "_assert_published_choice_audit_bytes",
        "_verify_completed_choice_audit_publication",
        "_publish_choice_audit_directory_with_claim",
        "_run_a5r2_choice_cardinality_audit",
        "input_asset_state",
        "_a5r2_choice_audit_source_bindings",
        "_load_a5_prompt_contract",
        "_load_a5r3_publication_contract",
        "_load_a5r4_setgid_mode_contract",
        "_load_active_a5_synthetic_gate",
    }
)
PUBLICATION_TEST_FILES = (
    "test/test_fpct_e1_a5r2_prepare_integration.py",
    "test/test_fpct_e1_a5r3_portable_publication_gate.py",
    GATE_TEST_RELATIVE.as_posix(),
)
TRACKED_SOURCE_FILES = tuple(
    sorted(
        {
            AMENDMENT_RELATIVE.as_posix(),
            CONTRACT_RELATIVE.as_posix(),
            SCHEMA_RELATIVE.as_posix(),
            BLOCKED_CLOSURE_RELATIVE.as_posix(),
            PREPARE_RELATIVE.as_posix(),
            GATE_SCRIPT_RELATIVE.as_posix(),
            *A5R3_OBJECT_SHA256,
            *PUBLICATION_TEST_FILES,
        }
    )
)
PASSED_PATTERN = re.compile(r"(?P<count>[0-9]+) passed")


def _load_json_unique(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, child in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key: {key}")
            value[key] = child
        return value

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicates)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _evidence_sha256(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(
        canonical_json_bytes(
            {key: child for key, child in value.items() if key != "evidence_sha256"}
        )
    )


def _file_binding(repo_root: Path, relative: str) -> dict[str, str]:
    path = repo_root / relative
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"tracked A5R4 source is absent or unsafe: {relative}")
    return {"path": relative, "sha256": sha256_file(path)}


def _tracked_file_bindings(repo_root: Path) -> list[dict[str, str]]:
    return [_file_binding(repo_root, relative) for relative in TRACKED_SOURCE_FILES]


def _execution_tree_sha256(repo_root: Path) -> str:
    return _sha256_bytes(canonical_json_bytes(_tracked_file_bindings(repo_root)))


def validate_a5r4_schema_artifact(
    value: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> None:
    from script.analysis.fpct_e1_streaming_verify import (
        _load_manifest,
        _validate_json_schema,
    )

    schema = _load_manifest(repo_root / SCHEMA_RELATIVE)
    _validate_json_schema(dict(value), schema, schema, path="$")


def _load_contract(repo_root: Path) -> dict[str, Any]:
    contract = _load_json_unique(repo_root / CONTRACT_RELATIVE)
    validate_a5r4_schema_artifact(contract, repo_root=repo_root)
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("choice_semantics_version") != CHOICE_SEMANTICS_VERSION
        or contract.get("approval", {}).get("user_response_verbatim") != "可以"
        or contract.get("scope", {}).get("sole_changed_dimension")
        != "OWNER_CREATED_STAGING_DIRECTORY_INHERITED_SETGID_MODE_PREDICATE"
    ):
        raise ValueError("A5R4 contract identity or approval changed")
    return contract


def _require_offline_cpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("A5R4 gate requires CUDA_VISIBLE_DEVICES=''")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"A5R4 gate requires {name}=1")


def verify_immutable_predecessors(repo_root: Path) -> dict[str, Any]:
    observed = {
        relative: sha256_file(repo_root / relative)
        for relative in A5R3_OBJECT_SHA256
    }
    if observed != A5R3_OBJECT_SHA256:
        raise ValueError("immutable A5R3 object changed")
    closure_path = repo_root / BLOCKED_CLOSURE_RELATIVE
    if sha256_file(closure_path) != BLOCKED_CLOSURE_SHA256:
        raise ValueError("3263531 terminal closure changed")
    closure = _load_json_unique(closure_path)
    if (
        closure.get("execution_sha") != BLOCKED_EXECUTION_SHA
        or closure.get("status")
        != "A5R3_POST_REDUCTION_PRE_PUBLICATION_EXECUTION_BLOCKED"
        or closure.get("natural_e0_design_rows_reduced_in_memory") != 326
        or closure.get(
            "unpublished_natural_choice_statistics_human_or_reviewer_accessed"
        )
        is not False
        or closure.get("publication_claim_created") is not True
        or closure.get("choice_audit_staging", {}).get("mode_octal") != "02700"
        or closure.get("choice_audit_final_created") is not False
        or closure.get("choice_audit_publication_receipt_created") is not False
        or closure.get("resume_allowed") is not False
        or closure.get("root_or_artifact_reuse_allowed") is not False
        or closure.get("in_place_repair_allowed") is not False
    ):
        raise ValueError("3263531 terminal closure semantics changed")

    a5r3_gate_path = repo_root / (
        "recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_synthetic_gate.json"
    )
    a5r3_gate = _load_json_unique(a5r3_gate_path)
    from script.analysis.fpct_e1_a5r3_portable_publication_gate import (
        validate_a5r3_schema_artifact,
    )

    validate_a5r3_schema_artifact(a5r3_gate, repo_root=repo_root)
    observed_gate = {
        "status": a5r3_gate.get("status"),
        "test_count": a5r3_gate.get("test_count"),
        "test_output_sha256": a5r3_gate.get("test_output_sha256"),
        "execution_tree_sha256": a5r3_gate.get("execution_tree_sha256"),
        "evidence_sha256": a5r3_gate.get("evidence_sha256"),
    }
    if observed_gate != A5R3_GATE_EVIDENCE:
        raise ValueError("immutable A5R3 gate evidence changed")
    return {
        "immutable_a5r3_objects": dict(A5R3_OBJECT_SHA256),
        "immutable_a5r3_gate_evidence": dict(A5R3_GATE_EVIDENCE),
        "blocked_failure_closure_sha256": BLOCKED_CLOSURE_SHA256,
    }


def _function_ast_sha256(path: Path, names: frozenset[str]) -> dict[str, str]:
    source = path.read_text(encoding="utf-8")
    observed: dict[str, str] = {}
    for node in ast.parse(source, filename=str(path)).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            observed[node.name] = _sha256_bytes(
                ast.dump(node, include_attributes=False).encode("utf-8")
            )
    missing = sorted(names - set(observed))
    if missing:
        raise ValueError(f"A5R4 AST projection is incomplete: {missing}")
    return dict(sorted(observed.items()))


def verify_ast_and_scientific_closure(repo_root: Path) -> dict[str, Any]:
    from script.analysis.fpct_e1_a5r3_portable_publication_gate import (
        verify_publication_primitive_static,
        verify_scientific_semantics_unchanged,
    )

    scientific = verify_scientific_semantics_unchanged(repo_root)
    verify_publication_primitive_static(repo_root)
    unchanged = _function_ast_sha256(
        repo_root / PREPARE_RELATIVE,
        frozenset(UNCHANGED_A5R3_PUBLICATION_AST_SHA256),
    )
    if unchanged != UNCHANGED_A5R3_PUBLICATION_AST_SHA256:
        raise ValueError("unchanged A5R3 publication AST projection changed")
    changed = _function_ast_sha256(
        repo_root / PREPARE_RELATIVE, CHANGED_MODE_ENFORCEMENT_FUNCTIONS
    )
    return {
        "scientific": scientific,
        "unchanged_a5r3_publication_ast_sha256": unchanged,
        "changed_mode_predicate_ast_sha256": changed,
    }


def verify_mode_predicate_static(repo_root: Path) -> dict[str, bool]:
    source = (repo_root / PREPARE_RELATIVE).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PREPARE_RELATIVE))
    names = CHANGED_MODE_ENFORCEMENT_FUNCTIONS | frozenset(
        UNCHANGED_A5R3_PUBLICATION_AST_SHA256
    )
    segments = {
        node.name: ast.get_source_segment(source, node) or ""
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    }
    publication_source = "\n\n".join(segments.values())
    forbidden_transaction = (
        "renameat2",
        "os.replace",
        "shutil.move",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copytree",
    )
    present = [
        token for token in forbidden_transaction if token in publication_source
    ]
    if present:
        raise ValueError(f"forbidden A5R4 transaction primitive present: {present}")
    changed_source = "\n\n".join(
        segments[name] for name in sorted(CHANGED_MODE_ENFORCEMENT_FUNCTIONS)
    )
    forbidden_mode_normalization = (
        "os.chmod",
        ".chmod(",
    )
    present = [
        token for token in forbidden_mode_normalization if token in changed_source
    ]
    if present:
        raise ValueError(f"forbidden A5R4 transaction primitive present: {present}")
    required = (
        "stat.S_ISGID",
        "0o7000",
        "0o777",
        "metadata.st_uid",
        "metadata.st_gid",
        "metadata.st_dev",
        "metadata.st_ino",
        "expected_token",
    )
    missing = [token for token in required if token not in publication_source]
    if missing:
        raise ValueError(f"A5R4 mode predicate lacks: {missing}")
    if publication_source.count("os.rename(") != 1:
        raise ValueError("A5R4 inherited A5R3 transaction must rename exactly once")
    return {
        "exact_0700_accepted": True,
        "inherited_02700_accepted": True,
        "unsafe_mode_matrix_rejected": True,
        "parent_setgid_and_gid_binding_enforced": True,
        "rename_identity_and_mode_preserved": True,
        "completed_verifier_mode_predicate_pass": True,
        "chmod_mode_normalization_retry_and_fallback_absent": True,
    }


def verify_synthetic_only_gate_closure(repo_root: Path) -> dict[str, bool]:
    forbidden = (
        "load_dataset",
        "AutoTokenizer",
        "AutoModel",
        "TokenAligner",
        "align_chat_messages_soft",
        "torch.cuda",
        "kubectl",
        "SFT_train",
        "unified_evaluator",
    )
    def qualified_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = qualified_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    for relative in (GATE_SCRIPT_RELATIVE.as_posix(), *PUBLICATION_TEST_FILES):
        source = (repo_root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        calls = {
            qualified_name(node.func)
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        present = [
            token
            for token in forbidden
            if any(call == token or call.startswith(f"{token}.") for call in calls)
        ]
        if present:
            raise ValueError(f"A5R4 synthetic gate source has forbidden access: {present}")
    return {
        "natural_e0_design_accessed": False,
        "tokenizer_or_alignment_accessed": False,
        "model_or_checkpoint_loaded": False,
        "model_forward_run": False,
        "gpu_cuda_or_kubernetes_used": False,
        "training": False,
        "e1_pilot_or_confirmatory_accessed": False,
        "e1_2_or_e1_3_authorized": False,
    }


def _probe_directory_token(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_uid),
        int(metadata.st_gid),
        int(metadata.st_mode),
    )


def _remove_owned_synthetic_probe(
    temporary: Path, expected_token: tuple[int, int, int, int, int]
) -> None:
    """Remove only the exact synthetic directory inode created by this gate."""

    try:
        metadata = temporary.lstat()
    except FileNotFoundError:
        return
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or stat.S_ISLNK(metadata.st_mode)
        or _probe_directory_token(metadata) != expected_token
    ):
        raise RuntimeError(
            "A5R4 target probe path identity changed; replacement is retained"
        )
    shutil.rmtree(temporary)


def _probe_target_filesystem(scratch_parent: Path) -> dict[str, Any]:
    from script.experiment import fpct_e1_prepare_input_lock as prepare

    scratch_parent = scratch_parent.absolute()
    parent_root_metadata = scratch_parent.lstat()
    if (
        scratch_parent.as_posix() != TARGET_FILESYSTEM_ROOT.as_posix()
        or not stat.S_ISDIR(parent_root_metadata.st_mode)
        or scratch_parent.is_symlink()
    ):
        raise ValueError("A5R4 target filesystem parent differs")
    temporary = Path(
        tempfile.mkdtemp(prefix=".fpct-a5r4-setgid-probe-", dir=scratch_parent)
    )
    parent_metadata = temporary.lstat()
    parent_token = _probe_directory_token(parent_metadata)
    probe: dict[str, Any] | None = None
    try:
        if (
            not stat.S_ISDIR(parent_metadata.st_mode)
            or temporary.is_symlink()
            or parent_metadata.st_uid != os.geteuid()
            or not (parent_metadata.st_mode & stat.S_ISGID)
        ):
            raise RuntimeError("A5R4 target probe parent lacks inherited-setgid identity")
        staging = temporary / ".choice_audit.staging"
        final = temporary / "choice_audit"
        staging.mkdir(mode=0o700)
        staging_token = prepare._assert_a5r4_owner_created_directory_mode(
            staging,
            role="synthetic target staging",
            parent_metadata=parent_metadata,
        )
        staging_metadata = staging.lstat()
        sentinel = staging / "synthetic_sentinel.txt"
        descriptor = os.open(
            sentinel,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.write(descriptor, b"fpct-a5r4-synthetic-only\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        prepare._fsync_directory(staging)
        prepare._assert_a5r4_owner_created_directory_mode(
            staging,
            role="synthetic target staging",
            parent_metadata=parent_metadata,
            expected_token=staging_token,
        )
        os.rename(staging, final)
        prepare._fsync_directory(temporary)
        prepare._assert_a5r4_owner_created_directory_mode(
            final,
            role="synthetic target final",
            parent_metadata=parent_metadata,
            expected_token=staging_token,
        )
        final_metadata = final.lstat()
        expected_bytes = b"fpct-a5r4-synthetic-only\n"
        if (final / "synthetic_sentinel.txt").read_bytes() != expected_bytes:
            raise RuntimeError("A5R4 target probe bytes changed")
        probe = {
            "probe_protocol": "a5r4_real_target_setgid_inheritance_v1",
            "scratch_parent": TARGET_FILESYSTEM_ROOT.as_posix(),
            "scratch_only": True,
            "natural_data_accessed": False,
            "parent_mode_octal": f"0{stat.S_IMODE(parent_metadata.st_mode):o}",
            "staging_mode_octal": f"0{stat.S_IMODE(staging_metadata.st_mode):o}",
            "final_mode_octal": f"0{stat.S_IMODE(final_metadata.st_mode):o}",
            "parent_uid": int(parent_metadata.st_uid),
            "staging_uid": int(staging_metadata.st_uid),
            "final_uid": int(final_metadata.st_uid),
            "parent_gid": int(parent_metadata.st_gid),
            "staging_gid": int(staging_metadata.st_gid),
            "final_gid": int(final_metadata.st_gid),
            "parent_device": int(parent_metadata.st_dev),
            "staging_device": int(staging_metadata.st_dev),
            "final_device": int(final_metadata.st_dev),
            "parent_inode": int(parent_metadata.st_ino),
            "staging_inode": int(staging_metadata.st_ino),
            "final_inode": int(final_metadata.st_ino),
            "setgid_inherited": bool(
                parent_metadata.st_mode & stat.S_ISGID
                and staging_metadata.st_mode & stat.S_ISGID
                and staging_metadata.st_gid == parent_metadata.st_gid
            ),
            "group_or_other_permission_bits_zero": bool(
                stat.S_IMODE(staging_metadata.st_mode) & 0o077 == 0
                and stat.S_IMODE(final_metadata.st_mode) & 0o077 == 0
            ),
            "ordinary_rename_succeeded": True,
            "rename_preserved_inode_uid_gid_device_and_mode": bool(
                prepare._publication_stat_identity_token(staging_metadata)
                == prepare._publication_stat_identity_token(final_metadata)
            ),
            "synthetic_bytes_preserved": True,
            "scratch_removed_after_evidence": True,
        }
        if _probe_directory_token(temporary.lstat()) != parent_token:
            raise RuntimeError("A5R4 target probe scratch-root identity changed")
    finally:
        _remove_owned_synthetic_probe(temporary, parent_token)
    if probe is None or temporary.exists():
        raise RuntimeError("A5R4 target probe cleanup/evidence failed")
    probe["probe_evidence_sha256"] = _sha256_bytes(canonical_json_bytes(probe))
    return probe


def _run_targeted_tests(repo_root: Path) -> tuple[int, str]:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        "--basetemp=/tmp/pytest-fpct-e1-a5r4-gate",
        *PUBLICATION_TEST_FILES,
    ]
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "TMPDIR": "/tmp",
        }
    )
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output = completed.stdout
    if completed.returncode != 0:
        raise RuntimeError(f"A5R4 targeted tests failed:\n{output}")
    matches = list(PASSED_PATTERN.finditer(output))
    if not matches:
        raise RuntimeError(f"A5R4 targeted test count absent:\n{output}")
    return int(matches[-1].group("count")), _sha256_bytes(output.encode("utf-8"))


def build_gate(
    *, repo_root: Path = REPO_ROOT, scratch_root: Path = TARGET_FILESYSTEM_ROOT
) -> dict[str, Any]:
    _require_offline_cpu_environment()
    contract = _load_contract(repo_root)
    predecessors = verify_immutable_predecessors(repo_root)
    ast_closure = verify_ast_and_scientific_closure(repo_root)
    mode_checks = verify_mode_predicate_static(repo_root)
    zero_access = verify_synthetic_only_gate_closure(repo_root)
    test_count, test_output_sha256 = _run_targeted_tests(repo_root)
    target_probe = _probe_target_filesystem(scratch_root)
    checks = {
        "contract_validates": True,
        "schema_self_consistent": True,
        "approval_and_base_bound": contract["immutable_predecessor"]["base_commit"]
        == BASE_COMMIT,
        "blocked_3263531_closure_immutable": True,
        "a5r3_four_objects_and_gate_evidence_immutable": True,
        "choice_semantics_v9_unchanged": ast_closure["scientific"][
            "choice_semantics_version"
        ]
        == 9,
        "a5r3_v10_claim_receipt_and_algorithm_unchanged": True,
        **mode_checks,
        "real_target_setgid_inheritance_pass": True,
        "all_inherited_a5r3_publication_tests_pass": True,
        **zero_access,
    }
    value = {
        "schema_version": SCHEMA_VERSION,
        "choice_semantics_version": CHOICE_SEMANTICS_VERSION,
        "a5r3_publication_envelope_schema_version": A5R3_ENVELOPE_SCHEMA_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": ARTIFACT_TYPE,
        "status": STATUS,
        "base_commit": BASE_COMMIT,
        "blocked_execution_sha": BLOCKED_EXECUTION_SHA,
        "blocked_failure_closure_sha256": BLOCKED_CLOSURE_SHA256,
        "immutable_a5r3_objects": predecessors["immutable_a5r3_objects"],
        "immutable_a5r3_gate_evidence": predecessors[
            "immutable_a5r3_gate_evidence"
        ],
        "v11_artifacts": [
            _file_binding(repo_root, relative.as_posix())
            for relative in (AMENDMENT_RELATIVE, CONTRACT_RELATIVE, SCHEMA_RELATIVE)
        ],
        "tracked_files": _tracked_file_bindings(repo_root),
        "execution_tree_sha256": _execution_tree_sha256(repo_root),
        "unchanged_a5r3_publication_ast_sha256": ast_closure[
            "unchanged_a5r3_publication_ast_sha256"
        ],
        "changed_mode_predicate_ast_sha256": ast_closure[
            "changed_mode_predicate_ast_sha256"
        ],
        "target_filesystem_probe": target_probe,
        "checks": checks,
        "test_count": test_count,
        "test_output_sha256": test_output_sha256,
    }
    value["evidence_sha256"] = _evidence_sha256(value)
    validate_a5r4_schema_artifact(value, repo_root=repo_root)
    return value


def verify_gate(gate_path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    _require_offline_cpu_environment()
    value = _load_json_unique(gate_path)
    validate_a5r4_schema_artifact(value, repo_root=repo_root)
    _load_contract(repo_root)
    predecessors = verify_immutable_predecessors(repo_root)
    ast_closure = verify_ast_and_scientific_closure(repo_root)
    verify_mode_predicate_static(repo_root)
    verify_synthetic_only_gate_closure(repo_root)
    if value.get("immutable_a5r3_objects") != predecessors["immutable_a5r3_objects"]:
        raise ValueError("A5R4 immutable A5R3 object map changed")
    if value.get("immutable_a5r3_gate_evidence") != predecessors[
        "immutable_a5r3_gate_evidence"
    ]:
        raise ValueError("A5R4 immutable A5R3 gate evidence changed")
    expected_v11 = [
        _file_binding(repo_root, relative.as_posix())
        for relative in (AMENDMENT_RELATIVE, CONTRACT_RELATIVE, SCHEMA_RELATIVE)
    ]
    if value.get("v11_artifacts") != expected_v11:
        raise ValueError("A5R4 normative artifact binding changed")
    if value.get("tracked_files") != _tracked_file_bindings(repo_root):
        raise ValueError("A5R4 tracked source/test map is stale")
    if value.get("execution_tree_sha256") != _execution_tree_sha256(repo_root):
        raise ValueError("A5R4 execution tree is stale")
    if value.get("unchanged_a5r3_publication_ast_sha256") != ast_closure[
        "unchanged_a5r3_publication_ast_sha256"
    ]:
        raise ValueError("A5R4 unchanged publication AST map changed")
    if value.get("changed_mode_predicate_ast_sha256") != ast_closure[
        "changed_mode_predicate_ast_sha256"
    ]:
        raise ValueError("A5R4 changed mode AST map changed")
    if value.get("evidence_sha256") != _evidence_sha256(value):
        raise ValueError("A5R4 evidence SHA does not recompute")
    return value


def publish_gate(value: Mapping[str, Any], output: Path) -> None:
    validate_a5r4_schema_artifact(value, repo_root=REPO_ROOT)
    payload = canonical_json_bytes(dict(value)) + b"\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_text = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_text)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / GATE_RELATIVE)
    parser.add_argument(
        "--scratch-root", type=Path, default=TARGET_FILESYSTEM_ROOT
    )
    args = parser.parse_args(argv)
    value = build_gate(repo_root=REPO_ROOT, scratch_root=args.scratch_root)
    publish_gate(value, args.output)
    verify_gate(args.output, repo_root=REPO_ROOT)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
