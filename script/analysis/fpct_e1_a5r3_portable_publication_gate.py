#!/usr/bin/env python3
"""Build the CPU-only, pre-natural A5R3 portable-publication hard gate.

This module is intentionally synthetic-only.  It binds the immutable A5R2
scientific semantics and the terminal ``e765d493`` failure closure, verifies
that the only successor change is the choice-audit publication primitive,
executes the publication unit tests, and performs one ordinary-rename probe on
the target filesystem class.  It never opens an E0-design row and never loads
a tokenizer, alignment object, model, or checkpoint.
"""

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
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from script.experiment.fpct_e1_a5_prompt_provenance import (
    canonical_json_bytes,
    sha256_file,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = 10
CHOICE_SEMANTICS_VERSION = 9
PROTOCOL_ID = "fpct_e1_mechanism_audit_v10_a5r3_portable_publication"
BASE_COMMIT = "55476286c84abb6925d9c11a6cffd6ef80a6a58f"
BLOCKED_EXECUTION_SHA = "e765d493733d9eec94c152506a1c57781e26fb41"
BLOCKED_CLOSURE_SHA256 = (
    "f6715f9d86b23ccc265f6ead7b24af47acaf1e1ecd9eb14552af7940d68f798c"
)

AMENDMENT_RELATIVE = Path("FPCT_E1_A5R3_PORTABLE_PUBLICATION_AMENDMENT.md")
CONTRACT_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_contract.json"
)
SCHEMA_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_schema.json"
)
GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_a5r3_portable_publication_synthetic_gate.json"
)
BLOCKED_CLOSURE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/executions/e765d493/input_lock_failure_receipt.json"
)
PRENATURAL_FAILURE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/executions/b6109443/input_lock_failure_receipt.json"
)
PRENATURAL_FAILURE_SHA256 = (
    "efbecc09a9b5bb7933beb0c9bf87dc2dedf4eb17314eec3aa263f963de56ee56"
)
PREPARE_RELATIVE = Path("script/experiment/fpct_e1_prepare_input_lock.py")
PROVENANCE_RELATIVE = Path("script/experiment/fpct_e1_a5_prompt_provenance.py")

A5R2_OBJECT_SHA256 = {
    "FPCT_E1_A5R2_CHOICE_CARDINALITY_AMENDMENT.md": (
        "89753bcbdec66d07c36bcfc3a5c636e66704cddce5546ac0e321d7a9ea053384"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_contract.json": (
        "a4bdf4a229d26b367fb8ea7c90adf39d72e94c56daf4d095346bf673c5c2eb1b"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_schema.json": (
        "9cb387628e4b8fcf6c978e082b8c3380dbc62406c6aa460892c679872d656d7f"
    ),
    "recipe/eval_recipe/fpct_e1/e1_a5r2_choice_cardinality_synthetic_gate.json": (
        "cfc8c7da3bb074b0a4fcb23276af8d8404516b656337abb5c3034724272d2cd9"
    ),
}

# Full-file and AST bindings are taken from the terminal A5R2 closure commit.
# The full provenance module contains the parser, v9 taxonomy, correction
# actions, reductions and decision rules.  The selected prepare functions bind
# population construction and canonical group ordering while allowing the
# publication-only functions to change.
SCIENTIFIC_FILE_SHA256 = {
    PROVENANCE_RELATIVE.as_posix(): (
        "9a554e492af1af0442be561a6f7c16e93a6ed1153332cc9e5d3bd9774bab3390"
    ),
    "recipe/eval_recipe/fpct_e1/e1_data_split_manifest.json": (
        "030b4236ed9bec82b145227259733b32a8c76af63adf2fa0f1282e3638b5b11d"
    ),
    "recipe/eval_recipe/fpct_e0/exploratory_dev_manifest.json": (
        "25fe8c4dceeaa1174e1433a02ec86f312d909d7d58c3c9a8c7f2caa9d908216a"
    ),
}
SCIENTIFIC_AST_SHA256 = {
    "_a5r2_dataset_identity": "c8a3f28919309b772b7811bc75446482fdee7adc37371fda4d771e6e613bb944",
    "_load_a5r2_choice_audit_example": "a327a7be5d08b44600ce024dbcfad2014a43f175b5f0308773aa70825ea31bf9",
    "_a5r2_feature_schema_sha256": "040e872929206c1f03d40ae1f0b683eef1f337e36845fb29b98264baa037394e",
    "_a5r2_allowlisted_choice_row": "420164334e16302d9c07c28e9a4a3786b1455bbd7a38cf572681f0ec388d6fe4",
    "_a5r2_choice_descriptor_pair": "4821912b909d4db2927c67452626ba4721d4446ac4cd559b307f6eacc3f50383",
    "_load_frozen_a5r2_dev": "3d22533f990958a9ceaec652b57949efa625c1b75cb0c1c3b5a79de9494e93f4",
    "_a5r2_frozen_identity_checks": "286db0c1f5a861717c778fb18ccfc3376a5a3eaf4d15c910f461068e258ff0a4",
    "_a5r2_pre_audit_asset_bindings": "ca6e88f514d35bf1975d07294f1e059dd4ad986a02e05e18caeb472146aab166",
    "_read_choice_cardinality_audit_ledger": "725a7abf7b7bc3328d08a589c74d73621dca329eaf243386a3eb329bbcc0f1d0",
    "_choice_audit_artifact_descriptor": "fee3e51c2dfe92e67ed95211b448a3426a853fe7f8700bb01318338fe7da8e2c",
    "_a5r2_evidence_sha256": "d9bad669c2523f7557423f2a62e6b86d7948d554d7905e7cb7f2f2e3ae851696",
}

PUBLICATION_FUNCTIONS = frozenset(
    {
        "_path_exists_no_follow",
        "_choice_audit_publication_claim_path",
        "_assert_secure_choice_audit_run_root",
        "_read_owner_publication_file",
        "_verify_owner_publication_file",
        "_poison_owned_publication_file",
        "_create_exclusive_owner_publication_file",
        "_choice_audit_publication_claim_payload",
        "_assert_choice_audit_publication_state_is_fresh",
        "_acquire_choice_audit_publication_claim",
        "_assert_published_choice_audit_bytes",
        "_choice_audit_publication_receipt_payload",
        "_verify_completed_choice_audit_publication",
        "_publish_choice_audit_directory_with_claim",
    }
)

PUBLICATION_TEST_FILES = (
    "test/test_fpct_e1_a5r2_prepare_integration.py",
    "test/test_fpct_e1_a5r3_portable_publication_gate.py",
)
TRACKED_SOURCE_FILES = tuple(
    sorted(
        {
            AMENDMENT_RELATIVE.as_posix(),
            CONTRACT_RELATIVE.as_posix(),
            SCHEMA_RELATIVE.as_posix(),
            BLOCKED_CLOSURE_RELATIVE.as_posix(),
            PRENATURAL_FAILURE_RELATIVE.as_posix(),
            PREPARE_RELATIVE.as_posix(),
            PROVENANCE_RELATIVE.as_posix(),
            "script/analysis/fpct_e1_a5r3_portable_publication_gate.py",
            *A5R2_OBJECT_SHA256,
            *SCIENTIFIC_FILE_SHA256,
            *PUBLICATION_TEST_FILES,
        }
    )
)
PASSED_PATTERN = re.compile(r"(?P<count>[0-9]+) passed")
TARGET_FILESYSTEM_ROOT = Path("/netdisk/lijunsi/fpct-e1")


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


def validate_a5r3_schema_artifact(
    value: Mapping[str, Any], *, repo_root: Path = REPO_ROOT
) -> None:
    from script.analysis.fpct_e1_streaming_verify import (
        _load_manifest,
        _validate_json_schema,
    )

    schema = _load_manifest(repo_root / SCHEMA_RELATIVE)
    _validate_json_schema(dict(value), schema, schema, path="$")


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
        raise ValueError(f"tracked A5R3 source is absent or unsafe: {relative}")
    return {"path": relative, "sha256": sha256_file(path)}


def _tracked_file_bindings(repo_root: Path) -> list[dict[str, str]]:
    return [_file_binding(repo_root, relative) for relative in TRACKED_SOURCE_FILES]


def _execution_tree_sha256(repo_root: Path) -> str:
    return _sha256_bytes(canonical_json_bytes(_tracked_file_bindings(repo_root)))


def _require_offline_cpu_environment() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("A5R3 gate requires CUDA_VISIBLE_DEVICES=''")
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        if os.environ.get(name) != "1":
            raise RuntimeError(f"A5R3 gate requires {name}=1")


def _verify_immutable_predecessors(repo_root: Path) -> dict[str, str]:
    observed = {
        relative: sha256_file(repo_root / relative)
        for relative in A5R2_OBJECT_SHA256
    }
    if observed != A5R2_OBJECT_SHA256:
        raise ValueError("immutable A5R2 four-object predecessor changed")
    closure_path = repo_root / BLOCKED_CLOSURE_RELATIVE
    if sha256_file(closure_path) != BLOCKED_CLOSURE_SHA256:
        raise ValueError("immutable e765d493 failure closure changed")
    closure = _load_json_unique(closure_path)
    if (
        closure.get("execution_sha") != BLOCKED_EXECUTION_SHA
        or closure.get("resume_allowed") is not False
        or closure.get("reuse_allowed") is not False
    ):
        raise ValueError("e765d493 terminal disposition changed")
    pre_natural_path = repo_root / PRENATURAL_FAILURE_RELATIVE
    if sha256_file(pre_natural_path) != PRENATURAL_FAILURE_SHA256:
        raise ValueError("immutable b6109443 pre-natural closure changed")
    pre_natural = _load_json_unique(pre_natural_path)
    if (
        pre_natural.get("execution_sha")
        != "b6109443e1b4c35eef73c322f30e9aec37194ce7"
        or pre_natural.get("natural_e0_design_rows_read") != 0
        or pre_natural.get("publication_claim_created") is not False
        or pre_natural.get("resume_allowed") is not False
        or pre_natural.get("root_or_artifact_reuse_allowed") is not False
    ):
        raise ValueError("b6109443 pre-natural terminal disposition changed")
    return observed


def _function_ast_sha256(path: Path, names: set[str] | frozenset[str]) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    observed: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            observed[node.name] = _sha256_bytes(
                ast.dump(node, include_attributes=False).encode("utf-8")
            )
    missing = sorted(set(names) - set(observed))
    if missing:
        raise ValueError(f"A5R3 AST projection is incomplete: {missing}")
    return dict(sorted(observed.items()))


def verify_scientific_semantics_unchanged(repo_root: Path) -> dict[str, Any]:
    observed_files = {
        relative: sha256_file(repo_root / relative)
        for relative in SCIENTIFIC_FILE_SHA256
    }
    if observed_files != SCIENTIFIC_FILE_SHA256:
        raise ValueError("A5R3 scientific source/population bytes changed")
    ast_bindings = _function_ast_sha256(
        repo_root / PREPARE_RELATIVE, frozenset(SCIENTIFIC_AST_SHA256)
    )
    if ast_bindings != dict(sorted(SCIENTIFIC_AST_SHA256.items())):
        raise ValueError("A5R3 population/parser AST projection changed")
    return {
        "choice_semantics_version": CHOICE_SEMANTICS_VERSION,
        "scientific_file_sha256": observed_files,
        "population_prepare_ast_sha256": ast_bindings,
    }


def _publication_source(repo_root: Path) -> str:
    path = repo_root / PREPARE_RELATIVE
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    segments: list[str] = []
    found: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in PUBLICATION_FUNCTIONS:
            found.add(node.name)
            segments.append(ast.get_source_segment(source, node) or "")
    missing = sorted(PUBLICATION_FUNCTIONS - found)
    if missing:
        raise ValueError(f"A5R3 publication function closure is incomplete: {missing}")
    return "\n\n".join(segments)


def verify_publication_primitive_static(repo_root: Path) -> dict[str, bool]:
    from script.experiment import fpct_e1_prepare_input_lock as prepare

    if prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_NAME != (
        ".choice_audit.publication.claim.json"
    ):
        raise ValueError("A5R3 canonical claim basename changed")
    if prepare.CHOICE_AUDIT_PUBLICATION_CLAIM_PROTOCOL_ID != PROTOCOL_ID:
        raise ValueError("A5R3 claim/receipt protocol ID changed")
    if prepare.CHOICE_AUDIT_ROOT_NAME != "choice_audit":
        raise ValueError("A5R3 final basename changed")
    if prepare.CHOICE_AUDIT_PUBLICATION_RECEIPT_NAME != (
        "choice_audit_publication_receipt.json"
    ):
        raise ValueError("A5R3 durable receipt basename changed")

    source = _publication_source(repo_root)
    if source.count('".choice_audit.staging"') < 2:
        raise ValueError("A5R3 fixed staging basename is not enforced")
    forbidden = (
        "renameat2",
        "os.replace",
        "shutil.move",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copytree",
        "copy_delete",
        ".unlink(",
    )
    present = [token for token in forbidden if token in source]
    if present:
        raise ValueError(f"forbidden A5R3 publication primitive present: {present}")
    if source.count("os.rename(") != 1:
        raise ValueError("A5R3 publication must contain exactly one ordinary rename")
    for required in (
        "os.O_CREAT",
        "os.O_EXCL",
        '"O_NOFOLLOW"',
        "os.fsync",
        "stat.S_ISREG",
        "st_nlink",
        "st_uid",
        "st_mode",
        "st_dev",
        "st_ino",
    ):
        if required not in source:
            raise ValueError(f"A5R3 publication primitive lacks {required}")

    prepare_source = (repo_root / PREPARE_RELATIVE).read_text(encoding="utf-8")
    prepare_tree = ast.parse(prepare_source, filename=str(PREPARE_RELATIVE))
    functions = {
        node.name: node
        for node in prepare_tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    audit_node = functions.get("_run_a5r2_choice_cardinality_audit")
    acquire_node = functions.get("_acquire_choice_audit_publication_claim")
    staging_node = functions.get("_create_choice_audit_staging_after_reduction")
    receipt_node = functions.get("_choice_audit_publication_receipt_payload")
    claim_node = functions.get("_choice_audit_publication_claim_payload")
    if not all((audit_node, acquire_node, staging_node, receipt_node, claim_node)):
        raise ValueError("A5R3 publication ordering closure is incomplete")

    def call_lines(node: ast.AST, terminal_name: str) -> list[int]:
        return sorted(
            int(child.lineno)
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and qualified_call_name(child.func).split(".")[-1] == terminal_name
        )

    def node_source(node: ast.AST) -> str:
        return ast.get_source_segment(prepare_source, node) or ""

    def qualified_call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = qualified_call_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    acquire_lines = call_lines(audit_node, "_acquire_choice_audit_publication_claim")
    natural_lines = call_lines(audit_node, "_load_a5r2_choice_audit_example")
    staging_lines = call_lines(
        audit_node, "_create_choice_audit_staging_after_reduction"
    )
    if (
        len(acquire_lines) != 1
        or len(natural_lines) != 1
        or len(staging_lines) != 1
        or acquire_lines[0] >= natural_lines[0]
        or staging_lines[0] <= natural_lines[0]
    ):
        raise ValueError(
            "A5R3 claim/natural-row/staging ordering is not prospective"
        )
    acquire_source = node_source(acquire_node)
    staging_source = node_source(staging_node)
    if (
        "_create_exclusive_owner_publication_file(" not in acquire_source
        or "staging_root.mkdir(" in acquire_source
        or "staging_root.mkdir(" not in staging_source
    ):
        raise ValueError("A5R3 claim/staging responsibilities are not separated")
    claim_source = node_source(claim_node)
    if any(token in claim_source for token in ("ledger_sha", "summary_sha", "lock_sha")):
        raise ValueError("A5R3 pre-row-one claim contains result-dependent hashes")
    receipt_source = node_source(receipt_node)
    for required in (
        '"publication_commit_point_reached": True',
        '"claim_sha256"',
        '"ledger_artifact"',
        '"summary_artifact"',
        '"lock_artifact"',
    ):
        if required not in receipt_source:
            raise ValueError(f"A5R3 durable receipt lacks {required}")
    return {
        "ordinary_rename_exactly_once": True,
        "renameat2_replace_copy_delete_absent": True,
        "exclusive_nofollow_owner_files": True,
        "runtime_inode_device_token_checked": True,
        "retained_claim_and_durable_receipt_supported": True,
    }


def verify_synthetic_only_gate_closure(repo_root: Path) -> dict[str, bool]:
    relative_files = (
        "script/analysis/fpct_e1_a5r3_portable_publication_gate.py",
        *PUBLICATION_TEST_FILES,
    )
    forbidden_names = {
        "load_dataset",
        "AutoTokenizer",
        "AutoModel",
        "TokenAligner",
        "align_chat_messages_soft",
        "torch.cuda",
        "kubectl",
        "SFT_train",
        "unified_evaluator",
    }

    def qualified_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            parent = qualified_name(node.value)
            return f"{parent}.{node.attr}" if parent else node.attr
        return ""

    for relative in relative_files:
        source = (repo_root / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                names.add(qualified_name(node.func))
            elif isinstance(node, ast.Import):
                names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                names.update(
                    f"{module}.{alias.name}" if module else alias.name
                    for alias in node.names
                )
        hits = sorted(
            candidate
            for candidate in names
            if any(
                candidate == forbidden
                or candidate.endswith(f".{forbidden}")
                or candidate.startswith(f"{forbidden}.")
                for forbidden in forbidden_names
            )
        )
        if hits:
            raise ValueError(f"non-synthetic access token in {relative}: {hits}")
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


def _probe_target_filesystem(scratch_root: Path) -> dict[str, Any]:
    scratch_root = scratch_root.absolute()
    metadata = scratch_root.lstat()
    if not stat.S_ISDIR(metadata.st_mode) or scratch_root.is_symlink():
        raise ValueError("A5R3 target scratch root is not a real directory")
    probe_summary = {
        "probe_protocol": "a5r3_synthetic_ordinary_same_filesystem_rename_v1",
        "source_basename": ".choice_audit.staging",
        "destination_basename": "choice_audit",
        "sentinel_sha256": _sha256_bytes(b"fpct-a5r3-synthetic-only\n"),
        "same_device": True,
        "ordinary_rename_succeeded": True,
        "natural_data_accessed": False,
        "scratch_only": True,
    }
    temporary = Path(
        tempfile.mkdtemp(prefix=".fpct-a5r3-publication-probe-", dir=scratch_root)
    )
    try:
        staging = temporary / ".choice_audit.staging"
        final = temporary / "choice_audit"
        staging.mkdir(mode=0o700)
        sentinel = staging / "synthetic_sentinel.txt"
        descriptor = os.open(
            sentinel,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            os.write(descriptor, b"fpct-a5r3-synthetic-only\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        directory_descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
        devices = {temporary.stat().st_dev, staging.stat().st_dev, sentinel.stat().st_dev}
        if len(devices) != 1:
            raise RuntimeError("A5R3 scratch probe is not on one filesystem")
        os.rename(staging, final)
        parent_descriptor = os.open(temporary, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        if (
            staging.exists()
            or not final.is_dir()
            or sha256_file(final / "synthetic_sentinel.txt")
            != probe_summary["sentinel_sha256"]
        ):
            raise RuntimeError("A5R3 ordinary rename probe did not preserve bytes")
    finally:
        shutil.rmtree(temporary)
    return {
        "natural_data_accessed": False,
        "ordinary_rename_succeeded": True,
        "same_device": True,
        "scratch_only": True,
        "probe_evidence_sha256": _sha256_bytes(canonical_json_bytes(probe_summary)),
    }


def _load_contract(repo_root: Path) -> dict[str, Any]:
    contract = _load_json_unique(repo_root / CONTRACT_RELATIVE)
    validate_a5r3_schema_artifact(contract, repo_root=repo_root)
    if (
        contract.get("schema_version") != SCHEMA_VERSION
        or contract.get("choice_semantics_version") != CHOICE_SEMANTICS_VERSION
        or contract.get("protocol_id") != PROTOCOL_ID
        or contract.get("approval", {}).get("approved") is not True
        or contract.get("approval", {}).get("user_response_verbatim") != "可以"
        or contract.get("immutable_predecessor", {}).get("base_commit") != BASE_COMMIT
        or contract.get("immutable_predecessor", {}).get("blocked_execution_sha")
        != BLOCKED_EXECUTION_SHA
        or contract.get("immutable_predecessor", {}).get("closure_sha256")
        != BLOCKED_CLOSURE_SHA256
        or contract.get("immutable_a5r2_objects") != A5R2_OBJECT_SHA256
    ):
        raise ValueError("A5R3 approval/version/predecessor identity changed")
    unchanged = contract.get("unchanged_scientific_contract", {})
    if (
        unchanged.get("distinct_content_groups") != 326
        or unchanged.get("task_counts")
        != {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
        or unchanged.get("restart_group_ordinal_one_based") != 1
        or any(
            unchanged.get(name) is not False
            for name in (
                "population_or_order_changed",
                "choice_parsing_or_cardinality_changed",
                "taxonomy_or_correction_actions_changed",
                "threshold_changed",
                "prompt_tokenizer_or_alignment_changed",
                "operator_or_estimand_changed",
                "a5r2_in_memory_reduction_may_be_used",
            )
        )
    ):
        raise ValueError("A5R3 unchanged scientific contract drifted")
    return contract


def _run_tests(repo_root: Path) -> tuple[int, str]:
    missing = [relative for relative in PUBLICATION_TEST_FILES if not (repo_root / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"A5R3 publication test closure is incomplete: {missing}")
    base_temp = repo_root / "local/tmp/pytest-e1-a5r3-portable-publication-gate"
    base_temp.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        f"--basetemp={base_temp}",
        *PUBLICATION_TEST_FILES,
    ]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
    }
    completed = subprocess.run(
        command,
        cwd=repo_root,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    matches = list(PASSED_PATTERN.finditer(output))
    if not matches:
        raise RuntimeError("A5R3 pytest output does not report a passed count")
    return int(matches[-1].group("count")), _sha256_bytes(output.encode("utf-8"))


def build_gate(
    *, repo_root: Path = REPO_ROOT, scratch_root: Path = TARGET_FILESYSTEM_ROOT
) -> dict[str, Any]:
    _require_offline_cpu_environment()
    predecessors = _verify_immutable_predecessors(repo_root)
    _load_contract(repo_root)
    verify_scientific_semantics_unchanged(repo_root)
    verify_publication_primitive_static(repo_root)
    synthetic_scope = verify_synthetic_only_gate_closure(repo_root)
    if synthetic_scope.get("tokenizer_or_alignment_accessed") is not False:
        raise ValueError("A5R3 synthetic gate touched tokenizer/alignment state")
    target_probe = _probe_target_filesystem(scratch_root)
    passed, output_sha256 = _run_tests(repo_root)
    v10_artifacts = [
        _file_binding(repo_root, relative.as_posix())
        for relative in (AMENDMENT_RELATIVE, CONTRACT_RELATIVE, SCHEMA_RELATIVE)
    ]
    checks = {
        "contract_validates": True,
        "schema_self_consistent": True,
        "approval_and_base_bound": True,
        "a5r2_closure_and_objects_immutable": True,
        "publication_primitive_is_only_change": True,
        "target_filesystem_scratch_ordinary_rename_pass": True,
        "renameat2_replace_copy_delete_absent": True,
        "exclusive_claim_concurrency_pass": True,
        "preexisting_regular_and_symlink_paths_refused": True,
        "claim_identity_owner_link_and_byte_tamper_refused": True,
        "staged_and_final_tamper_refused": True,
        "same_device_enforced": True,
        "rename_fsync_receipt_failures_terminal": True,
        "all_crash_points_fail_closed": True,
        "successful_final_equals_verified_staging": True,
        "retained_claim_and_success_receipt_cross_bindings_pass": True,
        "publisher_reentry_is_read_only_and_refuses_mutation": True,
        "claim_loser_read_only_no_blocked_pollution": True,
        **synthetic_scope,
    }
    contract = _load_contract(repo_root)
    value: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "choice_semantics_version": CHOICE_SEMANTICS_VERSION,
        "protocol_id": PROTOCOL_ID,
        "artifact_type": "a5r3_portable_publication_synthetic_gate",
        "status": "GO_PRE_NATURAL_A5R3_PORTABLE_PUBLICATION_HARD_GATE",
        "base_commit": BASE_COMMIT,
        "blocked_execution_sha": BLOCKED_EXECUTION_SHA,
        "blocked_failure_closure_sha256": BLOCKED_CLOSURE_SHA256,
        "a5r2_objects": predecessors,
        "v10_artifacts": v10_artifacts,
        "tracked_files": _tracked_file_bindings(repo_root),
        "execution_tree_sha256": _execution_tree_sha256(repo_root),
        "target_filesystem_probe": target_probe,
        "checks": checks,
        "crash_points_tested": contract["crash_points"],
        "test_count": passed,
        "test_output_sha256": output_sha256,
    }
    value["evidence_sha256"] = _evidence_sha256(value)
    validate_a5r3_schema_artifact(value, repo_root=repo_root)
    return value


def verify_gate(gate_path: Path, *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    _require_offline_cpu_environment()
    value = _load_json_unique(gate_path)
    validate_a5r3_schema_artifact(value, repo_root=repo_root)
    if value.get("choice_semantics_version") != CHOICE_SEMANTICS_VERSION:
        raise ValueError("A5R3 gate choice-semantics version changed")
    if value.get("a5r2_objects") != _verify_immutable_predecessors(repo_root):
        raise ValueError("A5R3 immutable predecessor map changed")
    _load_contract(repo_root)
    verify_scientific_semantics_unchanged(repo_root)
    verify_publication_primitive_static(repo_root)
    verify_synthetic_only_gate_closure(repo_root)
    expected_v10 = [
        _file_binding(repo_root, relative.as_posix())
        for relative in (AMENDMENT_RELATIVE, CONTRACT_RELATIVE, SCHEMA_RELATIVE)
    ]
    if value.get("v10_artifacts") != expected_v10:
        raise ValueError("A5R3 normative artifact binding changed")
    if value.get("tracked_files") != _tracked_file_bindings(repo_root):
        raise ValueError("A5R3 tracked source/test map is stale")
    if value.get("execution_tree_sha256") != _execution_tree_sha256(repo_root):
        raise ValueError("A5R3 execution tree is stale")
    if value.get("evidence_sha256") != _evidence_sha256(value):
        raise ValueError("A5R3 evidence SHA does not recompute")
    return value


def publish_gate(value: Mapping[str, Any], output: Path) -> None:
    validate_a5r3_schema_artifact(value, repo_root=REPO_ROOT)
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
    parser.add_argument("--scratch-root", type=Path, default=TARGET_FILESYSTEM_ROOT)
    args = parser.parse_args(argv)
    value = build_gate(repo_root=REPO_ROOT, scratch_root=args.scratch_root)
    publish_gate(value, args.output)
    verify_gate(args.output, repo_root=REPO_ROOT)
    print(json.dumps(value, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
