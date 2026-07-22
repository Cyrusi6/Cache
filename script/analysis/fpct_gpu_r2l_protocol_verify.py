#!/usr/bin/env python3
"""Verify the R2l function-level implementation boundary and protocol lock."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _mask_top_level_function(text: str, function_name: str) -> str:
    tree = ast.parse(text)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one top-level function {function_name!r}")
    node = matches[0]
    start = min([node.lineno] + [item.lineno for item in node.decorator_list]) - 1
    end = int(node.end_lineno)
    lines = text.splitlines(keepends=True)
    return "".join(lines[:start] + [f"<AUTHORIZED_FUNCTION:{function_name}>\n"] + lines[end:])


def _changed_paths(repo: Path, baseline: str) -> list[str]:
    tracked = git(repo, "diff", "--name-only", baseline, "--").splitlines()
    untracked = git(repo, "ls-files", "--others", "--exclude-standard").splitlines()
    return sorted({path for path in tracked + untracked if path})


def verify(repo: Path) -> dict[str, Any]:
    allow_path = repo / "recipe/eval_recipe/fpct_gpu_r2l/implementation_allowlist.json"
    truth_path = repo / "recipe/eval_recipe/fpct_gpu_r2l/semantic_map_truth_table.json"
    allow = json.loads(allow_path.read_text())
    truth = json.loads(truth_path.read_text())
    baseline = str(allow["baseline_commit"])
    if git(repo, "rev-parse", baseline) != baseline:
        raise ValueError("R2l baseline commit is unavailable")

    allowed_existing = set(allow["allowed_existing_non_scientific_files"])
    allowed_existing.update(allow["allowed_existing_scientific_functions"])
    prefixes = tuple(allow["allowed_new_prefixes"])
    changed = _changed_paths(repo, baseline)
    unexpected = [
        path
        for path in changed
        if path not in allowed_existing and not path.startswith(prefixes)
    ]
    if unexpected:
        raise ValueError(f"files outside R2l allowlist: {unexpected}")

    function_checks = {}
    for relative, functions in allow["allowed_existing_scientific_functions"].items():
        if len(functions) != 1:
            raise ValueError("R2l verifier currently requires one authorized function per file")
        name = functions[0]
        baseline_text = git(repo, "show", f"{baseline}:{relative}") + "\n"
        current_text = (repo / relative).read_text()
        masked_equal = _mask_top_level_function(
            baseline_text, name
        ) == _mask_top_level_function(current_text, name)
        if not masked_equal:
            raise ValueError(f"changes outside authorized function in {relative}")
        function_checks[relative] = {"authorized_function": name, "masked_equal": True}

    forbidden_checks = {}
    for relative, expected in allow["forbidden_file_sha256"].items():
        actual = sha256_file(repo / relative)
        if actual != expected:
            raise ValueError(f"forbidden file changed: {relative}")
        forbidden_checks[relative] = actual

    cases = {row["id"]: row for row in truth["cases"]}
    required_cases = {
        "native_outside_coverage",
        "covered_explicit_true",
        "covered_explicit_false",
        "covered_force_native_true",
        "covered_force_native_false",
        "covered_metadata_missing",
        "invalid_atom_native_semantics",
    }
    if set(cases) != required_cases:
        raise ValueError("semantic truth-table cases changed")
    if not cases["native_outside_coverage"]["expected_equivalent"]:
        raise ValueError("native outside coverage must be equivalent")
    if cases["covered_metadata_missing"]["expected_equivalent"]:
        raise ValueError("unknown sidecar metadata must fail closed")

    return {
        "schema_version": 1,
        "protocol_id": "fpct_gpu_r2l_protocol_verify_v1",
        "status": "GO",
        "baseline_commit": baseline,
        "head": git(repo, "rev-parse", "HEAD"),
        "changed_paths": changed,
        "function_boundary": function_checks,
        "forbidden_sha256": forbidden_checks,
        "truth_table_case_count": len(cases),
        "accuracy_or_correctness_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = verify(args.repo.resolve())
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
