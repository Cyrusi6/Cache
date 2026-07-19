from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from script.analysis import phase2a_1_selector_killtest as selector


CSV_FIELDS = [
    "subject",
    "question_id",
    "question",
    *selector.CHOICE_FIELDS,
    "is_correct",
    *selector.PRIMARY_FEATURES,
]


def _synthetic_row(
    question_id: str,
    *,
    correctness: str,
    candidate_count: str = "2",
) -> dict[str, str]:
    row = {
        "subject": "synthetic",
        "question_id": question_id,
        "question": f"Synthetic question {question_id}",
        "is_correct": correctness,
        "cot_input_length": "32",
        "candidate_count": candidate_count,
        "candidate_count_max": "4",
        "one_to_many_rate": "0.25",
        "boundary_mismatch": "0.10",
    }
    row.update({choice: f"choice-{choice}" for choice in selector.CHOICE_FIELDS})
    return row


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _synthetic_layout(
    tmp_path: Path,
) -> tuple[selector.SourceLayout, dict[str, Any], dict[str, Any]]:
    fit_receiver = _synthetic_row("fit", correctness="true")
    test_receiver = _synthetic_row("test", correctness="INVALID_TEST_OUTCOME")
    fit_b6 = _synthetic_row("fit", correctness="false")
    test_b6 = _synthetic_row(
        "test",
        correctness="INVALID_TEST_OUTCOME",
        candidate_count="INVALID_TEST_FEATURE",
    )

    receiver_path = tmp_path / "receiver.csv"
    b6_path = tmp_path / "b6.csv"
    _write_rows(receiver_path, [fit_receiver, test_receiver])
    _write_rows(b6_path, [fit_b6, test_b6])

    fit_hash = selector._content_hash(fit_receiver)
    test_hash = selector._content_hash(test_receiver)
    split_manifest = {
        "groups": [
            {"content_hash": fit_hash, "split": "fit"},
            {"content_hash": test_hash, "split": "test"},
        ]
    }
    feature_manifest = {"feature_order": list(selector.PRIMARY_FEATURES)}
    layout = selector.SourceLayout(
        artifact_root=tmp_path,
        pairs=(selector.PairSpec("pair", "Synthetic pair", True),),
        seeds=(42,),
        tasks=(selector.TaskSpec("task", 2),),
        receiver_paths={"task": receiver_path},
        b6_paths={("pair", 42, "task"): b6_path},
        source_artifacts=(),
        dataset_content_sha256="synthetic",
    )
    return layout, feature_manifest, split_manifest


def test_stage_test_without_authorization_is_rejected_before_source_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    layout, feature_manifest, split_manifest = _synthetic_layout(tmp_path)
    source_touched = False

    def fail_if_source_is_loaded(_protocol: object) -> selector.SourceLayout:
        nonlocal source_touched
        source_touched = True
        raise AssertionError("test source must remain unread without authorization")

    monkeypatch.setattr(selector, "_load_source_layout", fail_if_source_is_loaded)

    with pytest.raises(PermissionError, match="committed, remotely consumed attempt"):
        selector.load_observations(
            {},
            feature_manifest,
            split_manifest,
            stage="test",
            test_authorization=None,
        )

    assert source_touched is False
    assert layout.receiver_paths["task"].exists()


def test_develop_skips_test_row_before_invalid_feature_and_outcome_parse(
    tmp_path: Path,
) -> None:
    layout, feature_manifest, split_manifest = _synthetic_layout(tmp_path)

    observations, audit, returned_layout = selector.load_observations(
        {},
        feature_manifest,
        split_manifest,
        stage="develop",
        source_layout=layout,
    )

    assert returned_layout is layout
    assert len(observations) == 1
    assert observations[0].split == "fit"
    assert observations[0].question_id == "fit"
    assert audit["receiver_outcome_rows_parsed_by_split"]["test"] == 0
    assert audit["fused_outcome_rows_parsed_by_split"]["test"] == 0
    assert audit["rows_skipped_before_outcome_parse_by_split"]["test"] == 2


def test_write_once_helpers_never_clobber_existing_files(tmp_path: Path) -> None:
    json_path = tmp_path / "frozen.json"
    selector._write_json_once(json_path, {"version": 1})
    frozen_json = json_path.read_bytes()

    with pytest.raises(FileExistsError):
        selector._write_json_once(json_path, {"version": 2})
    assert json_path.read_bytes() == frozen_json

    csv_path = tmp_path / "frozen.csv"
    selector._write_csv_once(csv_path, [{"value": "first"}])
    frozen_csv = csv_path.read_bytes()

    with pytest.raises(FileExistsError):
        selector._write_csv_once(csv_path, [{"value": "second"}])
    assert csv_path.read_bytes() == frozen_csv


def test_single_parent_exact_diff_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    commit = "c" * 40
    parent = "p" * 40
    expected_paths = [
        selector.REPO_ROOT / "recipe/eval_recipe/phase2a_1/selection_lock.json",
        selector.REPO_ROOT / "recipe/eval_recipe/phase2a_1/locked_models/global.joblib",
    ]
    changed = "\n".join(
        str(path.relative_to(selector.REPO_ROOT)) for path in expected_paths
    )

    def exact_git(*args: str, **_kwargs: object) -> str:
        if args[:4] == ("rev-list", "--parents", "-n", "1"):
            return f"{commit} {parent}"
        if args[:3] == ("diff", "--name-only", parent):
            return changed
        raise AssertionError(f"unexpected git invocation: {args}")

    monkeypatch.setattr(selector, "_git", exact_git)
    selector._verify_single_parent_commit_diff(commit, parent, expected_paths)

    def extra_path_git(*args: str, **_kwargs: object) -> str:
        if args[:4] == ("rev-list", "--parents", "-n", "1"):
            return f"{commit} {parent}"
        if args[:3] == ("diff", "--name-only", parent):
            return f"{changed}\nunexpected.txt"
        raise AssertionError(f"unexpected git invocation: {args}")

    monkeypatch.setattr(selector, "_git", extra_path_git)
    with pytest.raises(ValueError, match="changed unexpected paths"):
        selector._verify_single_parent_commit_diff(commit, parent, expected_paths)

    def merge_commit_git(*args: str, **_kwargs: object) -> str:
        if args[:4] == ("rev-list", "--parents", "-n", "1"):
            return f"{commit} {parent} {'q' * 40}"
        raise AssertionError(f"unexpected git invocation: {args}")

    monkeypatch.setattr(selector, "_git", merge_commit_git)
    with pytest.raises(ValueError, match="must have the single parent"):
        selector._verify_single_parent_commit_diff(commit, parent, expected_paths)


def test_existing_remote_consumption_tag_rejects_claim_without_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing_commit = "a" * 40
    subprocess_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    monkeypatch.setattr(
        selector,
        "_remote_tag_target",
        lambda _tag: existing_commit,
    )

    def forbidden_subprocess_run(
        *args: object, **kwargs: object
    ) -> object:  # pragma: no cover - failure path only
        subprocess_calls.append((args, kwargs))
        raise AssertionError("existing remote tag must reject before tag creation or push")

    monkeypatch.setattr(selector.subprocess, "run", forbidden_subprocess_run)

    with pytest.raises(FileExistsError, match="already exists"):
        selector._claim_remote_test_consumption(
            "phase2a1-consumed",
            "b" * 40,
        )

    assert subprocess_calls == []
