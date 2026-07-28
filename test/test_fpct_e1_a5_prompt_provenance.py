from __future__ import annotations

import copy
import hashlib
import shutil
from pathlib import Path

import pytest

from script.experiment.fpct_e1_a5_prompt_provenance import (
    EXTRA_CHOICES_ONLY,
    HISTORICAL_EXACT,
    TASK_ORDER,
    attest_e0_renderer_identity,
    build_census_manifest,
    canonical_json_bytes,
    classify_prompt_relation,
    dual_anchor_record,
    historical_projected_example,
    raw_full_row_sha256,
    summarize_dual_anchor_census,
)
from script.experiment.fpct_e1_runtime_backend import canonical_content_sha256
from script.experiment import fpct_e1_prepare_input_lock as prepare


REPO_ROOT = Path(__file__).resolve().parents[1]


def _arc(choices: list[str], answer: str = "A") -> dict:
    return {
        "id": "row-1",
        "question": "Which answer?",
        "choices": {
            "label": [chr(65 + index) for index in range(len(choices))],
            "text": choices,
        },
        "answerKey": answer,
    }


def _record(example: dict, *, answer: str = "A") -> dict:
    choices = list(example["choices"]["text"])
    group = canonical_content_sha256(example["question"], choices[:4])
    historical_prompt = "Q\n" + "\n".join(choices[:4])
    production_prompt = "Q\n" + "\n".join(choices)
    return dual_anchor_record(
        task="ai2-arc",
        content_group_sha256=group,
        sample_key_sha256="2" * 64,
        source_row_id="1",
        example=example,
        gold_answer=answer,
        historical_rendered_prompt=historical_prompt,
        historical_alignment_sha256="3" * 64,
        production_rendered_prompt=production_prompt,
        production_alignment_sha256=(
            "3" * 64 if len(choices) == 4 else "4" * 64
        ),
        historical_prompt_token_count=10,
        production_prompt_token_count=10 + len(choices) - 4,
        production_certified_parent_count=2,
        production_logical_row_count=896,
        production_physical_chunk_count=1,
        historical_content_sha256=group,
    )


def test_four_choice_row_is_exact_dual_anchor() -> None:
    value = _record(_arc(["a", "b", "c", "d"]))
    assert value["prompt_relation"] == HISTORICAL_EXACT
    assert value["choice_difference_only"] is False
    assert value["historical_choice_count"] == value["production_choice_count"] == 4


def test_five_choice_gold_a_is_extra_choice_only_and_keeps_historical_anchor() -> None:
    value = _record(_arc(["a", "b", "c", "d", "extra"]), answer="A")
    assert value["prompt_relation"] == EXTRA_CHOICES_ONLY
    assert value["choice_difference_only"] is True
    assert value["production_choice_count"] == 5
    assert value["raw_choice_labels"] == list("ABCDE")


def test_five_choice_gold_e_fails_closed() -> None:
    with pytest.raises(ValueError, match="outside A-D"):
        _record(_arc(["a", "b", "c", "d", "extra"], answer="E"), answer="E")


def test_first_four_order_change_fails_closed() -> None:
    with pytest.raises(ValueError, match="first-four choice text/order changed"):
        classify_prompt_relation(
            historical_question="q",
            historical_choices=["a", "b", "c", "d"],
            historical_labels=list("ABCD"),
            production_question="q",
            production_choices=["b", "a", "c", "d", "e"],
            production_labels=list("ABCDE"),
            gold_answer="A",
            historical_rendered_prompt="h",
            production_rendered_prompt="p",
            historical_alignment_sha256="1" * 64,
            production_alignment_sha256="2" * 64,
        )


def test_suffix_text_change_keeps_first4_but_changes_full_runtime_anchor() -> None:
    first = _arc(["a", "b", "c", "d", "extra-one"])
    second = _arc(["a", "b", "c", "d", "extra-two"])
    first_record, second_record = _record(first), _record(second)
    assert (
        first_record["historical_first4_question_choices_sha256"]
        == second_record["historical_first4_question_choices_sha256"]
    )
    assert raw_full_row_sha256(first) != raw_full_row_sha256(second)
    assert (
        first_record["production_rendered_prompt_sha256"]
        != second_record["production_rendered_prompt_sha256"]
    )


def test_historical_content_anchor_tamper_fails_closed() -> None:
    example = _arc(["a", "b", "c", "d", "e"])
    group = canonical_content_sha256(example["question"], example["choices"]["text"][:4])
    tampered = copy.deepcopy(example)
    tampered["choices"]["text"][0] = "tampered"
    with pytest.raises(ValueError, match="content-group anchor changed"):
        dual_anchor_record(
            task="ai2-arc",
            content_group_sha256=group,
            sample_key_sha256="2" * 64,
            source_row_id="1",
            example=tampered,
            gold_answer="A",
            historical_rendered_prompt="h",
            historical_alignment_sha256="3" * 64,
            production_rendered_prompt="p",
            production_alignment_sha256="4" * 64,
            historical_prompt_token_count=1,
            production_prompt_token_count=2,
            production_certified_parent_count=1,
            production_logical_row_count=1,
            production_physical_chunk_count=1,
            # Both caller-supplied hashes remain the old frozen value.  The
            # provenance helper must independently hash the tampered example.
            historical_content_sha256=group,
        )


def test_projected_example_does_not_mutate_full_row() -> None:
    example = _arc(["a", "b", "c", "d", "e"])
    original = copy.deepcopy(example)
    projected = historical_projected_example("ai2-arc", example)
    assert example == original
    assert projected["choices"]["text"] == ["a", "b", "c", "d"]


def test_renderer_source_identity_and_tamper_fail_closed(tmp_path: Path) -> None:
    identity = attest_e0_renderer_identity(REPO_ROOT)
    assert identity["renderer_source_identity_attested"] is True
    assert identity["production_renderer_exactly_attested"] is False
    relatives = (
        "script/evaluation/unified_evaluator.py",
        "rosetta/utils/evaluate.py",
        "rosetta/model/aligner.py",
        "rosetta/train/dataset_adapters.py",
        "rosetta/utils/model_loading.py",
        "script/experiment/fpct_e0_runner.py",
    )
    for relative in relatives:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / relative, destination)
    evaluator = tmp_path / "script/evaluation/unified_evaluator.py"
    evaluator.write_bytes(evaluator.read_bytes() + b"\n# tampered\n")
    with pytest.raises(ValueError, match="UnifiedEvaluator differs"):
        attest_e0_renderer_identity(tmp_path)


def test_tokenizer_asset_walker_never_opens_or_records_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "tokenizer"
    root.mkdir()
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (root / name).write_text(f"{name}\n")
    weight = root / "model.safetensors"
    weight.write_bytes(b"must-never-be-read")
    observed: list[str] = []
    original = prepare.sha256_file

    def guarded(path: Path) -> str:
        observed.append(Path(path).name)
        if Path(path).suffix in {".safetensors", ".bin", ".pt", ".pth"}:
            raise AssertionError("A5 tokenizer walker opened a weight")
        return original(path)

    monkeypatch.setattr(prepare, "sha256_file", guarded)
    record = prepare.tokenizer_runtime_asset_tree(root)
    assert record["weight_or_checkpoint_file_opened"] is False
    assert "model.safetensors" not in observed
    assert all(
        not row["relative_path"].endswith(".safetensors")
        for row in record["files"]
    )


def test_tokenizer_asset_walker_accepts_hf_snapshot_file_symlinks(
    tmp_path: Path,
) -> None:
    blobs = tmp_path / "blobs"
    snapshot = tmp_path / "snapshots" / "frozen-revision"
    blobs.mkdir()
    snapshot.mkdir(parents=True)
    for index, name in enumerate(
        ("config.json", "tokenizer.json", "tokenizer_config.json")
    ):
        blob = blobs / f"blob-{index}"
        blob.write_text(f"{name}\n", encoding="utf-8")
        (snapshot / name).symlink_to(blob)
    (snapshot / "model.safetensors").write_bytes(b"must-never-be-read")

    record = prepare.tokenizer_runtime_asset_tree(snapshot)

    assert record["weight_or_checkpoint_file_opened"] is False
    assert {row["kind"] for row in record["files"]} == {"symlink_file"}
    assert {row["relative_path"] for row in record["files"]} == {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }


def _synthetic_census_record(task: str, index: int) -> dict:
    relation = EXTRA_CHOICES_ONLY if index % 3 == 0 else HISTORICAL_EXACT
    production_count = 5 if relation == EXTRA_CHOICES_ONLY else 4
    digest = hashlib.sha256(f"{task}:{index}".encode()).hexdigest()
    sample = hashlib.sha256(f"sample:{task}:{index}".encode()).hexdigest()
    return {
        "schema_version": 7,
        "protocol_id": "fpct_e1_mechanism_audit_v7_actual_e0_runtime_prompt",
        "artifact_type": "a5_prompt_census_record",
        "task": task,
        "content_group_sha256": digest,
        "sample_key_sha256": sample,
        "source_row_id": str(index),
        "historical_choice_count": 4,
        "production_choice_count": production_count,
        "raw_choice_labels": [chr(65 + value) for value in range(production_count)],
        "gold_answer": "A",
        "historical_first4_question_choices_sha256": "1" * 64,
        "historical_rendered_prompt_sha256": "2" * 64,
        "historical_alignment_sha256": "3" * 64,
        "raw_full_row_sha256": "4" * 64,
        "production_rendered_prompt_sha256": "5" * 64,
        "production_alignment_sha256": "6" * 64,
        "historical_prompt_token_count": 10,
        "production_prompt_token_count": 11,
        "production_certified_parent_count": 2,
        "production_logical_row_count": 896,
        "production_physical_chunk_count": 1,
        "prompt_relation": relation,
        "choice_difference_only": relation == EXTRA_CHOICES_ONLY,
    }


def test_full_326_population_summary_and_partition_semantics() -> None:
    counts = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
    records = []
    for task in TASK_ORDER:
        records.extend(
            sorted(
                (_synthetic_census_record(task, index) for index in range(counts[task])),
                key=lambda row: (
                    row["content_group_sha256"], row["sample_key_sha256"]
                ),
            )
        )
    summary = summarize_dual_anchor_census(records)
    assert summary["group_count"] == 326
    assert summary["task_counts"] == counts
    first_partition = records[:137] + records[137:]
    second_partition = records[:17] + records[17:251] + records[251:]
    assert hashlib.sha256(canonical_json_bytes(first_partition)).hexdigest() == hashlib.sha256(
        canonical_json_bytes(second_partition)
    ).hexdigest()
    artifact_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in records)
    manifest = build_census_manifest(
        records,
        execution_sha="a" * 40,
        run_uid="fpct-e1-a5-runtime-prompt-aaaaaaaa-v1",
        record_artifact={
            "relative_path": "a5_prompt_census_records.jsonl",
            "sha256": hashlib.sha256(artifact_payload).hexdigest(),
            "bytes": len(artifact_payload),
            "row_count": len(records),
        },
    )
    assert manifest["population_count"] == 326
    assert manifest["missing_row_count"] == manifest["duplicate_row_count"] == 0
    lexical_task_order = sorted(
        records,
        key=lambda row: (
            row["task"], row["content_group_sha256"], row["sample_key_sha256"]
        ),
    )
    with pytest.raises(ValueError, match="canonical task/group/sample order"):
        summarize_dual_anchor_census(lexical_task_order)
