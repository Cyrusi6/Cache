from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from script.analysis import fpct_e1_instrumentation_gate as gate
from script.experiment import fpct_e1_capture_runner as runner


def _parity() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "historical-test-payload",
        "status": "GO",
        "teacher_forced": {
            "logits_bitwise_equal": True,
            "loss_bitwise_equal": True,
            "cache_bitwise_equal": True,
        },
        "greedy_decode": {
            "steps": 3,
            "capture_forward_count": 3,
            "logits_bitwise_equal_each_step": True,
            "cache_bitwise_equal_each_step": True,
            "tokens_equal": True,
        },
        "capture_contract": {"stores_raw_kv": False},
    }


def _synthetic() -> dict:
    return {
        "schema_version": 1,
        "protocol_id": "historical-test-payload",
        "status": "GO",
        "query_changing": {
            "gamma_query_variance": 0.25,
            "posterior_top1_any_change": True,
            "forward_count": 2,
            "parent_query_count": 2,
        },
        "identical_candidates": {
            "gamma_kl_prior": 0.0,
            "gamma_tv_prior": 0.0,
            "jensen_gap": 0.0,
        },
        "stores_raw_kv": False,
    }


def _fixture_repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    (repo / "source").mkdir(parents=True)
    (repo / "test").mkdir(parents=True)
    (repo / "source/instrumentation.py").write_text("source-v1\n")
    (repo / "test/test_instrumentation.py").write_text("test-v1\n")
    monkeypatch.setattr(gate, "A4_SOURCE_FILES", ("source/instrumentation.py",))
    monkeypatch.setattr(gate, "A4_TEST_FILES", ("test/test_instrumentation.py",))
    monkeypatch.setattr(gate, "parity_evidence", _parity)
    monkeypatch.setattr(gate, "synthetic_evidence", _synthetic)
    recipe = repo / "recipe/eval_recipe/fpct_e1"
    recipe.mkdir(parents=True)
    return repo, recipe, repo / "source/instrumentation.py"


def _test_result() -> dict:
    return {
        "runner": "fixture",
        "test_files": list(gate.A4_TEST_FILES),
        "passed": 7,
        "exit_code": 0,
        "cuda_visible_devices": "",
        "offline_environment": True,
        "status": "GO",
    }


def _outputs(recipe: Path) -> tuple[Path, Path, Path]:
    return (
        recipe / "e1_instrumentation_parity_a4.json",
        recipe / "e1_synthetic_query_variance_a4.json",
        recipe / "e1_instrumentation_hard_gate_a4.json",
    )


def test_a4_successor_is_complete_immutable_and_accepted_by_execution_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, recipe, source = _fixture_repo(tmp_path, monkeypatch)
    historical = {
        path: f"historical:{path.name}\n".encode()
        for path in (
            recipe / "e1_instrumentation_parity.json",
            recipe / "e1_synthetic_query_variance.json",
            recipe / "e1_instrumentation_hard_gate.json",
        )
    }
    for path, content in historical.items():
        path.write_bytes(content)

    parity, synthetic, hard_gate = _outputs(recipe)
    result = gate.build_a4_successor_attestation(
        repo_root=repo,
        parity_output=parity,
        synthetic_output=synthetic,
        hard_gate_output=hard_gate,
        test_runner=lambda _root: _test_result(),
    )

    assert result["gate_id"] == runner.CONSOLIDATED_GATE_ID
    assert result["attestation_id"] == gate.A4_ATTESTATION_ID
    assert result["prospective_a4_successor"] is True
    assert result["status"] == "GO"
    assert set(result["checks"]) == set(runner.REQUIRED_GATE_CHECKS)
    assert all(result["checks"].values())
    assert [record["kind"] for record in result["evidence"]] == [
        "generated_evidence",
        "generated_evidence",
        "source",
        "test",
    ]
    assert result["evidence_closure"]["all_declared_files_hashed"] is True
    assert result["evidence_closure"]["source_test_stable_during_attestation"] is True
    assert json.loads(parity.read_text())["protocol_id"] == gate.A4_PARITY_PROTOCOL_ID
    assert json.loads(synthetic.read_text())["protocol_id"] == gate.A4_SYNTHETIC_PROTOCOL_ID
    assert all(path.read_bytes() == content for path, content in historical.items())

    def resolve(logical_path: str) -> Path:
        root, relative = runner._parse_logical(logical_path)
        assert root == "repo"
        return repo.joinpath(*relative.parts)

    verified = runner.verify_instrumentation_gate(
        hard_gate,
        hashlib.sha256(hard_gate.read_bytes()).hexdigest(),
        evidence_resolver=resolve,
    )
    assert verified["status"] == "GO"
    assert verified["checks"] == {name: True for name in runner.REQUIRED_GATE_CHECKS}

    source.write_text("tampered-after-attestation\n")
    with pytest.raises(RuntimeError, match="evidence SHA mismatch"):
        runner.verify_instrumentation_gate(hard_gate, evidence_resolver=resolve)
    with pytest.raises(FileExistsError, match="immutable and already exist"):
        gate.build_a4_successor_attestation(
            repo_root=repo,
            parity_output=parity,
            synthetic_output=synthetic,
            hard_gate_output=hard_gate,
            test_runner=lambda _root: _test_result(),
        )


def test_a4_builder_refuses_historical_v1_output_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, recipe, _source = _fixture_repo(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="historical v1.*immutable"):
        gate.build_a4_successor_attestation(
            repo_root=repo,
            parity_output=recipe / "e1_instrumentation_parity.json",
            synthetic_output=recipe / "e1_synthetic_query_variance_a4.json",
            hard_gate_output=recipe / "e1_instrumentation_hard_gate_a4.json",
            test_runner=lambda _root: _test_result(),
            canonical_paths_required=False,
        )


def test_concurrent_source_change_yields_blocked_not_go_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, recipe, source = _fixture_repo(tmp_path, monkeypatch)

    def changing_test_runner(_root: Path) -> dict:
        source.write_text("changed-during-attestation\n")
        return _test_result()

    with pytest.raises(RuntimeError, match="failed before publication"):
        gate.build_a4_successor_attestation(
            repo_root=repo,
            parity_output=_outputs(recipe)[0],
            synthetic_output=_outputs(recipe)[1],
            hard_gate_output=_outputs(recipe)[2],
            test_runner=changing_test_runner,
        )
    assert not any(path.exists() for path in _outputs(recipe))


def test_failed_cpu_suite_publishes_no_partial_successor_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, recipe, _source = _fixture_repo(tmp_path, monkeypatch)
    failed = _test_result()
    failed.update({"passed": 0, "exit_code": 1, "status": "BLOCKED"})
    with pytest.raises(RuntimeError, match="formula_oracles"):
        gate.build_a4_successor_attestation(
            repo_root=repo,
            parity_output=_outputs(recipe)[0],
            synthetic_output=_outputs(recipe)[1],
            hard_gate_output=_outputs(recipe)[2],
            test_runner=lambda _root: failed,
        )
    assert not any(path.exists() for path in _outputs(recipe))


def test_publication_failure_rolls_back_newly_linked_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, recipe, _source = _fixture_repo(tmp_path, monkeypatch)
    original_link = gate.os.link
    call_count = 0

    def fail_third_link(source: str, destination: Path) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 3:
            raise OSError("injected third-link failure")
        original_link(source, destination)

    monkeypatch.setattr(gate.os, "link", fail_third_link)
    with pytest.raises(OSError, match="third-link failure"):
        gate.build_a4_successor_attestation(
            repo_root=repo,
            parity_output=_outputs(recipe)[0],
            synthetic_output=_outputs(recipe)[1],
            hard_gate_output=_outputs(recipe)[2],
            test_runner=lambda _root: _test_result(),
        )
    assert not any(path.exists() for path in _outputs(recipe))
