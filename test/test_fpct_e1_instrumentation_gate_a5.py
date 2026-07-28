from __future__ import annotations

import json
from pathlib import Path

import pytest

from script.analysis import fpct_e1_instrumentation_gate as gate


def _parity() -> dict:
    return {
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
    }


def _synthetic() -> dict:
    return {
        "status": "GO",
        "query_changing": {
            "gamma_query_variance": 0.25,
            "posterior_top1_any_change": True,
            "forward_count": 2,
            "parent_query_count": 2,
        },
    }


def _result() -> dict:
    return {
        "runner": "pytest_no_model_a5",
        "passed": 3,
        "exit_code": 0,
        "output_sha256": "a" * 64,
        "status": "GO",
        "cuda_visible_devices": "",
        "offline_environment": True,
        "test_files": list(gate.A5_INSTRUMENTATION_TEST_FILES),
        "model_instantiated": False,
        "model_forward_run": False,
    }


def _build_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, tuple[Path, Path, Path], Path, Path]:
    repo = tmp_path / "repo"
    source = repo / "source.py"
    test = repo / "test.py"
    source.parent.mkdir(parents=True)
    source.write_text("source\n")
    test.write_text("test\n")
    recipe = repo / "recipe/eval_recipe/fpct_e1"
    recipe.mkdir(parents=True)
    monkeypatch.setattr(gate, "A4_SOURCE_FILES", ("source.py",))
    monkeypatch.setattr(gate, "A5_INSTRUMENTATION_TEST_FILES", ("test.py",))
    historical = {
        recipe / "e1_instrumentation_parity_a4.json": b"a4-parity\n",
        recipe / "e1_synthetic_query_variance_a4.json": b"a4-synthetic\n",
        recipe / "e1_instrumentation_hard_gate_a4.json": b"a4-gate\n",
    }
    for path, payload in historical.items():
        path.write_bytes(payload)
    monkeypatch.setattr(
        gate,
        "A4_PARITY_SHA256",
        gate.sha256_file(recipe / "e1_instrumentation_parity_a4.json"),
    )
    monkeypatch.setattr(
        gate,
        "verify_historical_a4_runtime_identity",
        lambda _root: {"source.py": gate.sha256_file(source)},
    )
    outputs = (
        recipe / "e1_instrumentation_parity_a5.json",
        recipe / "e1_synthetic_query_variance_a5.json",
        recipe / "e1_instrumentation_hard_gate_a5.json",
    )
    gate.build_a5_successor_attestation(
        repo_root=repo,
        parity_output=outputs[0],
        synthetic_output=outputs[1],
        hard_gate_output=outputs[2],
        test_runner=lambda _root: {
            **_result(),
            "test_files": ["test.py"],
        },
    )
    return repo, outputs, source, test


def test_a5_attestation_is_fresh_and_does_not_overwrite_a4(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    source = repo / "source.py"
    test = repo / "test.py"
    source.parent.mkdir(parents=True)
    source.write_text("source\n")
    test.write_text("test\n")
    recipe = repo / "recipe/eval_recipe/fpct_e1"
    recipe.mkdir(parents=True)
    monkeypatch.setattr(gate, "A4_SOURCE_FILES", ("source.py",))
    monkeypatch.setattr(gate, "A5_INSTRUMENTATION_TEST_FILES", ("test.py",))
    monkeypatch.setattr(
        gate,
        "parity_evidence",
        lambda: (_ for _ in ()).throw(AssertionError("A5 instantiated model parity")),
    )
    monkeypatch.setattr(
        gate,
        "synthetic_evidence",
        lambda: (_ for _ in ()).throw(AssertionError("A5 used model synthetic")),
    )
    historical = {
        recipe / "e1_instrumentation_parity_a4.json": b"a4-parity\n",
        recipe / "e1_synthetic_query_variance_a4.json": b"a4-synthetic\n",
        recipe / "e1_instrumentation_hard_gate_a4.json": b"a4-gate\n",
    }
    for path, payload in historical.items():
        path.write_bytes(payload)
    monkeypatch.setattr(
        gate,
        "A4_PARITY_SHA256",
        gate.sha256_file(recipe / "e1_instrumentation_parity_a4.json"),
    )
    monkeypatch.setattr(
        gate,
        "verify_historical_a4_runtime_identity",
        lambda _root: {"source.py": gate.sha256_file(source)},
    )
    outputs = (
        recipe / "e1_instrumentation_parity_a5.json",
        recipe / "e1_synthetic_query_variance_a5.json",
        recipe / "e1_instrumentation_hard_gate_a5.json",
    )
    result = gate.build_a5_successor_attestation(
        repo_root=repo,
        parity_output=outputs[0],
        synthetic_output=outputs[1],
        hard_gate_output=outputs[2],
        test_runner=lambda _root: _result(),
    )
    assert result["attestation_id"] == gate.A5_ATTESTATION_ID
    assert result["decision"] == "GO_PROSPECTIVE_A5_REATTESTATION"
    assert result["prospective_a5_successor"] is True
    assert json.loads(outputs[0].read_text())["protocol_id"] == gate.A5_PARITY_PROTOCOL_ID
    assert json.loads(outputs[1].read_text())["protocol_id"] == gate.A5_SYNTHETIC_PROTOCOL_ID
    assert all(path.read_bytes() == payload for path, payload in historical.items())
    with pytest.raises(FileExistsError, match="A5 successor artifacts"):
        gate.build_a5_successor_attestation(
            repo_root=repo,
            parity_output=outputs[0],
            synthetic_output=outputs[1],
            hard_gate_output=outputs[2],
            test_runner=lambda _root: _result(),
        )


def test_a5_builder_rejects_historical_a4_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    recipe = repo / "recipe/eval_recipe/fpct_e1"
    recipe.mkdir(parents=True)
    monkeypatch.setattr(gate, "A4_SOURCE_FILES", ())
    monkeypatch.setattr(gate, "A5_INSTRUMENTATION_TEST_FILES", ())
    with pytest.raises(ValueError, match="v1/A4.*immutable"):
        gate.build_a5_successor_attestation(
            repo_root=repo,
            parity_output=recipe / "e1_instrumentation_parity_a4.json",
            synthetic_output=recipe / "e1_synthetic_query_variance_a5.json",
            hard_gate_output=recipe / "e1_instrumentation_hard_gate_a5.json",
            test_runner=lambda _root: _result(),
            canonical_paths_required=False,
        )


@pytest.mark.parametrize(
    "corruption", ["synthetic", "hard_gate_firewall", "source", "duplicate_key"]
)
def test_a5_verifier_recomputes_all_evidence_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    repo, outputs, source, _test = _build_fixture(tmp_path, monkeypatch)
    if corruption == "synthetic":
        value = json.loads(outputs[1].read_text())
        value["model_instantiated"] = True
        outputs[1].write_bytes(gate.canonical_bytes(value))
    elif corruption == "hard_gate_firewall":
        value = json.loads(outputs[2].read_text())
        value["firewall"]["natural_data_accessed"] = True
        outputs[2].write_bytes(gate.canonical_bytes(value))
    elif corruption == "source":
        source.write_text("changed\n")
    elif corruption == "duplicate_key":
        outputs[0].write_text('{"schema_version":3,"schema_version":3}\n')
    else:  # pragma: no cover - parametrization is closed above
        raise AssertionError(corruption)
    with pytest.raises(ValueError):
        gate.verify_a5_successor_attestation(
            repo_root=repo,
            parity_output=outputs[0],
            synthetic_output=outputs[1],
            hard_gate_output=outputs[2],
        )


def test_a5_transaction_rolls_back_all_new_files_on_final_validation_failure(
    tmp_path: Path,
) -> None:
    outputs = tuple(
        (tmp_path / name, {"name": name})
        for name in ("parity.json", "synthetic.json", "gate.json")
    )
    with pytest.raises(RuntimeError, match="closure changed"):
        gate.publish_json_transaction_no_overwrite(
            outputs,
            final_validator=lambda: False,
        )
    assert not any(path.exists() for path, _value in outputs)
    assert not list(tmp_path.glob(".*.json.*"))


def test_a5_transaction_never_touches_preexisting_path(tmp_path: Path) -> None:
    existing = tmp_path / "parity.json"
    existing.write_bytes(b"immutable\n")
    outputs = (
        (existing, {"new": True}),
        (tmp_path / "synthetic.json", {"new": True}),
        (tmp_path / "gate.json", {"new": True}),
    )
    with pytest.raises(FileExistsError, match="already exist"):
        gate.publish_json_transaction_no_overwrite(outputs)
    assert existing.read_bytes() == b"immutable\n"
    assert not outputs[1][0].exists()
    assert not outputs[2][0].exists()
