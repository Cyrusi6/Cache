from __future__ import annotations

import ast
from pathlib import Path

from script.analysis.fpct_e1_instrumentation_gate import (
    a5_pure_tensor_instrumentation_evidence,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_a5_pure_tensor_accumulator_detects_query_change_without_model() -> None:
    evidence = a5_pure_tensor_instrumentation_evidence()
    assert evidence["status"] == "GO"
    assert evidence["query_changing"]["gamma_query_variance"] > 0
    assert evidence["query_changing"]["posterior_top1_any_change"] is True
    assert evidence["model_instantiated"] is False
    assert evidence["model_forward_run"] is False
    assert evidence["pretrained_weights_or_checkpoint_loaded"] is False


def test_a5_builder_does_not_call_model_based_evidence_functions() -> None:
    path = REPO_ROOT / "script/analysis/fpct_e1_instrumentation_gate.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "build_a5_successor_attestation"
    )
    called = {
        node.func.id
        for node in ast.walk(function)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "parity_evidence" not in called
    assert "synthetic_evidence" not in called
    assert "Qwen3ForCausalLM" not in called
    assert "RosettaModel" not in called
