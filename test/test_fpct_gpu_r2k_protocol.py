import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipe/eval_recipe/fpct_gpu_r2k"


def test_r2k_protocol_manifests_are_consistent():
    manifest = json.loads((RECIPE / "latency_audit_manifest.json").read_text())
    boundary = json.loads((RECIPE / "implementation_boundary.json").read_text())
    decision = json.loads((RECIPE / "decision_tree.json").read_text())
    diff = json.loads((RECIPE / "protocol_diff.json").read_text())
    assert manifest["immutable_predecessor"]["terminal"] == "GPU_ENGINEERING_BLOCKED_R2"
    assert manifest["immutable_predecessor"]["retry_or_reinterpret"] is False
    assert manifest["phases"]["diagnostic_only"]["may_produce_go"] is False
    assert manifest["phases"]["diagnostic_only"]["warmups_per_arm"] == 20
    assert manifest["phases"]["diagnostic_only"]["measured_forwards_per_arm"] == 50
    assert len(manifest["phases"]["diagnostic_only"]["block_order"]) == 8
    assert manifest["phases"]["immutable_confirmatory_gate"]["single_complete_execution"] is True
    assert manifest["phases"]["immutable_confirmatory_gate"]["required_original_checks"] == 23
    assert manifest["kernel_contract"]["prior_application_count"] == 1
    assert manifest["kernel_contract"]["group_value_5d_allocation_forbidden"] is True
    assert "math.md" in boundary["forbidden"]["paths"]
    assert "rosetta/model/aligner.py" in boundary["forbidden"]["paths"]
    assert decision["r2j_terminal_immutable"] is True
    assert diff["certifier_sidecar_projection_must_remain_identical"] is True
    assert not manifest["accuracy_or_correctness_accessed"]


def test_r2k_balanced_order_has_four_orders_each():
    manifest = json.loads((RECIPE / "latency_audit_manifest.json").read_text())
    order = manifest["phases"]["diagnostic_only"]["block_order"]
    assert order.count("c_post_then_f") == 4
    assert order.count("f_then_c_post") == 4
    assert order[:4] == [
        "c_post_then_f",
        "f_then_c_post",
        "f_then_c_post",
        "c_post_then_f",
    ]
