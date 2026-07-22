from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from script.analysis import fpct_cfm_harness_h1_audit as audit


ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "recipe/eval_recipe/fpct_cfm_harness_h1"


def test_strict_json_rejects_duplicate_and_nonfinite() -> None:
    with pytest.raises(audit.AuditError, match="duplicate"):
        audit.strict_loads('{"x":1,"x":2}')
    with pytest.raises(audit.AuditError, match="nonfinite"):
        audit.strict_loads('{"x":NaN}')


def test_source_discovery_finds_real_load_lock_contract() -> None:
    found = audit.discover_function(
        ROOT / "script/experiment/fpct_confirmatory_runner.py",
        "load_lock",
        {"payload"},
    )
    assert found == ["/image", "/manifest_sha256", "/run_uid", "/scientific_code_commit"]


def test_source_discovery_covers_alias_get_membership_subset_and_json_pointer(tmp_path: Path) -> None:
    source = tmp_path / "consumer.py"
    source.write_text(
        "def consume(payload):\n"
        "    required = {'run_uid', 'image'}\n"
        "    required.issubset(payload)\n"
        "    nested = payload.get('assets', {})\n"
        "    sidecar = nested['sidecar']\n"
        "    assert 'status' in payload\n"
        "    json_pointer(payload, '/resource_geometry/tinyllama_all_splits')\n"
        "    return sidecar\n",
        encoding="utf-8",
    )
    found = audit.discover_function(source, "consume", {"payload"})
    assert found == [
        "/assets",
        "/assets/sidecar",
        "/image",
        "/resource_geometry/tinyllama_all_splits",
        "/run_uid",
        "/status",
    ]


def test_historical_red_fixtures_are_preserved() -> None:
    r2m = audit.strict_load(ROOT / "recipe/eval_recipe/fpct_gpu_r2m/immutable_v1_run_lock.json")
    r2l = audit.strict_load(ROOT / "recipe/eval_recipe/fpct_gpu_r2l/immutable_v1_run_lock.json")
    r2j = audit.strict_load(ROOT / "recipe/eval_recipe/fpct_gpu_r2/fpct_gpu_r2j_run_lock.json")
    assert "manifest_sha256" not in r2m
    assert "resource_geometry" not in r2l
    assert r2j["manifest_sha256"] == audit.MANIFEST_SHA
    assert audit.sha256_file(ROOT / "recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json") == audit.MANIFEST_SHA


def test_compiler_is_deterministic_and_has_no_execution_authority(tmp_path: Path) -> None:
    one = audit.compile_outputs(ROOT, tmp_path / "one")
    two = audit.compile_outputs(ROOT, tmp_path / "two")
    assert one["candidate_lock_sha256"] == two["candidate_lock_sha256"]
    left = audit.strict_load(Path(one["candidate_lock_path"]))
    right = audit.strict_load(Path(two["candidate_lock_path"]))
    assert left == right
    assert left["scientific_output"] is False
    assert left["training_authorized"] is False
    assert left["manifest_sha256"] == left["nested_confirmatory_manifest_sha256"] == audit.MANIFEST_SHA
    configmap = audit.strict_load(Path(one["configmap_path"]))
    assert configmap["immutable"] is True
    assert configmap["data"]["candidate_lock_projection.json"] == Path(one["candidate_lock_path"]).read_text()
    assert configmap["metadata"]["annotations"] == {
        "fpct.cache/scientific-output": "false",
        "fpct.cache/training-authorized": "false",
    }


def test_strict_schema_rejects_bool_as_integer_and_extra_field() -> None:
    candidate = audit.compile_candidate(ROOT)
    schema = audit.strict_load(CFG / "schemas/execution_lock.schema.json")
    broken = copy.deepcopy(candidate)
    broken["training"]["world_size"] = True
    with pytest.raises(audit.AuditError, match="integer"):
        audit.validate_schema(broken, schema)
    broken = copy.deepcopy(candidate)
    broken["extra"] = 1
    with pytest.raises(audit.AuditError, match="extra"):
        audit.validate_schema(broken, schema)


def test_all_candidate_mutations_fail_closed() -> None:
    rows = audit.mutation_coverage(audit.compile_candidate(ROOT), ROOT)
    assert len(rows) >= 100
    assert all(row["fail_closed"] for row in rows), [row for row in rows if not row["fail_closed"]]


def test_arm_order_matches_manifest() -> None:
    source = audit.assigned_constant(ROOT / "script/experiment/fpct_confirmatory_runner.py", "ARM_ORDER")
    manifest = audit.strict_load(ROOT / "recipe/eval_recipe/fpct_confirmatory/confirmatory_manifest.json")
    expected = {int(seed): tuple(arms) for seed, arms in manifest["formal_training"]["arm_order"].items()}
    assert source == expected


def test_stage_graph_is_acyclic_and_complete() -> None:
    result = audit.graph_audit(audit.strict_load(CFG / "stage_graph.json"))
    assert result == {"node_count": 11, "edge_count": 10, "acyclic": True}


def test_all_artifact_schemas_are_top_level_strict() -> None:
    schemas = sorted((CFG / "schemas").glob("*.schema.json"))
    assert len(schemas) == 10
    for path in schemas:
        payload = audit.strict_load(path)
        assert payload["additionalProperties"] is False


def test_current_image_artifacts_lack_required_full_binding() -> None:
    required = {"run_uid", "lock_sha256", "image_digest", "prerequisite_sha256"}
    runner = ROOT / "script/experiment/fpct_confirmatory_runner.py"
    finalizer = ROOT / "script/analysis/fpct_gpu_r2m_finalize.py"
    for path, symbol in ((runner, "matched_smoke"), (runner, "train_triplet"), (finalizer, "finalize")):
        assert required - audit.dict_keys_in_function(path, symbol)


def test_candidate_jobs_have_no_unresolved_placeholders(tmp_path: Path) -> None:
    result = audit.compile_outputs(ROOT, tmp_path)
    for row in result["jobs"].values():
        text = Path(row["path"]).read_text()
        assert not audit.re.search(r"__[A-Z0-9_]+__", text)
        payload = json.loads(text)
        assert payload["metadata"]["labels"]["study"] == "h1-config-only"
        assert payload["metadata"]["annotations"] == {
            "fpct.cache/scientific-output": "false",
            "fpct.cache/training-authorized": "false",
        }


def test_argparse_discovery_covers_real_runner_and_controller() -> None:
    runner = audit.argparse_contract(ROOT / "script/experiment/fpct_confirmatory_runner.py")
    controller = audit.argparse_contract(ROOT / "script/experiment/fpct_gpu_r2_controller.py")
    assert {"matched-smoke", "train-triplet"}.issubset(runner["subcommands"])
    assert {"init", "transition", "record-triplet", "status"}.issubset(controller["subcommands"])
    assert any("--run-lock" in row["flags"] and row["keywords"].get("required") is True for row in runner["arguments"])
    assert any("--root" in row["flags"] and row["keywords"].get("required") is True for row in controller["arguments"])


def test_yaml_contract_records_identity_runtime_and_volumes() -> None:
    contract = audit.yaml_contract(ROOT / "recipe/k8s/fpct_gpu_r2m/matched_smoke_immutable_v1_job.yaml")
    assert contract["metadata"]["labels"]["run_uid"] == "fpct-r2m-80fb295-v1"
    assert contract["image"].endswith(audit.R2M_IMAGE)
    assert contract["resources"]["limits"]["nvidia.com/gpu"] == "2"
    assert contract["env"]["HF_HUB_OFFLINE"] == "1"
    assert any(row["mountPath"] == "/fpct-run" for row in contract["volume_mounts"])
