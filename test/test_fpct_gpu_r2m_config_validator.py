from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from script.analysis import fpct_gpu_r2m_config_validator as validator


ROOT = Path(__file__).resolve().parents[1]
R2K_LOCK = ROOT / "recipe/eval_recipe/fpct_gpu_r2k/immutable_v1_run_lock.json"
R2L_LOCK = ROOT / "recipe/eval_recipe/fpct_gpu_r2l/immutable_v1_run_lock.json"
SCHEMA = ROOT / "recipe/eval_recipe/fpct_gpu_r2m/run_lock.schema.json"
CANONICAL = ROOT / "recipe/eval_recipe/fpct_gpu_r2m/canonical_resource_geometry.json"


def canonical_payload() -> dict:
    return validator.strict_load(CANONICAL)


def candidate_lock() -> dict:
    source = validator.strict_load(R2K_LOCK)
    lock = copy.deepcopy(source)
    lock.update(
        {
            "protocol_id": "fpct_gpu_r2m_immutable_candidate_v1",
            "classification": "IMMUTABLE_CONFIRMATORY_GATE",
            "run_uid": "fpct-r2m-1234abc-v1",
            "scientific_code_commit": "1" * 40,
            "scientific_code_upstream": "1" * 40,
        }
    )
    lock["storage"]["shared_run_root"] = "/netdisk/lijunsi/fpct-confirmatory/fpct-r2m-1234abc-v1"
    lock["config_closure"] = {
        "geometry_projection_sha256": validator.EXPECTED_GEOMETRY_SHA256,
        "schema_sha256": "0" * 64,
        "consumer_pointer_manifest_sha256": "0" * 64,
        "implementation_allowlist_sha256": "0" * 64,
        "consumer_ast_sha256": "0" * 64,
        "canonical_parsed_sha256": "0" * 64,
    }
    return lock


def close_candidate_lock(lock: dict) -> dict:
    consumer = ROOT / "recipe/eval_recipe/fpct_gpu_r2m/consumer_pointer_manifest.json"
    allowlist = ROOT / "recipe/eval_recipe/fpct_gpu_r2m/implementation_allowlist.json"
    lock["config_closure"].update(
        {
            "schema_sha256": validator.sha256_file(SCHEMA),
            "consumer_pointer_manifest_sha256": validator.sha256_file(consumer),
            "implementation_allowlist_sha256": validator.sha256_file(allowlist),
            "consumer_ast_sha256": validator.aggregate_consumer_ast_sha(
                ROOT / "script/experiment/fpct_gpu_r2_runner.py"
            ),
        }
    )
    lock["config_closure"]["canonical_parsed_sha256"] = validator.canonical_parsed_sha(lock)
    return lock


def test_r2k_canonical_geometry_passes() -> None:
    result = validator.validate_geometry(validator.strict_load(R2K_LOCK), canonical_payload())
    assert result["row_count"] == 3
    assert result["task_names"] == ["ai2-arc", "mmlu-redux", "openbookqa"]


def test_r2l_lock_is_rejected_with_explicit_missing_pointer() -> None:
    lock = validator.strict_load(R2L_LOCK)
    schema = validator.strict_load(SCHEMA)
    with pytest.raises(validator.ConfigClosureError, match="/resource_geometry"):
        validator.validate_schema(lock, schema)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda lock: lock.pop("resource_geometry"), "/resource_geometry"),
        (lambda lock: lock["resource_geometry"].pop("tinyllama_all_splits"), "tinyllama_all_splits"),
        (lambda lock: lock["resource_geometry"]["tinyllama_all_splits"].update(tasks={}), "missing pointer|too few properties"),
        (lambda lock: lock["resource_geometry"]["tinyllama_all_splits"]["tasks"].pop("ai2-arc"), "ai2-arc"),
        (lambda lock: lock["resource_geometry"]["tinyllama_all_splits"]["tasks"].update(unknown={"mean": 1.0, "p95": 1.0, "max": 1.0}), "too many properties|additional properties"),
        (lambda lock: lock["resource_geometry"]["tinyllama_all_splits"].update(source_sha256="0" * 64), "const mismatch"),
        (lambda lock: lock["resource_geometry"]["tinyllama_all_splits"]["tasks"]["ai2-arc"].update(mean="bad"), "type mismatch"),
    ],
)
def test_schema_negative_mutations_fail_closed(mutation, match: str) -> None:
    lock = candidate_lock()
    mutation(lock)
    with pytest.raises(validator.ConfigClosureError, match=match):
        validator.validate_schema(lock, validator.strict_load(SCHEMA))


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, None, "1.2"])
def test_nonfinite_null_and_string_geometry_fail(bad) -> None:
    lock = candidate_lock()
    lock["resource_geometry"]["tinyllama_all_splits"]["tasks"]["ai2-arc"]["mean"] = bad
    with pytest.raises(validator.ConfigClosureError):
        validator.validate_schema(lock, validator.strict_load(SCHEMA))


def test_changed_numeric_and_projection_hash_fail() -> None:
    lock = candidate_lock()
    lock["resource_geometry"]["tinyllama_all_splits"]["tasks"]["ai2-arc"]["mean"] += 1e-6
    with pytest.raises(validator.ConfigClosureError, match="projection SHA|differs"):
        validator.validate_geometry(lock, canonical_payload())
    lock = candidate_lock()
    lock["config_closure"]["geometry_projection_sha256"] = "f" * 64
    assert lock["config_closure"]["geometry_projection_sha256"] != validator.EXPECTED_GEOMETRY_SHA256


def test_original_aggregate_geometry_consumer_fixture_is_exact() -> None:
    lock = candidate_lock()
    result = validator.exercise_original_geometry_consumer(
        lock, canonical_payload(), ROOT / "script/experiment/fpct_gpu_r2_runner.py"
    )
    assert result["certified_geometry"] == canonical_payload()["tinyllama_all_splits"]
    assert result["expansion_mean"] is True
    assert result["expansion_p95"] is True


def test_exact_byte_preflight_rejects_configmap_or_mount_mismatch(tmp_path: Path) -> None:
    git_lock = tmp_path / "git.json"
    configmap = tmp_path / "configmap.json"
    mounted = tmp_path / "mounted.json"
    git_lock.write_bytes(b'{"x":1}\n')
    configmap.write_bytes(git_lock.read_bytes())
    mounted.write_bytes(b'{"x":2}\n')
    with pytest.raises(validator.ConfigClosureError, match="bytes differ"):
        validator.validate_exact_bytes(git_lock, configmap_lock=configmap, mounted_lock=mounted)


def test_stale_identity_checks_are_fail_closed() -> None:
    lock = candidate_lock()
    with pytest.raises(validator.ConfigClosureError, match="stale run UID"):
        validator.validate_identity(lock, expected_run_uid="fpct-r2m-deadbee-v1")
    with pytest.raises(validator.ConfigClosureError, match="stale run root"):
        validator.validate_identity(
            lock, expected_root="/netdisk/lijunsi/fpct-confirmatory/fpct-r2m-deadbee-v1"
        )
    with pytest.raises(validator.ConfigClosureError, match="stale scientific commit"):
        validator.validate_identity(lock, expected_commit="2" * 40)


def test_consumer_manifest_blob_mismatch_fails(tmp_path: Path) -> None:
    lock = candidate_lock()
    manifest = {
        "consumers": [{
            "path": "script/experiment/fpct_gpu_r2_runner.py",
            "sha256": "0" * 64,
            "function": "aggregate_pretrained",
            "run_lock_pointers": [],
        }],
        "aggregate_geometry_consumer_ast_sha256": validator.aggregate_consumer_ast_sha(
            ROOT / "script/experiment/fpct_gpu_r2_runner.py"
        ),
    }
    with pytest.raises(validator.ConfigClosureError, match="consumer blob mismatch"):
        validator.validate_consumer_manifest(ROOT, manifest, lock)


def test_r2m_candidate_lock_passes_full_validator(tmp_path: Path) -> None:
    lock = close_candidate_lock(candidate_lock())
    lock_path = tmp_path / "lock.json"
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    result = validator.validate_run_lock(
        ROOT,
        lock_path,
        SCHEMA,
        CANONICAL,
        ROOT / "recipe/eval_recipe/fpct_gpu_r2m/consumer_pointer_manifest.json",
        ROOT / "recipe/eval_recipe/fpct_gpu_r2m/implementation_allowlist.json",
        expected_run_uid="fpct-r2m-1234abc-v1",
        expected_root="/netdisk/lijunsi/fpct-confirmatory/fpct-r2m-1234abc-v1",
        expected_commit="1" * 40,
    )
    assert result["status"] == "GO"
    assert result["aggregate_fixture"]["expansion_mean"] is True
    assert result["aggregate_fixture"]["expansion_p95"] is True


def test_immutable_configmap_contract(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    lock_path.write_text('{"x":1}\n')
    configmap = tmp_path / "configmap.json"
    configmap.write_text(
        json.dumps(
            {
                "kind": "ConfigMap",
                "immutable": True,
                "metadata": {"name": "fpct-r2m-lock"},
                "data": {"immutable_v1_run_lock.json": lock_path.read_text()},
            }
        )
    )
    result = validator.validate_configmap_object(configmap, lock_path)
    assert result["immutable"] is True
    bad = json.loads(configmap.read_text())
    bad["immutable"] = False
    configmap.write_text(json.dumps(bad))
    with pytest.raises(validator.ConfigClosureError, match="not immutable"):
        validator.validate_configmap_object(configmap, lock_path)
