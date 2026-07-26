from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from script.experiment import fpct_e1_k8s_lock_bundle as bundle
from script.experiment import fpct_e1_runtime_probe as runtime_probe


EXECUTION_SHA = "1" * 40
IMAGE_DIGEST = "registry.example/fpct@sha256:" + "2" * 64
SOURCE_TREE = "3" * 64


class _Cuda:
    def is_initialized(self): return False
    def is_available(self): return False
    def device_count(self): return 0


def _write_json(path: Path, value: object) -> bytes:
    raw = bundle.canonical_json_bytes(value)
    path.write_bytes(raw)
    return raw


def _sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    versions = {name: "1.0" for name in runtime_probe.PACKAGE_DISTRIBUTIONS}
    monkeypatch.setattr(runtime_probe, "_package_version", versions.__getitem__)
    checks = {name: True for name in bundle.REQUIRED_GATE_CHECKS}
    gate = {
        "gate_id": bundle.GATE_ID,
        "status": "GO",
        "checks": checks,
        "evidence": [{"logical_path": "repo://tiny.json", "sha256": "4" * 64}],
    }
    gate_path = tmp_path / "gate.json"
    gate_raw = _write_json(gate_path, gate)
    git_entries = [
        {
            "path": "tiny.py",
            "mode": "100644",
            "object_type": "blob",
            "object_id": "c" * 40,
            "bytes": 1,
            "sha256": "d" * 64,
        }
    ]
    git_entries_sha = bundle.sha256_bytes(bundle.canonical_json_bytes(git_entries))
    source_receipt = {
        "schema_version": 1,
        "protocol_id": bundle.SOURCE_SNAPSHOT_RECEIPT_PROTOCOL_ID,
        "status": "GO_IMMUTABLE_GIT_ARCHIVE",
        "execution_sha": EXECUTION_SHA,
        "dirty": False,
        "portable_root": ".",
        "repo": {
            "head": EXECUTION_SHA,
            "dirty": False,
            "git_tree_oid": "e" * 40,
            "object_format": "sha1",
            "ls_tree_entry_count": 1,
            "ls_tree_canonical_sha256": git_entries_sha,
        },
        "git_entries": git_entries,
        "git_entries_canonical_sha256": git_entries_sha,
        "snapshot": {
            "mounted_tree_canonical_sha256": SOURCE_TREE,
            "file_count": 1,
            "symlink_count": 0,
            "entry_count": 1,
            "total_bytes": 1,
            "portable_root": ".",
            "receipt_relative_path": ".fpct_e1_source_snapshot_receipt.json",
            "exact_git_archive_content": True,
            "no_extra_paths": True,
            "no_symlink_escape": True,
        },
        "construction": {
            "source": "git archive <execution_sha>",
            "git_ls_tree": "git ls-tree -r -z --full-tree --long <execution_sha>",
            "manual_copy_allowed": False,
            "worktree_clean_required": True,
        },
    }
    source_receipt["receipt_sha256"] = bundle.sha256_bytes(
        bundle.canonical_json_bytes(source_receipt)
    )
    source_receipt_path = tmp_path / "source-snapshot-receipt.json"
    source_receipt_raw = _write_json(source_receipt_path, source_receipt)
    source_verification = {
        "status": "GO_MOUNTED_SOURCE_SNAPSHOT",
        "execution_sha": EXECUTION_SHA,
        "git_tree_oid": source_receipt["repo"]["git_tree_oid"],
        "git_entries_canonical_sha256": git_entries_sha,
        "mounted_tree_canonical_sha256": SOURCE_TREE,
        "entry_count": 1,
        "total_bytes": 1,
        "receipt_sha256": source_receipt["receipt_sha256"],
    }
    runtime_source_verification = {
        **source_verification,
        "receipt_file_sha256": bundle.sha256_bytes(source_receipt_raw),
        "receipt_bytes": len(source_receipt_raw),
    }
    runtime = runtime_probe.build_runtime_probe(
        execution_sha=EXECUTION_SHA,
        image_digest=IMAGE_DIGEST,
        source_snapshot_tree_sha=SOURCE_TREE,
        source_snapshot_verification=runtime_source_verification,
        torch_module=SimpleNamespace(
            cuda=_Cuda(), version=SimpleNamespace(cuda="12.4")
        ),
    )
    runtime_path = tmp_path / "runtime.json"
    runtime_raw = _write_json(runtime_path, runtime)
    plan = {
        "schema_version": 2,
        "protocol_id": bundle.PLAN_PROTOCOL_ID,
        "status": "PREPARED_NO_MODEL_LOAD",
        "runtime_lock": {
            "execution_sha": EXECUTION_SHA,
            "image_digest": IMAGE_DIGEST,
            "runtime_provenance": {
                "sha256": bundle.sha256_bytes(runtime_raw),
                "identity": {
                    "schema_version": runtime["schema_version"],
                    "protocol_id": runtime["protocol_id"],
                    "status": runtime["status"],
                    "execution_sha": EXECUTION_SHA,
                    "image_digest": IMAGE_DIGEST,
                    "source_snapshot_tree_sha": SOURCE_TREE,
                    "source_snapshot_verification": runtime_source_verification,
                },
            },
        },
        "source_snapshot": {
            "execution_sha": EXECUTION_SHA,
            "tree": {"tree_sha256": "f" * 64},
            "canonical_tree_sha256": SOURCE_TREE,
            "git_receipt": {
                "sha256": bundle.sha256_bytes(source_receipt_raw),
                "bytes": len(source_receipt_raw),
                "verification": source_verification,
            },
        },
        "instrumentation_gate": {
            "sha256": bundle.sha256_bytes(gate_raw),
            "gate_id": bundle.GATE_ID,
            "status": "GO",
            "checks": checks,
        },
        "raw_topology_lock": {
            "status": "GO_MODEL_OUTPUT_FREE",
            "manifest": {"sha256": "6" * 64},
        },
    }
    plan["plan_sha256"] = bundle._plan_hash(plan)
    plan_path = tmp_path / "plan.json"
    _write_json(plan_path, plan)
    return plan_path, gate_path, runtime_path, source_receipt_path


def _finalized_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    plan_path, _, _, _ = _sources(tmp_path, monkeypatch)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    merged = {
        "relative_path": "e1_2_endpoints_rows.parquet",
        "sha256": "7" * 64,
        "bytes": 1234,
        "row_count": 42,
        "mode": "0o644",
    }
    analyzer = {
        "e1_mechanism_summary.json": {
            "sha256": "8" * 64, "bytes": 321, "mode": "0o644"
        },
        "e1_layer_head_summary.csv": {
            "sha256": "9" * 64, "bytes": 654, "mode": "0o644"
        },
    }
    verification = {"status": "GO", "row_count": 42, "tree_sha256": "a" * 64}
    stage = {
        "schema_version": 1,
        "protocol_id": bundle.PLAN_PROTOCOL_ID,
        "status": "GO",
        "stage": bundle.SOURCE_STAGE,
        "plan_sha256": plan["plan_sha256"],
        "closure_sha256": "b" * 64,
        "merged_capture": merged,
        "analyzer_verification": verification,
        "analyzer_artifacts": analyzer,
        "e1_pilot_consumed": False,
        "confirmatory_consumed": False,
    }
    stage_path = tmp_path / "stage-manifest.json"
    stage_raw = _write_json(stage_path, stage)
    receipt = {
        "schema_version": 1,
        "protocol_id": bundle.PLAN_PROTOCOL_ID,
        "status": "GO",
        "stage": bundle.FINALIZED_STAGE,
        "source_stage": bundle.SOURCE_STAGE,
        "plan_sha256": plan["plan_sha256"],
        "closure_sha256": stage["closure_sha256"],
        "raw_topology_lock": plan["raw_topology_lock"],
        "output_dir": "output://finalized/e1_2",
        "stage_artifact_manifest": {
            "sha256": bundle.sha256_bytes(stage_raw),
            "bytes": len(stage_raw),
        },
        "merged_capture": merged,
        "analyzer_verification": verification,
        "analyzer_artifacts": analyzer,
        "e1_pilot_consumed": False,
        "confirmatory_consumed": False,
    }
    receipt_path = tmp_path / "finalized-receipt.json"
    _write_json(receipt_path, receipt)
    return plan_path, receipt_path, stage_path


def test_builds_two_immutable_size_bounded_configmaps_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, gate, runtime, source_receipt = _sources(tmp_path, monkeypatch)
    output = tmp_path / "bundle"
    manifest = bundle.build_lock_bundle(
        plan_path=plan,
        instrumentation_gate_path=gate,
        runtime_probe_path=runtime,
        source_snapshot_receipt_path=source_receipt,
        output_dir=output,
    )
    suffix = f"{EXECUTION_SHA[:8]}-{manifest['plan_sha256'][:12]}"
    assert {record["name"] for record in manifest["configmaps"].values()} == {
        f"fpct-e1-lock-{suffix}", f"fpct-e1-runtime-{suffix}"
    }
    assert set(manifest["sources"]) == {
        "plan", "instrumentation_gate", "runtime_probe", "source_snapshot_receipt"
    }
    assert set(manifest["configmaps"]["plan_gate"]["keys"]) == {
        bundle.PLAN_KEY, bundle.GATE_KEY, bundle.SOURCE_SNAPSHOT_RECEIPT_KEY
    }
    for record in manifest["configmaps"].values():
        assert record["immutable"] is True
        assert record["data_payload_bytes"] < bundle.CONFIGMAP_LIMIT_BYTES
        assert record["serialized_object_bytes"] < bundle.CONFIGMAP_LIMIT_BYTES
        assert record["yaml_bytes"] < bundle.CONFIGMAP_LIMIT_BYTES
        raw = Path(record["yaml_path"]).read_bytes()
        assert bundle.sha256_bytes(raw) == record["yaml_sha256"]
        assert yaml.safe_load(raw)["immutable"] is True
    frozen_manifest = output / "fpct_e1_k8s_lock_bundle_manifest.json"
    assert json.loads(frozen_manifest.read_text(encoding="utf-8")) == manifest


def test_configmap_rejects_any_serialized_or_data_payload_at_one_mib() -> None:
    with pytest.raises(ValueError, match="1MiB"):
        bundle._configmap_yaml(
            name="fpct-e1-oversized",
            namespace="c2c-research",
            execution_sha=EXECUTION_SHA,
            plan_sha="5" * 64,
            values={"large.json": b"x" * bundle.CONFIGMAP_LIMIT_BYTES},
        )


def test_verify_rehashes_kubectl_exported_mounted_key_bytes_and_rejects_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan, gate, runtime, source_receipt = _sources(tmp_path, monkeypatch)
    output = tmp_path / "bundle"
    manifest = bundle.build_lock_bundle(
        plan_path=plan,
        instrumentation_gate_path=gate,
        runtime_probe_path=runtime,
        source_snapshot_receipt_path=source_receipt,
        output_dir=output,
    )
    exports = []
    for index, record in enumerate(manifest["configmaps"].values()):
        obj = yaml.safe_load(Path(record["yaml_path"]).read_text(encoding="utf-8"))
        obj["metadata"]["resourceVersion"] = str(index + 1)
        path = tmp_path / f"export-{index}.json"
        path.write_text(json.dumps(obj), encoding="utf-8")
        exports.append(path)
    bundle_manifest_path = output / "fpct_e1_k8s_lock_bundle_manifest.json"
    report = bundle.verify_exported_configmaps(
        bundle_manifest_path=bundle_manifest_path,
        configmap_json_paths=exports,
    )
    assert report["status"] == "VERIFIED_MOUNTED_KEY_BYTES"
    assert report["bundle_manifest_sha256"] == bundle.sha256_bytes(
        bundle_manifest_path.read_bytes()
    )
    receipt_path = tmp_path / "initial-mounted-byte-receipt.json"
    cli = [
        "verify", "--bundle-manifest", str(bundle_manifest_path),
        "--output", str(receipt_path),
    ]
    for exported in exports:
        cli.extend(("--configmap-json", str(exported)))
    assert bundle.main(cli) == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == report
    capsys.readouterr()

    bad = json.loads(exports[0].read_text(encoding="utf-8"))
    first_key = next(iter(bad["data"]))
    bad["data"][first_key] += "tamper"
    exports[0].write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="mounted ConfigMap key bytes changed"):
        bundle.verify_exported_configmaps(
            bundle_manifest_path=output / "fpct_e1_k8s_lock_bundle_manifest.json",
            configmap_json_paths=exports,
        )


def test_source_sha_or_runtime_identity_mismatch_fails_before_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, gate, runtime, source_receipt = _sources(tmp_path, monkeypatch)
    runtime.write_text(runtime.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="runtime probe byte SHA"):
        bundle.build_lock_bundle(
            plan_path=plan,
            instrumentation_gate_path=gate,
            runtime_probe_path=runtime,
            source_snapshot_receipt_path=source_receipt,
            output_dir=tmp_path / "bundle",
        )


@pytest.mark.parametrize(
    "image",
    [
        "sha256:" + "2" * 64,
        "registry.example/fpct:latest",
        "registry.example/fpct:tag@sha256:" + "2" * 64,
    ],
)
def test_bundle_rejects_unnamed_or_tagged_image_references(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image: str,
) -> None:
    plan_path, gate, runtime, source_receipt = _sources(tmp_path, monkeypatch)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["runtime_lock"]["image_digest"] = image
    plan["plan_sha256"] = bundle._plan_hash(plan)
    _write_json(plan_path, plan)
    with pytest.raises(ValueError, match="execution/image identity"):
        bundle.build_lock_bundle(
            plan_path=plan_path,
            instrumentation_gate_path=gate,
            runtime_probe_path=runtime,
            source_snapshot_receipt_path=source_receipt,
            output_dir=tmp_path / "bundle",
        )


def test_bundle_rejects_probe_bound_to_different_receipt_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_path, gate, runtime_path, source_receipt = _sources(tmp_path, monkeypatch)
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime["source_snapshot_verification"]["receipt_file_sha256"] = "0" * 64
    runtime_raw = _write_json(runtime_path, runtime)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    runtime_lock = plan["runtime_lock"]["runtime_provenance"]
    runtime_lock["sha256"] = bundle.sha256_bytes(runtime_raw)
    runtime_lock["identity"]["source_snapshot_verification"] = runtime[
        "source_snapshot_verification"
    ]
    plan["plan_sha256"] = bundle._plan_hash(plan)
    _write_json(plan_path, plan)
    with pytest.raises(ValueError, match="source-receipt identity"):
        bundle.build_lock_bundle(
            plan_path=plan_path,
            instrumentation_gate_path=gate,
            runtime_probe_path=runtime_path,
            source_snapshot_receipt_path=source_receipt,
            output_dir=tmp_path / "bundle",
        )


def test_initial_bundle_rejects_source_snapshot_receipt_byte_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, gate, runtime, source_receipt = _sources(tmp_path, monkeypatch)
    source_receipt.write_text(
        source_receipt.read_text(encoding="utf-8") + " ", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="receipt byte SHA/size"):
        bundle.build_lock_bundle(
            plan_path=plan,
            instrumentation_gate_path=gate,
            runtime_probe_path=runtime,
            source_snapshot_receipt_path=source_receipt,
            output_dir=tmp_path / "bundle",
        )


def test_build_finalized_binds_plan_receipt_closure_and_artifact_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    plan, receipt, stage = _finalized_sources(tmp_path, monkeypatch)
    output = tmp_path / "finalized-bundle"
    manifest = bundle.build_finalized_lock_bundle(
        plan_path=plan,
        finalized_receipt_path=receipt,
        stage_manifest_path=stage,
        output_dir=output,
    )
    expected_name = (
        f"fpct-e1-finalized-{manifest['plan_sha256'][:12]}-"
        f"{manifest['receipt_sha256'][:12]}"
    )
    record = manifest["configmaps"]["finalized"]
    assert record["name"] == expected_name
    assert set(record["keys"]) == {
        bundle.FINALIZED_RECEIPT_KEY, bundle.STAGE_MANIFEST_KEY
    }
    assert record["serialized_object_bytes"] < bundle.CONFIGMAP_LIMIT_BYTES
    assert len(manifest["artifact_tree"]) == 3
    assert bundle.SHA256.fullmatch(manifest["artifact_tree_sha256"])
    obj = yaml.safe_load(Path(record["yaml_path"]).read_text(encoding="utf-8"))
    exported = tmp_path / "finalized-export.json"
    exported.write_text(json.dumps(obj), encoding="utf-8")
    bundle_manifest_path = output / "fpct_e1_k8s_finalized_lock_bundle_manifest.json"
    verified = bundle.verify_finalized_exported_configmap(
        bundle_manifest_path=bundle_manifest_path,
        configmap_json_paths=[exported],
    )
    assert verified["status"] == "VERIFIED_FINALIZED_MOUNTED_KEY_BYTES"
    assert verified["bundle_manifest_sha256"] == bundle.sha256_bytes(
        bundle_manifest_path.read_bytes()
    )
    receipt_path = tmp_path / "finalized-mounted-byte-receipt.json"
    assert bundle.main([
        "verify-finalized", "--bundle-manifest", str(bundle_manifest_path),
        "--configmap-json", str(exported), "--output", str(receipt_path),
    ]) == 0
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == verified
    capsys.readouterr()


@pytest.mark.parametrize("command", ["verify", "verify-finalized"])
def test_verification_cli_requires_an_operative_output_receipt(
    tmp_path: Path, command: str,
) -> None:
    with pytest.raises(SystemExit):
        bundle.main([
            command,
            "--bundle-manifest", str(tmp_path / "bundle.json"),
            "--configmap-json", str(tmp_path / "export.json"),
        ])


def test_verify_finalized_rejects_mounted_tamper_and_nonimmutable_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, receipt, stage = _finalized_sources(tmp_path, monkeypatch)
    output = tmp_path / "finalized-bundle"
    manifest = bundle.build_finalized_lock_bundle(
        plan_path=plan,
        finalized_receipt_path=receipt,
        stage_manifest_path=stage,
        output_dir=output,
    )
    record = manifest["configmaps"]["finalized"]
    original = yaml.safe_load(Path(record["yaml_path"]).read_text(encoding="utf-8"))
    exported = tmp_path / "finalized-export.json"

    tampered = json.loads(json.dumps(original))
    tampered["data"][bundle.FINALIZED_RECEIPT_KEY] += " "
    exported.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="mounted ConfigMap key bytes changed"):
        bundle.verify_finalized_exported_configmap(
            bundle_manifest_path=output / "fpct_e1_k8s_finalized_lock_bundle_manifest.json",
            configmap_json_paths=[exported],
        )

    nonimmutable = json.loads(json.dumps(original))
    nonimmutable["immutable"] = False
    exported.write_text(json.dumps(nonimmutable), encoding="utf-8")
    with pytest.raises(ValueError, match="immutability"):
        bundle.verify_finalized_exported_configmap(
            bundle_manifest_path=output / "fpct_e1_k8s_finalized_lock_bundle_manifest.json",
            configmap_json_paths=[exported],
        )


def test_build_finalized_rejects_receipt_bound_to_wrong_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, receipt, stage = _finalized_sources(tmp_path, monkeypatch)
    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["plan_sha256"] = "f" * 64
    _write_json(receipt, value)
    with pytest.raises(ValueError, match="receipt identity/plan binding"):
        bundle.build_finalized_lock_bundle(
            plan_path=plan,
            finalized_receipt_path=receipt,
            stage_manifest_path=stage,
            output_dir=tmp_path / "finalized-bundle",
        )
