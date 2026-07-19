from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "script/analysis/fpct_3_7_certified_support_audit.py"
SPEC = importlib.util.spec_from_file_location("fpct_3_7_certified_support", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def test_legal_count_masks_invalid_and_rejects_bad_mass() -> None:
    assert audit.legal_count([0, 1, -1, -1], [0.5, 0.5, 0.0, 0.0], 2) == 2
    assert audit.legal_count([0, 3, -1, -1], [1.0, 0.0, 0.0, 0.0], 2) == 1
    with pytest.raises(audit.CertifiedAuditError, match="positive mass on invalid"):
        audit.legal_count([0, 3, -1, -1], [0.5, 0.5, 0.0, 0.0], 2)
    with pytest.raises(audit.CertifiedAuditError, match="duplicate"):
        audit.legal_count([0, 0, -1, -1], [0.5, 0.5, 0.0, 0.0], 2)
    with pytest.raises(audit.CertifiedAuditError, match="L1 normalized"):
        audit.legal_count([0, 1, -1, -1], [0.4, 0.4, 0.0, 0.0], 2)
    with pytest.raises(audit.CertifiedAuditError, match="uniform"):
        audit.legal_count([0, 1, -1, -1], [0.7, 0.3, 0.0, 0.0], 2)
    assert audit.legal_count(
        [0, 2, -1, -1],
        [0.5, 0.5, 0.0, 0.0],
        1,
        original_source_length=3,
    ) == 1


def test_wilson_is_finite() -> None:
    low, high = audit.wilson(30, 100)
    assert low is not None and high is not None
    assert 0 <= low <= 0.30 <= high <= 1


def compact_row(pair: str, task: str, positive: int, total: int) -> dict:
    return {
        "pair": pair,
        "task": task,
        "positive_groups": positive,
        "total_groups": total,
        "support_rate": positive / total,
    }


def test_readiness_uses_certified_support_and_frozen_ranking() -> None:
    rows = []
    values = {
        "tinyllama": (31, 40, 40),
        "qwen25_0p5b": (30, 40, 40),
        "llama32_1b": (10, 10, 100),
        "qwen3_1p7b": (100, 100, 100),
    }
    for pair, counts in values.items():
        for task, count in zip(audit.TASK_ORDER, counts):
            rows.append(compact_row(pair, task, count, 100))
    readiness, status, ranking, selected = audit.derive_readiness(rows)
    assert status == "CROSS_PAIR_PILOT_READY"
    assert ranking == ["tinyllama", "qwen25_0p5b"]
    assert selected == "tinyllama"
    control = next(row for row in readiness if row["pair"] == "qwen3_1p7b")
    assert not control["ready"]


def test_tinyllama_can_be_ready_without_global_rank_one() -> None:
    rows = []
    values = {
        "tinyllama": (31, 31, 40),
        "qwen25_0p5b": (50, 50, 50),
        "llama32_1b": (1, 1, 1),
        "qwen3_1p7b": (100, 100, 100),
    }
    for pair, counts in values.items():
        for task, count in zip(audit.TASK_ORDER, counts):
            rows.append(compact_row(pair, task, count, 100))
    readiness, status, ranking, selected = audit.derive_readiness(rows)
    assert status == "CROSS_PAIR_PILOT_READY"
    assert ranking == ["qwen25_0p5b", "tinyllama"]
    assert selected == "qwen25_0p5b"
    assert next(row for row in readiness if row["pair"] == "tinyllama")["ready"]


@pytest.mark.parametrize(
    "positive,status",
    [(0, "NO_SUPPORT"), (1, "DIAGNOSTIC_ONLY")],
)
def test_not_ready_status_distinguishes_zero_support(positive: int, status: str) -> None:
    rows = [
        compact_row(pair, task, positive, 100)
        for pair in audit.PAIR_ORDER
        for task in audit.TASK_ORDER
    ]
    _readiness, actual, ranking, selected = audit.derive_readiness(rows)
    assert actual == status
    assert ranking == []
    assert selected is None


def test_quantiles_are_deterministic() -> None:
    result = audit.quantiles([1.0, 2.0, 3.0, 4.0])
    assert result == {
        "mean": 2.5,
        "p50": 2.5,
        "p90": pytest.approx(3.7),
        "p95": pytest.approx(3.85),
        "max": 4.0,
    }


def test_script_has_no_model_forward_or_outcome_fields() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "AutoModel" not in source
    assert ".cuda(" not in source
    assert "correctness" not in audit.SAMPLE_COLUMNS
    assert "prediction" not in audit.SAMPLE_COLUMNS
    assert "offset_uncertified_sample" in audit.SAMPLE_COLUMNS
    assert "bonferroni9_wilson_lcb_sensitivity" in audit.COMPACT_COLUMNS


def test_finalize_and_independent_verify_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    execution_sha = "a" * 40
    root = tmp_path / "results" / f"rev_{execution_sha}"
    root.mkdir(parents=True)
    lock = {
        "execution_sha": execution_sha,
        "canonical_input": {
            "total_rows": len(audit.TASK_ORDER) * 4,
            "total_distinct_content_groups": len(audit.TASK_ORDER) * 4,
        },
        "gpu_authorized": False,
    }
    monkeypatch.setattr(audit, "verify_lock", lambda *_args, **_kwargs: (root, lock))
    receiver_config = tmp_path / "qwen-config.json"
    receiver_config.write_text(
        json.dumps(
            {
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "hidden_size": 8,
            }
        )
    )
    monkeypatch.setattr(audit, "RECEIVER_CONFIG_PATH", receiver_config)
    audit.atomic_json(
        root / "controller_state.json",
        {
            "schema_version": 1,
            "execution_sha": execution_sha,
            "state": "FROZEN",
            "completed_shards": [],
            "held_out_test_released": False,
        },
    )

    natural_splits = ("fit", "calibration", "model-selection", "test")
    for pair in audit.PAIR_ORDER:
        for task in audit.TASK_ORDER:
            sample_rows = []
            group_rows = []
            exception_rows = []
            for ordinal, split in enumerate(natural_splits):
                positive = pair == "tinyllama"
                exact = pair == "qwen3_1p7b"
                raw_m1 = 0 if positive else 1
                raw_m2 = 1 if positive else 0
                certified_m1 = raw_m1
                certified_m2 = raw_m2
                sample_hash = f"{pair}-{task}-{split}-sample"
                group_hash = f"{pair}-{task}-{split}-group"
                sample_rows.append({
                    "schema_version": 1,
                    "pair": pair,
                    "pair_type": "same_tokenizer_control" if exact else "heterogeneous",
                    "task": task,
                    "split": split,
                    "sample_key_sha256": sample_hash,
                    "content_group_sha256": group_hash,
                    "eligible_parent_count": 1,
                    "raw_m0": 0,
                    "raw_m1": raw_m1,
                    "raw_m2": raw_m2,
                    "raw_m3": 0,
                    "raw_m4": 0,
                    "certified_m0": 0,
                    "certified_m1": certified_m1,
                    "certified_m2": certified_m2,
                    "certified_m3": 0,
                    "certified_m4": 0,
                    "offset_uncertified_parent_count": 0,
                    "offset_uncertified_sample": 0,
                    "has_raw_m2": int(positive),
                    "has_certified_m2": int(positive),
                    "has_raw_m3": 0,
                    "has_certified_m3": 0,
                    "has_raw_m4": 0,
                    "has_certified_m4": 0,
                    "exact_control": int(exact),
                    "receiver_native_slots": 2,
                    "raw_extra_slots": int(positive),
                    "certified_extra_slots": int(positive),
                    "raw_expansion_ratio": 1.5 if positive else 1.0,
                    "certified_expansion_ratio": 1.5 if positive else 1.0,
                })
                group_rows.append({
                    "schema_version": 1,
                    "pair": pair,
                    "pair_type": "same_tokenizer_control" if exact else "heterogeneous",
                    "task": task,
                    "split": split,
                    "content_group_sha256": group_hash,
                    "group_member_count": 1,
                    "member_consistent": 1,
                    "has_raw_m2": int(positive),
                    "has_certified_m2": int(positive),
                    "has_raw_m3": 0,
                    "has_certified_m3": 0,
                    "has_raw_m4": 0,
                    "has_certified_m4": 0,
                    "offset_uncertified": 0,
                })
                if positive:
                    exception_rows.append({
                        "schema_version": 1,
                        "pair": pair,
                        "task": task,
                        "split": split,
                        "sample_key_sha256": sample_hash,
                        "content_group_sha256": group_hash,
                        "parent_index": ordinal,
                        "raw_m": 2,
                        "sanitized_m": 2,
                        "certified": 1,
                        "offset_uncertified": 0,
                        "reason": "certified_disjoint_partition",
                    })
            directory = audit.shard_dir(root, pair, task)
            directory.mkdir(parents=True)
            artifacts = {}
            for kind, columns, rows in (
                ("sample", audit.SAMPLE_COLUMNS, sample_rows),
                ("group", audit.GROUP_COLUMNS, group_rows),
                ("exception", audit.EXCEPTION_COLUMNS, exception_rows),
            ):
                path = directory / f"{kind}.csv"
                count = audit.atomic_csv(path, columns, rows)
                artifacts[kind] = {
                    "path": str(path),
                    "rows": count,
                    "sha256": audit.sha256_file(path),
                    "bytes": path.stat().st_size,
                }
            audit.atomic_json(
                directory / "manifest.json",
                {
                    "schema_version": 1,
                    "status": "complete",
                    "execution_sha": execution_sha,
                    "pair": pair,
                    "task": task,
                    "artifacts": artifacts,
                },
            )

    result = audit.finalize(execution_sha, Path("ignored"))
    assert result["global_readiness"] == "DIAGNOSTIC_ONLY"
    assert result["confirmatory_pair_status"] == "NO_GO_GPU"
    assert not result["support_gate_passed"]
    audit.verify(execution_sha, Path("ignored"))
    interrupted_state = audit.read_json(root / "controller_state.json")
    interrupted_state.update({
        "state": "FROZEN",
        "completed_shards": [],
        "support_gate_passed": False,
    })
    audit.atomic_json(root / "controller_state.json", interrupted_state)
    resumed = audit.finalize(execution_sha, Path("ignored"))
    assert resumed == result
    assert audit.read_json(root / "controller_state.json")["state"] == (
        "CERTIFIED_SUPPORT_COMPLETE"
    )
