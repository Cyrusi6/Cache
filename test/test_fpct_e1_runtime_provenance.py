from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from rosetta.utils.evaluate import load_projector_state_attested
from script.experiment.fpct_e1_capture_runner import sha256_file
from script.experiment.fpct_e1_prepare_input_lock import runtime_asset_tree
from script.experiment.fpct_e1_runtime_backend import verify_runtime_assets_before_load


def _asset_tree(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir()
    (path / "config.json").write_text("{}\n", encoding="utf-8")
    (path / "tokenizer.json").write_text('{"version":"1"}\n', encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"tiny-fixture-weights")
    return path


def test_runtime_asset_tree_records_root_symlink_and_detects_weight_tamper(tmp_path: Path) -> None:
    real = _asset_tree(tmp_path, "receiver-real")
    linked = tmp_path / "receiver"
    linked.symlink_to(real.name, target_is_directory=True)
    frozen = runtime_asset_tree(linked)
    assert frozen["root_kind"] == "symlink_dir"
    assert frozen["root_symlink_target"] == real.name
    assert {row["relative_path"] for row in frozen["files"]} == {
        "config.json",
        "tokenizer.json",
        "model.safetensors",
    }
    assert frozen["file_count"] == 3
    (real / "model.safetensors").write_bytes(b"tampered")
    observed = runtime_asset_tree(linked)
    assert observed["tree_sha256"] != frozen["tree_sha256"]


def test_backend_verifies_both_complete_asset_trees_before_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receiver = _asset_tree(tmp_path, "receiver")
    sender = _asset_tree(tmp_path, "sender")
    assets = {
        "receiver": {
            "model_id": "Qwen/Qwen3-0.6B",
            **runtime_asset_tree(receiver),
        },
        "sender": {
            "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            **runtime_asset_tree(sender),
        },
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runtime_assets": assets}), encoding="utf-8")
    paths = {
        "Qwen/Qwen3-0.6B": str(receiver),
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": str(sender),
    }
    monkeypatch.setattr(
        "rosetta.utils.model_loading.resolve_model_path", lambda model_id: paths[model_id]
    )
    request = {
        "input_lock_manifest_path": str(manifest),
        "input_lock_manifest_sha256": sha256_file(manifest),
        "runtime_assets": assets,
    }
    record = verify_runtime_assets_before_load(request)
    assert set(record) == {"receiver", "sender"}
    assert all(value["verified_before_model_or_tokenizer_load"] for value in record.values())

    (sender / "model.safetensors").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sender runtime asset tree changed"):
        verify_runtime_assets_before_load(request)


def test_projector_strict_attested_and_legacy_non_strict_modes() -> None:
    exact = torch.nn.Linear(3, 2)
    state = {key: value.clone() for key, value in exact.state_dict().items()}
    target = torch.nn.Linear(3, 2)
    strict = load_projector_state_attested(
        target, state, strict_attested=True
    )
    assert strict["strict"] is True
    assert strict["missing_keys"] == []
    assert strict["unexpected_keys"] == []
    for name, value in state.items():
        assert torch.equal(target.state_dict()[name], value)

    missing_bias = {"weight": state["weight"]}
    with pytest.raises(RuntimeError):
        load_projector_state_attested(
            torch.nn.Linear(3, 2), missing_bias, strict_attested=True
        )
    legacy = load_projector_state_attested(
        torch.nn.Linear(3, 2), missing_bias, strict_attested=False
    )
    assert legacy["strict"] is False
    assert legacy["missing_keys"] == ["bias"]
    assert legacy["unexpected_keys"] == []
