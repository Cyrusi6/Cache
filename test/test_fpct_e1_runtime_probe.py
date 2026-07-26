from __future__ import annotations

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from script.experiment import fpct_e1_runtime_probe as probe


EXECUTION_SHA = "1" * 40
IMAGE_DIGEST = "sha256:" + "2" * 64
SOURCE_TREE = "3" * 64


class _FakeCuda:
    def __init__(self, *, available: bool) -> None:
        self.available = available
        self.initialized = False

    def is_initialized(self) -> bool:
        return self.initialized

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1 if self.available else 0

    def get_device_properties(self, index: int):
        assert index == 0
        self.initialized = True
        return SimpleNamespace(
            name="Tiny GPU",
            total_memory=48 * 1024**3,
            major=8,
            minor=9,
            multi_processor_count=128,
            uuid="GPU-tiny",
        )


def _fake_torch(*, available: bool):
    return SimpleNamespace(
        cuda=_FakeCuda(available=available),
        version=SimpleNamespace(cuda="12.4"),
    )


@pytest.fixture(autouse=True)
def _versions(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {
        "torch": "2.6.0",
        "transformers": "4.51.0",
        "tokenizers": "0.21.0",
        "pyarrow": "19.0.0",
    }
    monkeypatch.setattr(probe, "_package_version", versions.__getitem__)


def test_gpu_probe_freezes_identity_versions_device_and_false_load_flags() -> None:
    payload = probe.build_runtime_probe(
        execution_sha=EXECUTION_SHA,
        image_digest=IMAGE_DIGEST,
        source_snapshot_tree_sha=SOURCE_TREE,
        torch_module=_fake_torch(available=True),
    )
    assert payload["protocol_id"] == probe.PROTOCOL_ID
    assert payload["execution_sha"] == EXECUTION_SHA
    assert payload["runtime"]["packages"]["transformers"] == "4.51.0"
    assert payload["runtime"]["torch_cuda_build"] == "12.4"
    assert payload["runtime"]["cuda"]["device_count"] == 1
    assert payload["runtime"]["cuda"]["devices"][0]["total_memory_bytes"] == 48 * 1024**3
    assert payload["firewall"] == {
        "model_loaded": False,
        "tokenizer_loaded": False,
        "checkpoint_loaded": False,
        "model_output_accessed": False,
    }
    assert payload["offline"]["HF_HUB_OFFLINE"] == "1"


def test_cpu_probe_has_no_devices_and_schema_rejects_identity_tamper() -> None:
    payload = probe.build_runtime_probe(
        execution_sha=EXECUTION_SHA,
        image_digest=IMAGE_DIGEST,
        source_snapshot_tree_sha=SOURCE_TREE,
        torch_module=_fake_torch(available=False),
    )
    assert payload["runtime"]["cuda"]["available"] is False
    assert payload["runtime"]["cuda"]["devices"] == []
    with pytest.raises(ValueError, match="identity"):
        probe.validate_runtime_probe({**payload, "protocol_id": "changed"})
    with pytest.raises(ValueError, match="runtime schema"):
        probe.validate_runtime_probe(
            {**payload, "runtime": {key: value for key, value in payload["runtime"].items() if key != "python"}}
        )


def test_atomic_probe_write_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    payload = probe.build_runtime_probe(
        execution_sha=EXECUTION_SHA,
        image_digest=IMAGE_DIGEST,
        source_snapshot_tree_sha=SOURCE_TREE,
        torch_module=_fake_torch(available=False),
    )
    output = tmp_path / "runtime_probe.json"
    probe.atomic_write_probe(output, payload)
    assert output.read_bytes() == probe.canonical_json_bytes(payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    with pytest.raises(FileExistsError):
        probe.atomic_write_probe(output, payload)


def test_probe_source_cannot_import_or_load_model_tokenizer_checkpoint() -> None:
    path = Path(probe.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any("AutoModel" in name or "AutoTokenizer" in name for name in imported)
    assert "from_pretrained" not in source
    assert "torch.load" not in source
    assert "checkpoint_loaded\": False" in source
