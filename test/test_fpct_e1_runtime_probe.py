from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from script.experiment import fpct_e1_runtime_probe as probe


EXECUTION_SHA = "1" * 40
IMAGE_DIGEST = "registry.example/fpct@sha256:" + "2" * 64
SOURCE_TREE = "3" * 64
SOURCE_VERIFICATION = {
    "status": "GO_MOUNTED_SOURCE_SNAPSHOT",
    "execution_sha": EXECUTION_SHA,
    "git_tree_oid": "4" * 40,
    "git_entries_canonical_sha256": "5" * 64,
    "mounted_tree_canonical_sha256": SOURCE_TREE,
    "entry_count": 10,
    "total_bytes": 2048,
    "receipt_sha256": "6" * 64,
    "receipt_file_sha256": "7" * 64,
    "receipt_bytes": 4096,
}


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
        source_snapshot_verification=SOURCE_VERIFICATION,
        torch_module=_fake_torch(available=True),
    )
    assert payload["protocol_id"] == probe.PROTOCOL_ID
    assert payload["execution_sha"] == EXECUTION_SHA
    assert payload["source_snapshot_verification"] == SOURCE_VERIFICATION
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
        source_snapshot_verification=SOURCE_VERIFICATION,
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
    with pytest.raises(ValueError, match="source-snapshot verification identity"):
        probe.validate_runtime_probe(
            {
                **payload,
                "source_snapshot_verification": {
                    **SOURCE_VERIFICATION,
                    "receipt_sha256": "0" * 63,
                },
            }
        )


def test_atomic_probe_write_is_canonical_and_refuses_overwrite(tmp_path: Path) -> None:
    payload = probe.build_runtime_probe(
        execution_sha=EXECUTION_SHA,
        image_digest=IMAGE_DIGEST,
        source_snapshot_tree_sha=SOURCE_TREE,
        source_snapshot_verification=SOURCE_VERIFICATION,
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


def test_mounted_source_verification_precedes_probe_and_binds_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    calls = []

    def verify(receipt, root, execution_sha):
        calls.append((receipt, root, execution_sha))
        return {
            key: value
            for key, value in SOURCE_VERIFICATION.items()
            if key not in {"receipt_file_sha256", "receipt_bytes"}
        }

    monkeypatch.setattr(probe, "verify_source_snapshot_receipt", verify)
    receipt = tmp_path / ".fpct_e1_source_snapshot_receipt.json"
    receipt_value = {"synthetic": True}
    receipt.write_bytes(probe.canonical_json_bytes(receipt_value))
    root = tmp_path
    observed = probe.verify_mounted_source_snapshot(
        execution_sha=EXECUTION_SHA,
        source_snapshot_tree_sha=SOURCE_TREE,
        source_snapshot_root=root,
        source_snapshot_receipt=receipt,
    )
    assert observed == {
        **{
            key: value
            for key, value in SOURCE_VERIFICATION.items()
            if key not in {"receipt_file_sha256", "receipt_bytes"}
        },
        "receipt_file_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "receipt_bytes": receipt.stat().st_size,
    }
    assert calls == [(receipt, root, EXECUTION_SHA)]

    with pytest.raises(ValueError, match="verification identity"):
        probe.verify_mounted_source_snapshot(
            execution_sha=EXECUTION_SHA,
            source_snapshot_tree_sha="f" * 64,
            source_snapshot_root=root,
            source_snapshot_receipt=receipt,
        )


def _main_arguments(tmp_path: Path) -> list[str]:
    return [
        "--output", str(tmp_path / "runtime_probe.json"),
        "--execution-sha", EXECUTION_SHA,
        "--image-digest", IMAGE_DIGEST,
        "--source-snapshot-tree-sha", SOURCE_TREE,
        "--source-snapshot-root", str(tmp_path / "snapshot"),
        "--source-snapshot-receipt",
        str(tmp_path / "snapshot" / ".fpct_e1_source_snapshot_receipt.json"),
    ]


def test_runtime_probe_main_verifies_mounted_source_before_build_and_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys,
) -> None:
    events = []
    payload = {"synthetic": "runtime-probe"}

    def verify(**kwargs):
        events.append(("verify-mounted", kwargs))
        return dict(SOURCE_VERIFICATION)

    def build(**kwargs):
        events.append(("build", kwargs))
        assert kwargs["source_snapshot_verification"] == SOURCE_VERIFICATION
        return payload

    def write(path, value):
        events.append(("write", path, value))

    monkeypatch.setattr(probe, "verify_mounted_source_snapshot", verify)
    monkeypatch.setattr(probe, "build_runtime_probe", build)
    monkeypatch.setattr(probe, "atomic_write_probe", write)
    arguments = _main_arguments(tmp_path)
    assert probe.main(arguments) == 0
    assert [event[0] for event in events] == ["verify-mounted", "build", "write"]
    verify_call = events[0][1]
    assert verify_call == {
        "execution_sha": EXECUTION_SHA,
        "source_snapshot_tree_sha": SOURCE_TREE,
        "source_snapshot_root": tmp_path / "snapshot",
        "source_snapshot_receipt": tmp_path
        / "snapshot"
        / ".fpct_e1_source_snapshot_receipt.json",
    }
    assert events[2] == ("write", tmp_path / "runtime_probe.json", payload)
    assert json.loads(capsys.readouterr().out) == payload


@pytest.mark.parametrize(
    "missing_flag",
    ("--source-snapshot-root", "--source-snapshot-receipt"),
)
def test_runtime_probe_main_requires_both_mounted_source_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_flag: str,
) -> None:
    arguments = _main_arguments(tmp_path)
    index = arguments.index(missing_flag)
    del arguments[index : index + 2]
    monkeypatch.setattr(
        probe,
        "verify_mounted_source_snapshot",
        lambda **_kwargs: pytest.fail("parser admitted missing mounted-source argument"),
    )
    with pytest.raises(SystemExit) as error:
        probe.main(arguments)
    assert error.value.code == 2
