from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from script.experiment import fpct_e1_runtime_probe_renderer as renderer
from script.experiment import fpct_e1_source_snapshot_lock as source_lock


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "recipe/k8s/fpct_e1/runtime_probe_job.yaml"
RENDERER_SOURCE = Path(renderer.__file__).resolve()
RUNTIME_PROBE_SOURCE = REPO_ROOT / renderer.RUNTIME_PROBE_RELATIVE.as_posix()
SOURCE_LOCK_SOURCE = REPO_ROOT / renderer.SOURCE_LOCK_RELATIVE.as_posix()
EXECUTION_SHA = "1" * 40
IMAGE_DIGEST = "registry.example/fpct@sha256:" + "2" * 64
SOURCE_TREE_SHA = "3" * 64
REQUIRED_PLACEHOLDERS = set(renderer.PLACEHOLDER_COUNTS)


def _make_source_snapshot(tmp_path: Path, monkeypatch):
    source = tmp_path / "source-snapshot"
    source.mkdir()
    source_template = source / Path(renderer.TEMPLATE_RELATIVE.as_posix())
    source_producer = source / Path(renderer.PRODUCER_RELATIVE.as_posix())
    source_runtime_probe = source / Path(renderer.RUNTIME_PROBE_RELATIVE.as_posix())
    source_lock_file = source / Path(renderer.SOURCE_LOCK_RELATIVE.as_posix())
    source_template.parent.mkdir(parents=True)
    source_producer.parent.mkdir(parents=True)
    source_template.write_bytes(TEMPLATE.read_bytes())
    source_producer.write_bytes(RENDERER_SOURCE.read_bytes())
    source_runtime_probe.write_bytes(RUNTIME_PROBE_SOURCE.read_bytes())
    source_lock_file.write_bytes(SOURCE_LOCK_SOURCE.read_bytes())
    files = [source_template, source_producer, source_runtime_probe, source_lock_file]
    git_entries = []
    mounted_entries = []
    for index, path in enumerate(sorted(files)):
        relative = path.relative_to(source).as_posix()
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        git_entries.append(
            {
                "path": relative,
                "mode": "100644",
                "object_type": "blob",
                "object_id": f"{index + 1:040x}",
                "bytes": len(raw),
                "sha256": digest,
            }
        )
        mounted_entries.append(
            {
                "path": relative,
                "mode": "100644",
                "kind": "file",
                "bytes": len(raw),
                "sha256": digest,
            }
        )
    entries_sha = source_lock.sha256_bytes(
        source_lock.canonical_json_bytes(git_entries)
    )
    mounted_tree_sha = source_lock.sha256_bytes(
        source_lock.canonical_json_bytes(mounted_entries)
    )
    receipt = {
        "schema_version": source_lock.SCHEMA_VERSION,
        "protocol_id": source_lock.PROTOCOL_ID,
        "status": "GO_IMMUTABLE_GIT_ARCHIVE",
        "execution_sha": EXECUTION_SHA,
        "dirty": False,
        "portable_root": ".",
        "repo": {
            "head": EXECUTION_SHA,
            "dirty": False,
            "git_tree_oid": "a" * 40,
            "object_format": "sha1",
            "ls_tree_entry_count": len(git_entries),
            "ls_tree_canonical_sha256": entries_sha,
        },
        "git_entries": git_entries,
        "git_entries_canonical_sha256": entries_sha,
        "snapshot": {
            "mounted_tree_canonical_sha256": mounted_tree_sha,
            "file_count": len(mounted_entries),
            "symlink_count": 0,
            "entry_count": len(mounted_entries),
            "total_bytes": sum(row["bytes"] for row in mounted_entries),
            "portable_root": ".",
            "receipt_relative_path": renderer.SOURCE_RECEIPT_RELATIVE.as_posix(),
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
    receipt["receipt_sha256"] = source_lock.receipt_sha256(receipt)
    receipt_path = source / Path(renderer.SOURCE_RECEIPT_RELATIVE.as_posix())
    receipt_path.write_bytes(source_lock.canonical_json_bytes(receipt))
    monkeypatch.setattr(renderer, "__file__", str(source_producer))
    return source, source_template, mounted_tree_sha, receipt_path


def _render(
    tmp_path: Path,
    monkeypatch,
    output_name: str = "rendered",
    source_bundle=None,
):
    source, source_template, tree_sha, receipt = (
        source_bundle or _make_source_snapshot(tmp_path, monkeypatch)
    )
    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir(exist_ok=True)
    runtime_root = runtime_parent / EXECUTION_SHA
    result = renderer.render_runtime_probe_job(
        template_path=source_template,
        output_dir=tmp_path / output_name,
        execution_sha=EXECUTION_SHA,
        image_digest=IMAGE_DIGEST,
        source_snapshot_host=str(source),
        source_snapshot_tree_sha=tree_sha,
        runtime_probe_run_root=str(runtime_root),
    )
    return result, source, source_template, tree_sha, receipt, runtime_root, tmp_path / output_name


def _snapshot_bytes_state(source: Path):
    return tuple(
        (path.relative_to(source).as_posix(), hashlib.sha256(path.read_bytes()).hexdigest())
        for path in sorted(source.rglob("*"))
        if path.is_file()
    )


def test_runtime_probe_template_has_exact_five_placeholder_identities() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    observed = set(re.findall(r"__[A-Z0-9_]+__", text))
    assert observed == REQUIRED_PLACEHOLDERS
    assert len(observed) == 5
    assert {
        placeholder: text.count(placeholder) for placeholder in observed
    } == renderer.PLACEHOLDER_COUNTS
    assert "/netdisk/lijunsi/fpct-e1" not in text


def test_renderer_is_deterministic_unique_root_only_and_self_hashed(
    tmp_path: Path, monkeypatch,
) -> None:
    bundle = _make_source_snapshot(tmp_path, monkeypatch)
    first, source, source_template, tree_sha, receipt, runtime_root, first_dir = _render(
        tmp_path, monkeypatch, "render-a", bundle
    )
    second, _, _, _, _, _, second_dir = _render(
        tmp_path, monkeypatch, "render-b", bundle
    )
    first_yaml = (first_dir / renderer.RENDERED_JOB_FILENAME).read_bytes()
    second_yaml = (second_dir / renderer.RENDERED_JOB_FILENAME).read_bytes()
    first_manifest = (first_dir / renderer.MANIFEST_FILENAME).read_bytes()
    second_manifest = (second_dir / renderer.MANIFEST_FILENAME).read_bytes()
    assert first_yaml == second_yaml
    assert first_manifest == second_manifest
    assert first == second
    assert first["status"] == renderer.STATUS
    assert first["rendered_job"]["sha256"] == hashlib.sha256(first_yaml).hexdigest()
    assert first["rendered_job"]["bytes"] == len(first_yaml)
    projected = dict(first)
    manifest_sha = projected.pop("manifest_sha256")
    assert manifest_sha == renderer.sha256_bytes(renderer.canonical_json_bytes(projected))
    assert first["runtime_directory"] == {
        "host_root": str(runtime_root),
        "container_root": str(runtime_root),
        "same_absolute_path": True,
        "execution_sha_terminal_component": True,
        "output_path": str(runtime_root / renderer.OUTPUT_FILENAME),
        "writable_scope": "this execution's runtime-probe directory only",
    }
    assert first["source_snapshot"]["host_path"] == str(source)
    assert first["source_snapshot"]["canonical_tree_sha256"] == tree_sha
    assert first["source_snapshot"]["receipt_path"] == str(receipt)
    assert first["source_snapshot"]["verification"]["status"] == "GO_MOUNTED_SOURCE_SNAPSHOT"
    assert first["source_snapshot"]["verification"]["receipt_sha256"] == json.loads(
        receipt.read_text()
    )["receipt_sha256"]
    producer = first["producer"]
    producer_path = Path(producer["path"])
    assert producer_path == Path(renderer.__file__).resolve()
    assert producer["logical_path"] == f"source://{renderer.PRODUCER_RELATIVE.as_posix()}"
    assert producer["exact_snapshot_producer"] is True
    assert producer["bytes"] == producer_path.stat().st_size
    assert producer["sha256"] == renderer.sha256_file(producer_path)
    assert first["firewall"] == {
        "kubectl_invoked": False,
        "network_accessed": False,
        "model_loaded": False,
        "tokenizer_loaded": False,
        "checkpoint_loaded": False,
        "model_output_accessed": False,
    }

    manifest = yaml.safe_load(first_yaml)
    assert re.search(r"__[A-Z0-9_]+__", first_yaml.decode()) is None
    assert manifest["metadata"]["name"] == f"fpct-e1-runtime-probe-{EXECUTION_SHA}"
    pod = manifest["spec"]["template"]["spec"]
    assert pod["nodeName"] == "4090-48gx2"
    container = pod["containers"][0]
    assert container["image"] == IMAGE_DIGEST
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "readOnlyRootFilesystem": True,
    }
    environment = {row["name"]: row["value"] for row in container["env"]}
    assert environment == renderer.EXPECTED_ENVIRONMENT
    arguments = renderer._argument_map(container["args"])
    assert arguments["--source-snapshot-root"] == "/opt/fpct"
    assert arguments["--source-snapshot-receipt"] == "/opt/fpct/.fpct_e1_source_snapshot_receipt.json"
    mounts = {row["name"]: row for row in container["volumeMounts"]}
    assert mounts["source-snapshot"] == {
        "name": "source-snapshot",
        "mountPath": "/opt/fpct",
        "readOnly": True,
    }
    assert mounts["runtime-output"] == {
        "name": "runtime-output",
        "mountPath": str(runtime_root),
    }
    assert mounts["tmp"] == {"name": "tmp", "mountPath": "/tmp"}
    volumes = {row["name"]: row for row in pod["volumes"]}
    assert volumes["source-snapshot"]["hostPath"] == {
        "path": str(source),
        "type": "Directory",
    }
    assert volumes["runtime-output"]["hostPath"] == {
        "path": str(runtime_root),
        "type": "DirectoryOrCreate",
    }


def test_renderer_rejects_overwrite_unresolved_or_changed_template(
    tmp_path: Path, monkeypatch,
) -> None:
    bundle = _make_source_snapshot(tmp_path, monkeypatch)
    result, source, source_template, tree_sha, _, runtime_root, output = _render(
        tmp_path, monkeypatch, source_bundle=bundle
    )
    assert result["status"] == renderer.STATUS
    with pytest.raises(FileExistsError):
        renderer.render_runtime_probe_job(
            template_path=source_template,
            output_dir=output,
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_root),
        )

    changed = TEMPLATE.read_text(encoding="utf-8").replace(
        "__RUNTIME_PROBE_RUN_ROOT__", "/broad/shared/root", 1
    )
    with pytest.raises(ValueError, match="placeholder universe|occurrence counts"):
        renderer._validate_template(changed)

    extra = TEMPLATE.read_text(encoding="utf-8") + "\n# __UNAPPROVED_PLACEHOLDER__\n"
    with pytest.raises(ValueError, match="placeholder universe"):
        renderer._validate_template(extra)

    with pytest.raises(ValueError, match="template_path must exactly equal"):
        renderer.render_runtime_probe_job(
            template_path=TEMPLATE,
            output_dir=tmp_path / "external-template-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_root),
        )


@pytest.mark.parametrize(
    "runtime_root,match",
    [
        ("/netdisk/lijunsi/fpct-e1", "full execution SHA"),
        ("relative/" + EXECUTION_SHA, "absolute POSIX"),
        ("/runtime/../" + EXECUTION_SHA, "canonical"),
    ],
)
def test_renderer_rejects_broad_relative_or_noncanonical_runtime_roots(
    tmp_path: Path, runtime_root: str, match: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match=match):
        renderer.render_runtime_probe_job(
            template_path=TEMPLATE,
            output_dir=tmp_path / "output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=SOURCE_TREE_SHA,
            runtime_probe_run_root=runtime_root,
        )


def test_renderer_rejects_runtime_output_inside_source_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    runtime_root = source / "runtime" / EXECUTION_SHA
    with pytest.raises(ValueError, match="must be disjoint"):
        renderer.render_runtime_probe_job(
            template_path=TEMPLATE,
            output_dir=tmp_path / "output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=SOURCE_TREE_SHA,
            runtime_probe_run_root=str(runtime_root),
        )


def test_renderer_requires_existing_source_and_fresh_runtime_root(tmp_path: Path) -> None:
    source = tmp_path / "missing-source"
    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir()
    runtime_root = runtime_parent / EXECUTION_SHA
    with pytest.raises(FileNotFoundError, match="source_snapshot_host directory is absent"):
        renderer.render_runtime_probe_job(
            template_path=TEMPLATE,
            output_dir=tmp_path / "missing-source-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=SOURCE_TREE_SHA,
            runtime_probe_run_root=str(runtime_root),
        )
    source.mkdir()
    runtime_root.mkdir()
    with pytest.raises(FileExistsError, match="runtime_probe_run_root already exists"):
        renderer.render_runtime_probe_job(
            template_path=TEMPLATE,
            output_dir=tmp_path / "reused-runtime-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=SOURCE_TREE_SHA,
            runtime_probe_run_root=str(runtime_root),
        )


def test_renderer_rejects_source_and_parent_symlink_aliases(
    tmp_path: Path, monkeypatch,
) -> None:
    base = tmp_path / "real-base"
    base.mkdir()
    source, source_template, tree_sha, _ = _make_source_snapshot(base, monkeypatch)
    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir()
    runtime_root = runtime_parent / EXECUTION_SHA

    source_alias = tmp_path / "source-alias"
    source_alias.symlink_to(source, target_is_directory=True)
    alias_template = source_alias / renderer.TEMPLATE_RELATIVE.as_posix()
    with pytest.raises(ValueError, match="source_snapshot_host contains a symlink"):
        renderer.render_runtime_probe_job(
            template_path=alias_template,
            output_dir=tmp_path / "source-alias-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source_alias),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_root),
        )

    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(base, target_is_directory=True)
    parent_source = parent_alias / "source-snapshot"
    parent_template = parent_source / renderer.TEMPLATE_RELATIVE.as_posix()
    with pytest.raises(ValueError, match="source_snapshot_host contains a symlink"):
        renderer.render_runtime_probe_job(
            template_path=parent_template,
            output_dir=tmp_path / "parent-alias-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(parent_source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_root),
        )


def test_renderer_rejects_runtime_or_output_parent_symlink_alias(
    tmp_path: Path, monkeypatch,
) -> None:
    source, source_template, tree_sha, _ = _make_source_snapshot(tmp_path, monkeypatch)
    real_runtime_parent = tmp_path / "real-runtime"
    real_runtime_parent.mkdir()
    runtime_alias = tmp_path / "runtime-alias"
    runtime_alias.symlink_to(real_runtime_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="runtime_probe_run_root parent contains a symlink"):
        renderer.render_runtime_probe_job(
            template_path=source_template,
            output_dir=tmp_path / "runtime-alias-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_alias / EXECUTION_SHA),
        )

    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir()
    real_output_parent = tmp_path / "real-output"
    real_output_parent.mkdir()
    output_alias = tmp_path / "output-alias"
    output_alias.symlink_to(real_output_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="output_dir parent contains a symlink"):
        renderer.render_runtime_probe_job(
            template_path=source_template,
            output_dir=output_alias / "render",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_parent / EXECUTION_SHA),
        )


def test_renderer_rejects_missing_runtime_parent_and_template_or_producer_symlink(
    tmp_path: Path, monkeypatch,
) -> None:
    source, source_template, tree_sha, _ = _make_source_snapshot(tmp_path, monkeypatch)
    missing_runtime_root = tmp_path / "missing-runtime-parent" / EXECUTION_SHA
    with pytest.raises(FileNotFoundError, match="runtime_probe_run_root parent directory is absent"):
        renderer.render_runtime_probe_job(
            template_path=source_template,
            output_dir=tmp_path / "missing-runtime-parent-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(missing_runtime_root),
        )

    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir()
    source_template.unlink()
    source_template.symlink_to(TEMPLATE)
    with pytest.raises(ValueError, match="runtime-probe template contains a symlink"):
        renderer.render_runtime_probe_job(
            template_path=source_template,
            output_dir=tmp_path / "template-symlink-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_parent / EXECUTION_SHA),
        )

    second_base = tmp_path / "producer-case"
    second_base.mkdir()
    source2, template2, tree2, _ = _make_source_snapshot(second_base, monkeypatch)
    producer2 = source2 / renderer.PRODUCER_RELATIVE.as_posix()
    producer2.unlink()
    producer2.symlink_to(RENDERER_SOURCE)
    runtime_parent2 = second_base / "runtime"
    runtime_parent2.mkdir()
    with pytest.raises(ValueError, match="renderer producer contains a symlink"):
        renderer.render_runtime_probe_job(
            template_path=template2,
            output_dir=second_base / "producer-symlink-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source2),
            source_snapshot_tree_sha=tree2,
            runtime_probe_run_root=str(runtime_parent2 / EXECUTION_SHA),
        )


def test_renderer_rejects_output_dir_containment_pollution(
    tmp_path: Path, monkeypatch,
) -> None:
    source, source_template, tree_sha, _ = _make_source_snapshot(tmp_path, monkeypatch)
    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir()
    runtime_root = runtime_parent / EXECUTION_SHA
    polluted_outputs = (
        source,
        source / "rendered",
        runtime_parent,
        runtime_root,
        runtime_root / "rendered",
        tmp_path,
    )
    for index, output in enumerate(polluted_outputs):
        with pytest.raises(ValueError, match="output_dir must be disjoint"):
            renderer.render_runtime_probe_job(
                template_path=source_template,
                output_dir=output,
                execution_sha=EXECUTION_SHA,
                image_digest=IMAGE_DIGEST,
                source_snapshot_host=str(source),
                source_snapshot_tree_sha=tree_sha,
                runtime_probe_run_root=str(runtime_root),
            )


def test_renderer_rejects_template_receipt_and_producer_tamper(
    tmp_path: Path, monkeypatch,
) -> None:
    source, source_template, tree_sha, receipt = _make_source_snapshot(
        tmp_path, monkeypatch
    )
    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir()
    runtime_root = runtime_parent / EXECUTION_SHA

    monkeypatch.setattr(renderer, "__file__", str(RENDERER_SOURCE))
    with pytest.raises(ValueError, match="exact source snapshot producer"):
        renderer.render_runtime_probe_job(
            template_path=source_template,
            output_dir=tmp_path / "external-producer-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_root),
        )
    monkeypatch.setattr(
        renderer, "__file__", str(source / renderer.PRODUCER_RELATIVE.as_posix())
    )

    original_receipt = receipt.read_bytes()
    receipt.write_bytes(original_receipt + b" ")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        renderer.render_runtime_probe_job(
            template_path=source_template,
            output_dir=tmp_path / "receipt-tamper-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_root),
        )
    receipt.write_bytes(original_receipt)

    with pytest.raises(ValueError, match="verification identity changed|verification differs"):
        renderer.render_runtime_probe_job(
            template_path=source_template,
            output_dir=tmp_path / "wrong-tree-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha="f" * 64,
            runtime_probe_run_root=str(runtime_root),
        )

    source_template.write_bytes(source_template.read_bytes() + b"\n# tamper\n")
    with pytest.raises(ValueError, match="file/mode/bytes differs|mounted-tree attestation differs"):
        renderer.render_runtime_probe_job(
            template_path=source_template,
            output_dir=tmp_path / "template-tamper-output",
            execution_sha=EXECUTION_SHA,
            image_digest=IMAGE_DIGEST,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(runtime_root),
        )


@pytest.mark.parametrize(
    "execution_sha,image_digest,tree_sha,match",
    [
        ("A" * 40, IMAGE_DIGEST, SOURCE_TREE_SHA, "execution_sha"),
        (EXECUTION_SHA, "fpct:latest", SOURCE_TREE_SHA, "image_digest"),
        (EXECUTION_SHA, "sha256:" + "2" * 64, SOURCE_TREE_SHA, "image_digest"),
        (
            EXECUTION_SHA,
            "registry.example/fpct:latest@sha256:" + "2" * 64,
            SOURCE_TREE_SHA,
            "image_digest",
        ),
        (EXECUTION_SHA, IMAGE_DIGEST, "3" * 63, "source_snapshot_tree_sha"),
    ],
)
def test_renderer_rejects_unfrozen_identities(
    tmp_path: Path,
    execution_sha: str,
    image_digest: str,
    tree_sha: str,
    match: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(ValueError, match=match):
        renderer.render_runtime_probe_job(
            template_path=TEMPLATE,
            output_dir=tmp_path / "output",
            execution_sha=execution_sha,
            image_digest=image_digest,
            source_snapshot_host=str(source),
            source_snapshot_tree_sha=tree_sha,
            runtime_probe_run_root=str(tmp_path / "runtime" / EXECUTION_SHA),
        )


def test_renderer_manifest_is_valid_json_and_records_exact_template(
    tmp_path: Path, monkeypatch,
) -> None:
    result, _, source_template, _, _, _, output = _render(tmp_path, monkeypatch)
    observed = json.loads((output / renderer.MANIFEST_FILENAME).read_text())
    assert observed == result
    assert observed["source_template"]["path"] == str(source_template)
    assert observed["source_template"]["sha256"] == renderer.sha256_file(source_template)
    assert observed["source_template"]["bytes"] == source_template.stat().st_size


def test_renderer_cli_writes_the_same_manifest(tmp_path: Path, monkeypatch, capsys) -> None:
    source, source_template, tree_sha, _ = _make_source_snapshot(tmp_path, monkeypatch)
    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir()
    runtime_root = runtime_parent / EXECUTION_SHA
    output = tmp_path / "cli-render"
    assert renderer.main(
        [
            "--template", str(source_template),
            "--output-dir", str(output),
            "--execution-sha", EXECUTION_SHA,
            "--image-digest", IMAGE_DIGEST,
            "--source-snapshot-host", str(source),
            "--source-snapshot-tree-sha", tree_sha,
            "--runtime-probe-run-root", str(runtime_root),
        ]
    ) == 0
    printed = json.loads(capsys.readouterr().out)
    stored = json.loads((output / renderer.MANIFEST_FILENAME).read_text())
    assert printed == stored


def test_fresh_subprocess_uses_only_snapshot_modules_and_writes_no_bytecode(
    tmp_path: Path, monkeypatch,
) -> None:
    source, source_template, tree_sha, receipt = _make_source_snapshot(
        tmp_path, monkeypatch
    )
    producer = source / renderer.PRODUCER_RELATIVE.as_posix()
    conflict = tmp_path / "conflicting-pythonpath"
    conflict_module = conflict / "script/experiment/fpct_e1_runtime_probe.py"
    conflict_lock = conflict / "script/experiment/fpct_e1_source_snapshot_lock.py"
    conflict_module.parent.mkdir(parents=True)
    (conflict / "script/__init__.py").write_text("# hostile regular package\n")
    (conflict / "script/experiment/__init__.py").write_text("# hostile package\n")
    marker = tmp_path / "hostile-imported"
    hostile = f"from pathlib import Path\nPath({str(marker)!r}).write_text('imported')\nraise RuntimeError('hostile import')\n"
    conflict_module.write_text(hostile)
    conflict_lock.write_text(hostile)
    runtime_parent = tmp_path / "runtime"
    runtime_parent.mkdir()
    runtime_root = runtime_parent / EXECUTION_SHA
    render_output = tmp_path / "fresh-render"
    before = _snapshot_bytes_state(source)
    receipt_before = hashlib.sha256(receipt.read_bytes()).hexdigest()
    environment = dict(os.environ)
    environment.pop("PYTHONDONTWRITEBYTECODE", None)
    environment["PYTHONPATH"] = str(conflict)
    process = subprocess.run(
        [
            sys.executable,
            str(producer),
            "--template", str(source_template),
            "--output-dir", str(render_output),
            "--execution-sha", EXECUTION_SHA,
            "--image-digest", IMAGE_DIGEST,
            "--source-snapshot-host", str(source),
            "--source-snapshot-tree-sha", tree_sha,
            "--runtime-probe-run-root", str(runtime_root),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    assert marker.exists() is False
    assert _snapshot_bytes_state(source) == before
    assert hashlib.sha256(receipt.read_bytes()).hexdigest() == receipt_before
    assert list(source.rglob("__pycache__")) == []
    assert list(source.rglob("*.pyc")) == []
    rendered = json.loads(
        (render_output / renderer.MANIFEST_FILENAME).read_text()
    )
    assert json.loads(process.stdout) == rendered
    assert rendered["local_import_provenance"] == {
        "runtime_probe": str(source / renderer.RUNTIME_PROBE_RELATIVE.as_posix()),
        "source_snapshot_lock": str(source / renderer.SOURCE_LOCK_RELATIVE.as_posix()),
    }
    assert rendered["producer"]["path"] == str(producer)
