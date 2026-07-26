from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = REPO_ROOT / "recipe/k8s/fpct_e1/runtime_probe_job.yaml"
REQUIRED_PLACEHOLDERS = {
    "__EXECUTION_SHA__",
    "__IMMUTABLE_IMAGE_DIGEST__",
    "__SOURCE_SNAPSHOT_HOST__",
    "__SOURCE_SNAPSHOT_TREE_SHA__",
    "__RUNTIME_PROBE_OUTPUT_PATH__",
}


def _reject_unrendered_apply(text: str) -> None:
    unresolved = set(re.findall(r"__[A-Z0-9_]+__", text))
    if unresolved:
        raise ValueError(f"render-only K8s template has unresolved placeholders: {sorted(unresolved)}")


def test_runtime_probe_job_is_render_only_offline_read_only_and_one_gpu() -> None:
    text = TEMPLATE.read_text(encoding="utf-8")
    assert set(re.findall(r"__[A-Z0-9_]+__", text)) == REQUIRED_PLACEHOLDERS
    with pytest.raises(ValueError, match="unresolved placeholders"):
        _reject_unrendered_apply(text)

    manifest = yaml.safe_load(text)
    assert manifest["metadata"]["annotations"]["fpct.openai.com/render-only-source"] == "true"
    pod = manifest["spec"]["template"]["spec"]
    assert pod["nodeName"] == "4090-48gx2"
    container = pod["containers"][0]
    assert container["image"] == "__IMMUTABLE_IMAGE_DIGEST__"
    assert container["resources"]["limits"]["nvidia.com/gpu"] == "1"
    assert container["resources"]["requests"]["nvidia.com/gpu"] == "1"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert container["command"][1] == "/opt/fpct/script/experiment/fpct_e1_runtime_probe.py"
    assert REQUIRED_PLACEHOLDERS - {"__SOURCE_SNAPSHOT_HOST__"} <= set(container["args"]) | {container["image"]}
    environment = {row["name"]: row["value"] for row in container["env"]}
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"
    mounts = {row["name"]: row for row in container["volumeMounts"]}
    assert mounts["source-snapshot"] == {
        "name": "source-snapshot", "mountPath": "/opt/fpct", "readOnly": True
    }
    assert mounts["e1-results"]["mountPath"] == "/fpct-e1"
    volumes = {row["name"]: row for row in pod["volumes"]}
    assert volumes["source-snapshot"]["hostPath"]["path"] == "__SOURCE_SNAPSHOT_HOST__"
    assert volumes["e1-results"]["hostPath"]["path"] == "/netdisk/lijunsi/fpct-e1"
