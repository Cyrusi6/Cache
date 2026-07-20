from __future__ import annotations

import json
from pathlib import Path
import yaml

from script.experiment.fpct_gpu_r2_k8s import render


def test_r2_job_render_is_run_lock_bound(tmp_path: Path) -> None:
    template = tmp_path / "job.yaml"
    template.write_text(
        "name: __JOB_NAME__\nimage: __IMAGE_DIGEST__\nseed: __SEED__\n"
        "run: __RUN_UID__\nnode: __NODE_NAME__\ncm: __RUN_LOCK_CONFIGMAP__\n"
        "labels:\n  git_sha: \"__GIT_SHA_SHORT__\"\n"
    )
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({
        "status": "PRE_OUTPUT_LOCKED_R2",
        "run_uid": "r2-test",
        "scientific_code_commit": "1" * 40,
        "image": {"reference": "example@sha256:" + "2" * 64},
        "kubernetes": {"config_map": "r2-lock", "node_name": "gpu-node"},
    }))
    output = tmp_path / "rendered.yaml"
    result = render(template, lock, output, job_name="r2-seed-45", seed=45)
    text = output.read_text()
    assert "__" not in text
    assert "r2-seed-45" in text
    assert "seed: 45" in text
    assert yaml.safe_load(text)["labels"]["git_sha"] == "11111111"
    assert result["output_sha256"]
