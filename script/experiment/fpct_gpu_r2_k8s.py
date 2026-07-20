from __future__ import annotations

"""Deterministically render run-lock-bound FPCT GPU R2 Kubernetes Jobs."""

import argparse
import hashlib
import json
from pathlib import Path


class R2K8sError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(
    template: Path,
    run_lock: Path,
    output: Path,
    *,
    job_name: str,
    seed: int | None = None,
) -> dict:
    lock = json.loads(run_lock.read_text())
    if lock.get("status") != "PRE_OUTPUT_LOCKED_R2":
        raise R2K8sError("R2 Job rendering requires PRE_OUTPUT_LOCKED_R2")
    scientific_sha = str(lock["scientific_code_commit"])
    replacements = {
        "__JOB_NAME__": job_name,
        "__GIT_SHA_SHORT__": scientific_sha[:8],
        "__RUN_UID__": str(lock["run_uid"]),
        "__IMAGE_DIGEST__": str(lock["image"]["reference"]),
        "__RUN_LOCK_CONFIGMAP__": str(lock["kubernetes"]["config_map"]),
        "__NODE_NAME__": str(lock["kubernetes"]["node_name"]),
    }
    if seed is not None:
        replacements["__SEED__"] = str(seed)
    text = template.read_text()
    for old, new in replacements.items():
        text = text.replace(old, new)
    unresolved = sorted(
        token for token in (
            "__JOB_NAME__", "__GIT_SHA_SHORT__", "__RUN_UID__",
            "__IMAGE_DIGEST__", "__RUN_LOCK_CONFIGMAP__", "__NODE_NAME__",
            "__SEED__",
        ) if token in text
    )
    if unresolved:
        raise R2K8sError(f"unresolved K8s placeholders: {unresolved}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        prior = output.read_text()
        if prior != text:
            raise R2K8sError("refusing to overwrite different submitted YAML")
    else:
        output.write_text(text)
    return {
        "template": str(template.resolve()),
        "template_sha256": sha256_file(template),
        "run_lock": str(run_lock.resolve()),
        "run_lock_sha256": sha256_file(run_lock),
        "output": str(output.resolve()),
        "output_sha256": sha256_file(output),
        "job_name": job_name,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--run-lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    print(json.dumps(render(
        args.template, args.run_lock, args.output,
        job_name=args.job_name, seed=args.seed,
    ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
