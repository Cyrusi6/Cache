#!/usr/bin/env python3
"""Bootstrap a persistent C2C venv, then exec the requested command."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from typing import Sequence, cast

EXTRAS = "dev,training,evaluation"


def normalize_command(command: Sequence[str]) -> list[str]:
    normalized = list(command)
    if normalized and normalized[0] == "--":
        normalized = normalized[1:]
    if not normalized:
        raise ValueError("没有提供容器内命令")
    return normalized


def runtime_identity(project_root: Path) -> dict[str, str]:
    pyproject = project_root / "pyproject.toml"
    digest = hashlib.sha256()
    digest.update(pyproject.read_bytes())
    digest.update(EXTRAS.encode("utf-8"))
    digest.update(sys.version.encode("utf-8"))
    image = os.environ.get("C2C_RUNTIME_IMAGE", "unknown")
    digest.update(image.encode("utf-8"))
    return {
        "fingerprint": digest.hexdigest(),
        "python": sys.version,
        "image": image,
        "extras": EXTRAS,
    }


def load_marker(path: Path) -> dict[str, str] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        return cast(dict[str, str], data)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_marker(path: Path, data: dict[str, str]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def environment_root(runtime_dir: Path, identity: dict[str, str]) -> Path:
    return runtime_dir / "envs" / identity["fingerprint"]


def bootstrap_runtime(runtime_dir: Path, project_root: Path) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    lock_path = runtime_dir / "bootstrap.lock"
    identity = runtime_identity(project_root)
    target_root = environment_root(runtime_dir, identity)
    marker_path = target_root / "identity.json"
    venv_path = target_root / "venv"

    with lock_path.open("a+") as lock_file:
        print("[c2c-k8s] 等待运行环境锁", flush=True)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        marker = load_marker(marker_path)
        if marker == identity and (venv_path / "bin" / "python").exists():
            print("[c2c-k8s] 复用现有运行环境", flush=True)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return venv_path

        if target_root.exists():
            # identity.json is written last. Without it, this directory was never
            # published to a running command and is safe to retry under the lock.
            print("[c2c-k8s] 清理未发布的失败环境", flush=True)
            shutil.rmtree(target_root)

        target_root.parent.mkdir(parents=True, exist_ok=True)
        target_root.mkdir()
        try:
            print("[c2c-k8s] 创建不可变持久 venv", flush=True)
            venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_path)
            venv_python = venv_path / "bin" / "python"
            print("[c2c-k8s] 安装项目依赖", flush=True)
            subprocess.run(
                [
                    str(venv_python),
                    "-m",
                    "pip",
                    "install",
                    "--disable-pip-version-check",
                    "-e",
                    f"{project_root}[{EXTRAS}]",
                ],
                check=True,
            )
            write_marker(marker_path, identity)
        except Exception:
            shutil.rmtree(target_root, ignore_errors=True)
            raise
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
    return venv_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-dir", type=Path, default=Path("/runtime"))
    parser.add_argument("--project-root", type=Path, default=Path("/workspace/Cache"))
    parser.add_argument("--no-bootstrap", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = normalize_command(args.command)
    project_root = args.project_root.resolve()
    if not project_root.is_dir():
        raise RuntimeError(f"项目目录不存在：{project_root}")

    environment = os.environ.copy()
    home = Path(environment.get("HOME", args.runtime_dir / "home"))
    home.mkdir(parents=True, exist_ok=True)
    (home / ".cache").mkdir(parents=True, exist_ok=True)
    if not args.no_bootstrap:
        venv_path = bootstrap_runtime(args.runtime_dir.resolve(), project_root)
        environment["VIRTUAL_ENV"] = str(venv_path)
        environment["PATH"] = f"{venv_path / 'bin'}:{environment['PATH']}"

    os.chdir(project_root)
    print(f"[c2c-k8s] exec: {command}", flush=True)
    os.execvpe(command[0], command, environment)


if __name__ == "__main__":
    raise SystemExit(main())
