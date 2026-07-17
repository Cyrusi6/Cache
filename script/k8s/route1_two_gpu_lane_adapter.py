#!/usr/bin/env python3
"""Run a generated Route-1 lane plan on two GPUs.

This is an experiment-local infrastructure adapter.  It leaves the frozen source
plan and tracked training code untouched, materializes explicit two-process
recipes with the same effective global batch, and records the adapted recipe hash
in the normal checkpoint provenance contract.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SUITE_PATH = REPO_ROOT / "script/analysis/route1_identifiability_suite.py"
SOURCE_PREFIX = Path("local/tmp/route1_identifiability_suite")
ADAPTED_PREFIX = Path("local/tmp/route1_identifiability_suite_2gpu48")
GPU_LAYOUT = {
    "ai2-arc": [0],
    "openbookqa": [1],
    "mmlu-redux": [0, 1],
}


def _load_suite() -> Any:
    spec = importlib.util.spec_from_file_location("route1_ident_suite", SUITE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load suite module from {SUITE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mkdir_nfs(path: Path, attempts: int = 8) -> None:
    for attempt in range(attempts):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except FileExistsError:
            if path.is_dir():
                return
            if attempt + 1 == attempts:
                raise
            time.sleep(0.05 * (attempt + 1))
        else:
            if path.is_dir():
                return
    raise RuntimeError(f"failed to create directory on shared storage: {path}")


def _write_json(path: Path, data: Any) -> None:
    _mkdir_nfs(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_yaml(path: Path, data: Any) -> None:
    _mkdir_nfs(path.parent)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _repo_ref(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def _adapted_path(source: Path) -> Path:
    source = source.resolve()
    relative = source.relative_to(REPO_ROOT.resolve())
    try:
        suffix = relative.relative_to(SOURCE_PREFIX)
    except ValueError as exc:
        raise ValueError(f"source artifact is outside {SOURCE_PREFIX}: {source}") from exc
    return REPO_ROOT / ADAPTED_PREFIX / suffix


def _prepare_output_directories(run: Mapping[str, Any]) -> None:
    training = run["training"]
    if training.get("required"):
        checkpoint = _resolve(training["selected_checkpoint"])
        _mkdir_nfs(checkpoint.parent)
    for output in run["evaluation"]["output_dirs"].values():
        _mkdir_nfs(_resolve(output))
    diagnostics = run.get("gate_diagnostics", {})
    output_dir = diagnostics.get("output_dir")
    if output_dir:
        _mkdir_nfs(_resolve(output_dir))


def materialize(
    source_plan: Path,
    adapted_plan: Path,
    *,
    node_profile: str,
    gpu_memory_gib: int,
    requested_gpus: int = 2,
) -> dict[str, Any]:
    suite = _load_suite()
    source_plan = source_plan.resolve()
    source = json.loads(source_plan.read_text(encoding="utf-8"))
    lane = str(source.get("lane", "")).strip()
    if not lane or source.get("phase") != "phase1":
        raise ValueError("the two-GPU adapter requires a named phase1 lane plan")
    if requested_gpus < 2:
        raise ValueError("requested_gpus must be at least two")

    commit = suite._git_commit_sha(REPO_ROOT)
    adapted = copy.deepcopy(source)
    adapted["hardware"] = {
        "node_profile": node_profile,
        "requested_gpus": int(requested_gpus),
        "used_training_gpus": 2,
        "gpu_memory_gib": int(gpu_memory_gib),
        "placement": "two_gpu_single_node_pod",
        "training_processes": 2,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "effective_global_batch_size": 32,
        "source_plan_sha256": _sha256(source_plan),
        "world_size_note": (
            "2-process DDP is mathematically batch-equivalent to the canonical "
            "4-process run but is not expected to be bitwise identical."
        ),
    }

    for run in adapted["runs"]:
        training = run["training"]
        if training.get("required"):
            source_config = _resolve(training["config"])
            target_config = _adapted_path(source_config)
            config = json.loads(source_config.read_text(encoding="utf-8"))
            train = config["training"]
            original = {
                "num_processes": int(train.get("num_processes", 0)),
                "per_device_train_batch_size": int(
                    train.get("per_device_train_batch_size", 0)
                ),
                "gradient_accumulation_steps": int(
                    train.get("gradient_accumulation_steps", 0)
                ),
            }
            if original != {
                "num_processes": 4,
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 8,
            }:
                raise ValueError(
                    f"unexpected canonical batch profile for {run['run_id']}: {original}"
                )
            source_effective_batch = (
                original["num_processes"]
                * original["per_device_train_batch_size"]
                * original["gradient_accumulation_steps"]
            )
            train["num_processes"] = 2
            train["gradient_accumulation_steps"] = 16
            target_effective_batch = (
                train["num_processes"]
                * int(train["per_device_train_batch_size"])
                * int(train["gradient_accumulation_steps"])
            )
            if source_effective_batch != target_effective_batch or target_effective_batch != 32:
                raise ValueError("effective global batch changed during 2x48 adaptation")
            _write_json(target_config, config)
            training["config"] = _repo_ref(target_config)
            training["num_processes"] = 2
            training["effective_global_batch_size"] = 32
            training["source_config_sha256"] = _sha256(source_config)
            training["checkpoint_provenance"] = suite._checkpoint_provenance_contract(
                run_id=str(run["run_id"]),
                train_config_path=target_config,
                train_config=config,
                repo_root=REPO_ROOT,
                git_commit=commit,
            )

        new_configs: dict[str, str] = {}
        for dataset, source_value in run["evaluation"]["configs"].items():
            source_config = _resolve(source_value)
            target_config = _adapted_path(source_config)
            config = yaml.safe_load(source_config.read_text(encoding="utf-8"))
            config["eval"]["gpu_ids"] = list(GPU_LAYOUT[dataset])
            _write_yaml(target_config, config)
            new_configs[dataset] = _repo_ref(target_config)
        run["evaluation"]["configs"] = new_configs
        run["evaluation"]["gpu_layout"] = copy.deepcopy(GPU_LAYOUT)

        diagnostics = run.get("gate_diagnostics", {})
        command = diagnostics.get("inner_command")
        if command and "--eval-config" in command:
            index = command.index("--eval-config") + 1
            command[index] = new_configs["mmlu-redux"]
        _prepare_output_directories(run)

    state_dir = _resolve(adapted.get("state_dir", adapted_plan.parent / "lane_state"))
    _mkdir_nfs(state_dir)
    _mkdir_nfs(state_dir / "completed")
    _mkdir_nfs(state_dir / "failed")
    _write_json(adapted_plan.resolve(), adapted)
    _write_json(
        adapted_plan.with_suffix(adapted_plan.suffix + ".provenance.json"),
        {
            "schema_version": 1,
            "source_plan": str(source_plan),
            "source_plan_sha256": _sha256(source_plan),
            "adapted_plan": str(adapted_plan.resolve()),
            "adapted_plan_sha256": _sha256(adapted_plan.resolve()),
            "git_commit": commit,
            "lane": adapted["lane"],
            "node_profile": node_profile,
            "gpu_memory_gib": int(gpu_memory_gib),
            "requested_gpus": int(requested_gpus),
            "training_processes": 2,
            "effective_global_batch_size": 32,
            "evaluation_gpu_layout": GPU_LAYOUT,
        },
    )
    return adapted


def _run_two_gpu_triplet(
    *,
    arc_config: Path,
    openbookqa_config: Path,
    mmlu_config: Path,
    python_executable: str,
    child_env: Mapping[str, str] | None = None,
) -> int:
    configs = {
        "ARC": (arc_config.resolve(), [0]),
        "OpenBookQA": (openbookqa_config.resolve(), [1]),
        "MMLU-Redux": (mmlu_config.resolve(), [0, 1]),
    }
    for name, (path, expected) in configs.items():
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        actual = config.get("eval", {}).get("gpu_ids")
        if actual != expected:
            raise ValueError(f"{name} gpu_ids={actual!r}; expected {expected!r}")

    evaluator = REPO_ROOT / "script/evaluation/unified_evaluator.py"
    processes: dict[str, subprocess.Popen[Any]] = {}
    try:
        for name in ("ARC", "OpenBookQA"):
            path = configs[name][0]
            command = [python_executable, str(evaluator), "--config", str(path)]
            print(f"[{name}] starting: {' '.join(command)}", flush=True)
            processes[name] = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=dict(child_env) if child_env is not None else None,
            )
        while processes:
            for name in list(processes):
                return_code = processes[name].poll()
                if return_code is None:
                    continue
                processes.pop(name)
                if return_code != 0:
                    for sibling in processes.values():
                        sibling.terminate()
                    for sibling in processes.values():
                        sibling.wait()
                    return 1
                print(f"[{name}] completed successfully.", flush=True)
            if processes:
                time.sleep(0.2)
    except BaseException:
        for process in processes.values():
            process.terminate()
        for process in processes.values():
            process.wait()
        raise

    mmlu_path = configs["MMLU-Redux"][0]
    command = [python_executable, str(evaluator), "--config", str(mmlu_path)]
    print(f"[MMLU-Redux] starting: {' '.join(command)}", flush=True)
    return int(
        subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=dict(child_env) if child_env is not None else None,
        ).returncode
    )


def _select_startup_gpus(max_used_mib: int) -> tuple[list[str], list[int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows: list[tuple[str, int]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        uuid, used = (part.strip() for part in line.split(",", maxsplit=1))
        rows.append((uuid, int(used)))
    healthy = [(uuid, used) for uuid, used in rows if used <= max_used_mib]
    if len(healthy) < 2:
        raise RuntimeError(
            f"fewer than two allocated GPUs are idle enough: rows={rows}, "
            f"threshold={max_used_mib}"
        )
    selected = healthy[:2]
    selected_uuids = [uuid for uuid, _used in selected]
    selected_used = [used for _uuid, used in selected]
    print(
        f"[2gpu] startup GPU selection passed: visible={len(rows)} "
        f"selected={selected_uuids} used_mib={selected_used}",
        flush=True,
    )
    return selected_uuids, selected_used


def _adapt_two_gpu_command(command: list[str]) -> tuple[list[str], bool]:
    adapted = list(command)
    if "--nproc_per_node=4" not in adapted:
        return adapted, False
    index = adapted.index("--nproc_per_node=4")
    adapted[index] = "--nproc_per_node=2"
    return adapted, True


def run(
    source_plan: Path,
    adapted_plan: Path,
    gate_file: Path,
    state_dir: Path,
    *,
    node_profile: str,
    gpu_memory_gib: int,
    requested_gpus: int,
    max_startup_used_mib: int,
) -> int:
    suite = _load_suite()
    selected_uuids, selected_used_mib = _select_startup_gpus(
        max_startup_used_mib
    )
    child_env = dict(os.environ)
    child_env["CUDA_VISIBLE_DEVICES"] = ",".join(selected_uuids)
    adapted = materialize(
        source_plan,
        adapted_plan,
        node_profile=node_profile,
        gpu_memory_gib=gpu_memory_gib,
        requested_gpus=requested_gpus,
    )
    lane = str(adapted["lane"])
    _write_json(
        adapted_plan.with_suffix(adapted_plan.suffix + ".allocation.json"),
        {
            "schema_version": 1,
            "lane": lane,
            "requested_gpus": int(requested_gpus),
            "selected_gpu_uuids": selected_uuids,
            "selected_gpu_used_mib_at_start": selected_used_mib,
            "cuda_visible_devices": child_env["CUDA_VISIBLE_DEVICES"],
        },
    )

    def run_command(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[Any]:
        command, training_adapted = _adapt_two_gpu_command(command)
        if training_adapted:
            print(
                f"[{lane}/2gpu] adapted training command: {' '.join(command)}",
                flush=True,
            )
        else:
            print(
                f"[{lane}/2gpu] passthrough command: {' '.join(command)}",
                flush=True,
            )
        kwargs.setdefault("env", child_env)
        return subprocess.run(command, **kwargs)

    def triplet_runner(**kwargs: Any) -> int:
        return _run_two_gpu_triplet(**kwargs, child_env=child_env)

    return int(
        suite.run_lane_plan(
            plan_path=adapted_plan.resolve(),
            gate_file=gate_file.resolve(),
            state_dir=state_dir.resolve(),
            reuse_complete=True,
            dependency_timeout_seconds=259200.0,
            dependency_poll_seconds=10.0,
            python_executable=sys.executable,
            run_command=run_command,
            triplet_runner=triplet_runner,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--adapted-plan", type=Path, required=True)
    parser.add_argument("--gate-file", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--node-profile", required=True)
    parser.add_argument("--gpu-memory-gib", type=int, required=True)
    parser.add_argument("--requested-gpus", type=int, default=2)
    parser.add_argument("--max-startup-used-mib", type=int, default=4096)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        materialize(
            args.source_plan,
            args.adapted_plan,
            node_profile=args.node_profile,
            gpu_memory_gib=args.gpu_memory_gib,
            requested_gpus=args.requested_gpus,
        )
        print(f"prepared {args.adapted_plan.resolve()}")
        return 0
    return run(
        args.source_plan,
        args.adapted_plan,
        args.gate_file,
        args.state_dir,
        node_profile=args.node_profile,
        gpu_memory_gib=args.gpu_memory_gib,
        requested_gpus=args.requested_gpus,
        max_startup_used_mib=args.max_startup_used_mib,
    )


if __name__ == "__main__":
    raise SystemExit(main())
