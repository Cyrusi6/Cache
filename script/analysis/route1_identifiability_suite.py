#!/usr/bin/env python3
"""Generate and execute the Route-1 v2.2 identifiability run plan.

The ``generate`` subcommand only writes recipes and machine-readable lane plans;
it does not submit Kubernetes jobs. ``run-lane`` executes one four-GPU lane,
while ``run-triplet`` launches ARC, OpenBookQA, and MMLU-Redux concurrently on
GPUs ``[0]``, ``[1]``, and ``[2, 3]`` respectively.
"""

from __future__ import annotations

import argparse
import copy
import csv
import glob as glob_module
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE = (
    REPO_ROOT / "recipe/train_recipe/identifiability/route1_v22_base.json"
)
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "local/tmp/route1_identifiability_suite"
DEFAULT_STEP1_REUSE_OVERRIDE = (
    REPO_ROOT / "recipe/train_recipe/identifiability/reuse_step1_b6.json"
)
FROZEN_TEMPLATE_SHA256 = (
    "188191f21317c49372c2671ef40739bc8dbeb698e6c495721002b71c250be212"
)
SPLIT_MANIFESTS: dict[int, str] = {
    42: (
        "recipe/train_recipe/identifiability/splits/"
        "mmlu_aux2048_seed42_april_v22.json"
    ),
    43: (
        "recipe/train_recipe/identifiability/splits/"
        "mmlu_aux2048_seed43_seeded.json"
    ),
    44: (
        "recipe/train_recipe/identifiability/splits/"
        "mmlu_aux2048_seed44_seeded.json"
    ),
}

LANE_HARDWARE: dict[str, dict[str, Any]] = {
    "lane_a": {
        "node_profile": "24gx4",
        "requested_gpus": 4,
        "placement": "exclusive_four_gpu_node",
    },
    "lane_b": {
        "node_profile": "24gx8",
        "requested_gpus": 4,
        "placement": "first_four_gpu_pod_on_shared_eight_gpu_node",
        "shared_node_group": "bc_24gx8",
    },
    "lane_c": {
        "node_profile": "24gx8",
        "requested_gpus": 4,
        "placement": "second_four_gpu_pod_on_shared_eight_gpu_node",
        "shared_node_group": "bc_24gx8",
    },
}
LANE_BALANCE_ORDER: tuple[str, ...] = ("lane_b", "lane_c", "lane_a")
LANE_A_AFFINITY_PAIRS: frozenset[str] = frozenset({"llama32_1b"})
CONDITIONAL_PAIR_LANES: dict[str, str] = {
    "llama32_1b": "lane_a",
    "qwen3_1p7b": "lane_b",
    "qwen25_0p5b": "lane_c",
}

REPORT_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("c2c_longest_vs_receiver", "B0", "B1"),
    ("hard_span_vs_receiver", "B0", "B2"),
    ("hard_span_vs_longest", "B1", "B2"),
    ("soft_candidates", "B2", "B3"),
    ("static_entropy", "B3", "B4"),
    ("gate_capacity", "B2-constant", "B5"),
    ("gate_capacity_static_scale_confounded", "B2", "B5"),
    ("full_over_hard_span", "B2", "B6"),
    ("full_over_static_entropy", "B4", "B6"),
    ("full_over_gate_only", "B5", "B6"),
    ("entropy_values", "B6-constant", "B6"),
    ("entropy_position", "B6-shuffle", "B6"),
)

RECEIVER_MODEL = "Qwen/Qwen3-0.6B"
SHARER_PAIRS: dict[str, dict[str, str]] = {
    "tinyllama": {
        "short": "tiny",
        "label": "TinyLlama-1.1B -> Qwen3-0.6B",
        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    },
    "qwen3_1p7b": {
        "short": "q31p7",
        "label": "Qwen3-1.7B -> Qwen3-0.6B",
        "model_id": "Qwen/Qwen3-1.7B",
    },
    "qwen25_0p5b": {
        "short": "q25p5",
        "label": "Qwen2.5-0.5B -> Qwen3-0.6B",
        "model_id": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    "llama32_1b": {
        "short": "l32",
        "label": "Llama-3.2-1B -> Qwen3-0.6B",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
    },
}

SOFT_ALIGNMENT_COMMON: dict[str, Any] = {
    "alignment_strategy": "soft_span_overlap_v2",
    "soft_alignment_score_mode": "uniform",
    "soft_alignment_boundary_bonus": 0.5,
    "soft_alignment_boundary_tolerance": 1,
    "soft_alignment_min_weight": 0.0,
    "soft_alignment_confidence_alpha": 0.5,
    "soft_alignment_confidence_floor": 0.5,
    "soft_alignment_fallback_confidence": 0.25,
}

VARIANTS: dict[str, dict[str, Any]] = {
    "b1": {
        "label": "B1",
        "description": "C2C longest hard remapping",
        "alignment_strategy": "longest",
        "top_k": None,
        "confidence_mode": "none",
        "confidence_control_mode": "native",
        "gate_mode": "none",
    },
    "b2": {
        "label": "B2",
        "description": "hard offset span (soft-span implementation with top-k=1)",
        "alignment_strategy": "soft_span_overlap_v2",
        "top_k": 1,
        "confidence_mode": "none",
        "confidence_control_mode": "native",
        "gate_mode": "none",
    },
    "b2_constant": {
        "label": "B2-constant",
        "description": "top-k=1 with constant 0.93 confidence and no adaptive gate",
        "alignment_strategy": "soft_span_overlap_v2",
        "top_k": 1,
        "confidence_mode": "entropy",
        "confidence_control_mode": "constant",
        "confidence_constant_value": 0.93,
        "gate_mode": "none",
    },
    "b3": {
        "label": "B3",
        "description": "soft uniform span without confidence or adaptive gate",
        "alignment_strategy": "soft_span_overlap_v2",
        "top_k": 4,
        "confidence_mode": "none",
        "confidence_control_mode": "native",
        "gate_mode": "none",
    },
    "b4": {
        "label": "B4",
        "description": "soft uniform span with static entropy confidence",
        "alignment_strategy": "soft_span_overlap_v2",
        "top_k": 4,
        "confidence_mode": "entropy",
        "confidence_control_mode": "native",
        "gate_mode": "none",
    },
    "b5": {
        "label": "B5",
        "description": "top-k=1 with constant confidence and token/head gate",
        "alignment_strategy": "soft_span_overlap_v2",
        "top_k": 1,
        "confidence_mode": "entropy",
        "confidence_control_mode": "constant",
        "confidence_constant_value": 0.93,
        "gate_mode": "token_mlp",
    },
    "b6": {
        "label": "B6",
        "description": "complete v2.2: soft span, entropy, token/head gate",
        "alignment_strategy": "soft_span_overlap_v2",
        "top_k": 4,
        "confidence_mode": "entropy",
        "confidence_control_mode": "native",
        "gate_mode": "token_mlp",
    },
    "b6_constant": {
        "label": "B6-constant",
        "description": "v2.2 with constant 0.93 confidence instead of entropy",
        "alignment_strategy": "soft_span_overlap_v2",
        "top_k": 4,
        "confidence_mode": "entropy",
        "confidence_control_mode": "constant",
        "confidence_constant_value": 0.93,
        "gate_mode": "token_mlp",
    },
    "b6_shuffle": {
        "label": "B6-shuffle",
        "description": "v2.2 with within-sequence shuffled entropy/confidence",
        "alignment_strategy": "soft_span_overlap_v2",
        "top_k": 4,
        "confidence_mode": "entropy",
        "confidence_control_mode": "shuffle",
        "gate_mode": "token_mlp",
    },
}

EVAL_LAYOUT: dict[str, dict[str, Any]] = {
    "ai2-arc": {
        "config_name": "arc",
        "gpu_ids": [0],
        "expected_rows": 1150,
        "length_group": "subjects",
    },
    "openbookqa": {
        "config_name": "openbookqa",
        "gpu_ids": [1],
        "expected_rows": 500,
        "length_group": "subjects",
    },
    "mmlu-redux": {
        "config_name": "mmlu",
        "gpu_ids": [2, 3],
        "expected_rows": 5615,
        "length_group": "subcategories",
    },
}
EXPECTED_PROJECTOR_COUNT = 28
CHECKPOINT_PROVENANCE_FILENAME = "route1_identifiability_provenance.json"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def _ensure_directory(
    path: Path,
    *,
    attempts: int = 8,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> None:
    """Create a directory robustly across concurrent shared-NFS writers."""
    if attempts < 1:
        raise ValueError("attempts must be positive")
    last_error: FileExistsError | None = None
    for attempt in range(attempts):
        try:
            path.mkdir(parents=True, exist_ok=True)
        except FileExistsError as error:
            last_error = error
            if path.is_dir():
                return
            if attempt + 1 < attempts:
                sleep_fn(0.05 * (attempt + 1))
                continue
            raise
        if path.is_dir():
            return
        if attempt + 1 < attempts:
            sleep_fn(0.05 * (attempt + 1))
    if last_error is not None:
        raise last_error
    raise OSError(f"directory was not visible after creation: {path}")


def _write_json(path: Path, data: Any) -> None:
    _ensure_directory(path.parent)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_yaml(path: Path, data: Any) -> None:
    _ensure_directory(path.parent)
    path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _repo_reference(path: Path, repo_root: Path) -> str:
    path = path.resolve()
    try:
        return path.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_sha(repo_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            check=True,
            capture_output=True,
            text=True,
        )
        commit = result.stdout.strip().lower()
    except FileNotFoundError:
        # The stager may run in a minimal image without the git executable. Its
        # pre-staged path already verifies an exact detached HEAD, so reading that
        # value preserves the immutable revision contract.
        try:
            commit = (repo_root / ".git/HEAD").read_text(encoding="utf-8").strip()
        except OSError as exc:
            raise ValueError(
                f"Cannot resolve git commit without git executable: {repo_root}"
            ) from exc
        if commit.startswith("ref:"):
            raise ValueError(
                "git executable is unavailable and the fallback checkout is not "
                f"detached: {commit!r}"
            )
        commit = commit.lower()
    if re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"Invalid git commit from {repo_root}: {commit!r}")
    return commit


def _checkpoint_provenance_contract(
    *,
    run_id: str,
    train_config_path: Path,
    train_config: Mapping[str, Any],
    repo_root: Path,
    git_commit: str,
) -> dict[str, Any]:
    split_value = train_config.get("data", {}).get("split_indices_path")
    if not split_value:
        raise ValueError(f"{run_id} train config is missing split_indices_path")
    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = repo_root / split_path
    split_path = split_path.resolve()
    split_manifest = _read_json(split_path)
    indices_integrity = split_manifest.get("indices_sha256")
    if not isinstance(indices_integrity, Mapping) or not indices_integrity:
        raise ValueError(f"{run_id} split manifest has no indices_sha256 integrity")
    return {
        "schema_version": 1,
        "run_id": run_id,
        "git_commit": git_commit,
        "train_config_sha256": _sha256(train_config_path),
        "split_manifest_sha256": _sha256(split_path),
        "split_indices_sha256": dict(indices_integrity),
        "dataset_canonical_sha256": split_manifest.get("dataset", {}).get(
            "canonical_sha256"
        ),
    }


def _variant_slug(label: str) -> str:
    return label.lower().replace("-", "_")


def _run_id(pair_key: str, variant_key: str, seed: int) -> str:
    return f"{pair_key}__{variant_key}__seed_{seed}"


def _job_base_name(
    pair_key: str,
    variant_key: str,
    seed: int,
    kind: str,
) -> str:
    pair_short = SHARER_PAIRS[pair_key]["short"] if pair_key != "receiver" else "recv"
    variant_short = {
        "b2_constant": "b2c",
        "b6_constant": "b6c",
        "b6_shuffle": "b6s",
    }.get(variant_key, variant_key)
    return f"r1id-{pair_short}-{variant_short}-s{seed}-{kind}"


def _ensure_template_is_v22(template: dict[str, Any], template_path: Path) -> None:
    try:
        model = template["model"]
        training = template["training"]
        data = template["data"]
    except KeyError as error:
        raise ValueError(
            f"Template {template_path} is missing section {error}"
        ) from error

    expected = {
        "base_model": RECEIVER_MODEL,
        "alignment_strategy": "soft_span_overlap_v2",
        "soft_alignment_top_k": 4,
        "soft_alignment_score_mode": "uniform",
        "soft_alignment_confidence_mode": "entropy",
    }
    for key, value in expected.items():
        if model.get(key) != value:
            raise ValueError(
                f"Template {template_path} is not the requested v2.2 baseline: "
                f"model.{key}={model.get(key)!r}, expected {value!r}"
            )
    if data.get("type") != "MMLUChatDataset":
        raise ValueError("The identifiability suite requires MMLUChatDataset")
    kwargs = data.get("kwargs", {})
    if kwargs.get("split") != "auxiliary_train" or kwargs.get("num_samples") != 2048:
        raise ValueError("The identifiability suite requires MMLU auxiliary_train 2048")
    if int(training.get("num_processes", 0)) != 4:
        raise ValueError("The v2.2 template must use four training processes")


def _apply_variant(
    model: dict[str, Any],
    variant_key: str,
    seed: int,
) -> None:
    variant = VARIANTS[variant_key]
    model["is_do_alignment"] = True
    model["alignment_strategy"] = variant["alignment_strategy"]

    projector_params = model["projector"]["params"]
    projector_params["alignment_confidence_gate_mode"] = variant["gate_mode"]
    projector_params["alignment_confidence_max_delta"] = 2.0

    soft_keys = (set(SOFT_ALIGNMENT_COMMON) - {"alignment_strategy"}) | {
        "soft_alignment_top_k",
        "soft_alignment_confidence_mode",
        "soft_alignment_confidence_control_mode",
        "soft_alignment_confidence_constant_value",
        "soft_alignment_confidence_shuffle_seed",
    }
    if variant["alignment_strategy"] == "longest":
        for key in soft_keys:
            model.pop(key, None)
        return

    model.update(SOFT_ALIGNMENT_COMMON)
    model["soft_alignment_top_k"] = variant["top_k"]
    model["soft_alignment_confidence_mode"] = variant["confidence_mode"]
    model["soft_alignment_confidence_control_mode"] = variant["confidence_control_mode"]
    model.pop("soft_alignment_confidence_constant_value", None)
    model.pop("soft_alignment_confidence_shuffle_seed", None)
    if "confidence_constant_value" in variant:
        model["soft_alignment_confidence_constant_value"] = variant[
            "confidence_constant_value"
        ]
    if variant["confidence_control_mode"] == "shuffle":
        model["soft_alignment_confidence_shuffle_seed"] = seed


def _build_train_config(
    template: dict[str, Any],
    pair_key: str,
    variant_key: str,
    seed: int,
    checkpoint_root: str,
) -> dict[str, Any]:
    config = copy.deepcopy(template)
    config["model"]["base_model"] = RECEIVER_MODEL
    config["model"]["teacher_model"] = SHARER_PAIRS[pair_key]["model_id"]
    _apply_variant(config["model"], variant_key, seed)

    config["training"]["seed"] = seed
    config["training"]["num_processes"] = 4
    config["data"]["type"] = "MMLUChatDataset"
    config["data"]["kwargs"]["split"] = "auxiliary_train"
    config["data"]["kwargs"]["num_samples"] = 2048
    config["data"]["split_indices_path"] = SPLIT_MANIFESTS[seed]
    config["data"].pop("split_mode", None)
    config["data"].pop("split_indices_output", None)

    config["output"]["output_dir"] = checkpoint_root
    wandb_config = config["output"].setdefault("wandb_config", {})
    wandb_config["run_name"] = _run_id(pair_key, variant_key, seed)
    return config


def _rosetta_eval_config(
    train_config: dict[str, Any],
    checkpoint_dir: str,
) -> dict[str, Any]:
    train_model = train_config["model"]
    allowed_keys = {
        "base_model",
        "teacher_model",
        "is_do_alignment",
        "alignment_strategy",
        "include_response",
        "multi_source_fusion_mode",
        *SOFT_ALIGNMENT_COMMON.keys(),
        "soft_alignment_top_k",
        "soft_alignment_confidence_mode",
        "soft_alignment_confidence_control_mode",
        "soft_alignment_confidence_constant_value",
        "soft_alignment_confidence_shuffle_seed",
    }
    rosetta_config = {
        key: copy.deepcopy(value)
        for key, value in train_model.items()
        if key in allowed_keys
    }
    rosetta_config["checkpoints_dir"] = checkpoint_dir
    return rosetta_config


def _build_eval_config(
    dataset: str,
    gpu_ids: list[int],
    output_dir: str,
    train_config: dict[str, Any] | None = None,
    checkpoint_dir: str | None = None,
) -> dict[str, Any]:
    if train_config is None:
        model = {
            "model_name": RECEIVER_MODEL,
            "generation_config": {"do_sample": False, "max_new_tokens": 64},
        }
    else:
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is required for Rosetta evaluation")
        model = {
            "model_name": "Rosetta",
            "rosetta_config": _rosetta_eval_config(train_config, checkpoint_dir),
            "generation_config": {"do_sample": False, "max_new_tokens": 64},
        }
    return {
        "model": model,
        "output": {"output_dir": output_dir},
        "eval": {
            "dataset": dataset,
            "gpu_ids": list(gpu_ids),
            "answer_method": "generate",
            "use_cot": False,
            "use_template": True,
            "sample_interval": 1,
            "math_grading_method": "comprehensive",
            "gate_diagnostics": True,
            "gate_diagnostics_mode": "compact",
            "gate_saturation_low_threshold": 0.05,
            "gate_saturation_high_threshold": 0.95,
            "gate_relative_token_bins": 10,
        },
    }


def _requires_posthoc_gate_diagnostics(
    train_config: Mapping[str, Any] | None,
) -> bool:
    if train_config is None:
        return False
    return (
        train_config.get("model", {})
        .get("projector", {})
        .get("params", {})
        .get("alignment_confidence_gate_mode", "none")
        == "token_mlp"
    )


def _k8s_submit_command(
    name: str, inner_command: list[str], *, gpus: int = 4
) -> list[str]:
    return [
        "bash",
        "bash/k8s/gpu_job.sh",
        "submit",
        "--name",
        name,
        "--gpus",
        str(gpus),
        "--",
        *inner_command,
    ]


def _job_record(
    job_id: str,
    kind: str,
    stage: str,
    run_id: str,
    pipeline_lane: str,
    command: list[str],
    depends_on_jobs: Iterable[str] = (),
    conditional: bool = False,
    gpus: int = 4,
) -> dict[str, Any]:
    return {
        "id": job_id,
        "kind": kind,
        "stage": stage,
        "run_id": run_id,
        "pipeline_lane": pipeline_lane,
        "conditional": conditional,
        "gpus": int(gpus),
        "depends_on_jobs": list(depends_on_jobs),
        "command": command,
        "shell_command": shlex.join(command),
    }


def _planned_runs() -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    # Stage 1 reproduction, reused as B6 in Stage 2.
    runs.append(
        {
            "pair": "tinyllama",
            "variant": "b6",
            "seed": 42,
            "stage": "stage1_reproduce_b6",
            "reused_in": ["stage2_single_seed_decomposition"],
            "conditional": False,
        }
    )
    runs.append(
        {
            "pair": "receiver",
            "variant": "b0",
            "seed": 42,
            "stage": "stage2_single_seed_decomposition",
            "conditional": False,
        }
    )
    for variant in ("b1", "b2", "b2_constant", "b3", "b4", "b5"):
        runs.append(
            {
                "pair": "tinyllama",
                "variant": variant,
                "seed": 42,
                "stage": "stage2_single_seed_decomposition",
                "conditional": False,
            }
        )
    for variant in ("b6_constant", "b6_shuffle"):
        runs.append(
            {
                "pair": "tinyllama",
                "variant": variant,
                "seed": 42,
                "stage": "stage3_entropy_counterfactuals",
                "conditional": False,
            }
        )
    for seed in (43, 44):
        for variant in ("b2", "b2_constant", "b3", "b4", "b5", "b6"):
            runs.append(
                {
                    "pair": "tinyllama",
                    "variant": variant,
                    "seed": seed,
                    "stage": "stage4_tinyllama_multiseed",
                    "conditional": False,
                }
            )

    for pair_key in ("qwen3_1p7b", "qwen25_0p5b", "llama32_1b"):
        for variant in ("b1", "b2", "b3", "b5", "b6"):
            runs.append(
                {
                    "pair": pair_key,
                    "variant": variant,
                    "seed": 42,
                    "stage": "stage5_cross_pair_seed42",
                    "conditional": False,
                }
            )
        for seed in (43, 44):
            for variant in ("b1", "b2", "b3", "b5", "b6"):
                runs.append(
                    {
                        "pair": pair_key,
                        "variant": variant,
                        "seed": seed,
                        "stage": "stage5_cross_pair_multiseed",
                        "conditional": True,
                    }
                )
    return runs


def _fixed_pipeline_lane(run: Mapping[str, Any]) -> str | None:
    pair = str(run["pair"])
    if pair in LANE_A_AFFINITY_PAIRS:
        return "lane_a"
    if str(run["stage"]) == "stage1_reproduce_b6":
        return "lane_a"
    if bool(run.get("conditional", False)):
        return CONDITIONAL_PAIR_LANES.get(pair)
    return None


def _assign_pipeline_lanes(
    planned_runs: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply pair affinity first, then balance remaining work per phase."""
    phase_loads = {
        False: {lane: 0 for lane in LANE_HARDWARE},
        True: {lane: 0 for lane in LANE_HARDWARE},
    }
    fixed_lanes: list[str | None] = []
    for run in planned_runs:
        conditional = bool(run.get("conditional", False))
        fixed_lane = _fixed_pipeline_lane(run)
        fixed_lanes.append(fixed_lane)
        if fixed_lane is not None:
            phase_loads[conditional][fixed_lane] += 1

    assigned: list[dict[str, Any]] = []
    for run, fixed_lane in zip(planned_runs, fixed_lanes):
        conditional = bool(run.get("conditional", False))
        lane = fixed_lane
        if lane is None:
            loads = phase_loads[conditional]
            lane = min(
                LANE_BALANCE_ORDER,
                key=lambda candidate: (
                    loads[candidate],
                    LANE_BALANCE_ORDER.index(candidate),
                ),
            )
            loads[lane] += 1
        placed = dict(run)
        placed["pipeline_lane"] = lane
        assigned.append(placed)
    return assigned


def _stage_definitions() -> list[dict[str, Any]]:
    return [
        {
            "id": "stage1_reproduce_b6",
            "step": 1,
            "depends_on": [],
            "purpose": "Reproduce TinyLlama B6 seed 42 before spending the full budget.",
            "completion_gate": {
                "type": "manual_metric_check",
                "reference_macro_mean": 0.5082,
                "criterion": "B6 seed 42 is close to the recorded 50.82 macro mean.",
                "on_failure": "stop_all_later_stages_and_investigate_versions",
            },
        },
        {
            "id": "stage2_single_seed_decomposition",
            "step": 2,
            "depends_on": ["stage1_reproduce_b6:gate_passed"],
            "purpose": (
                "TinyLlama seed-42 B0-B6 component decomposition plus the "
                "B2-constant gate control; reuse Stage-1 B6."
            ),
        },
        {
            "id": "stage3_entropy_counterfactuals",
            "step": 3,
            "depends_on": ["stage1_reproduce_b6:gate_passed"],
            "purpose": "Run B6-constant and B6-shuffle on the TinyLlama pair.",
        },
        {
            "id": "stage4_tinyllama_multiseed",
            "step": 4,
            "depends_on": ["stage1_reproduce_b6:gate_passed"],
            "purpose": "Add seeds 43 and 44 for TinyLlama B2/B2-constant/B3-B6.",
        },
        {
            "id": "stage5_cross_pair_seed42",
            "step": 5,
            "depends_on": ["stage1_reproduce_b6:gate_passed"],
            "purpose": "Screen B1/B2/B3/B5/B6 on three additional sharers at seed 42.",
        },
        {
            "id": "stage5_cross_pair_multiseed",
            "step": 6,
            "depends_on": ["stage5_cross_pair_seed42:direction_consistent"],
            "conditional": True,
            "purpose": "Only if seed-42 directions agree, add seeds 43 and 44 cross-pair.",
        },
    ]


def _dependency_map(runs: Sequence[Mapping[str, Any]]) -> dict[str, list[str]]:
    dependencies: dict[str, list[str]] = {}
    previous_by_lane_phase: dict[tuple[str, bool], str] = {}
    for run in runs:
        key = (str(run["pipeline_lane"]), bool(run["conditional"]))
        run_id = str(run["id"])
        previous = previous_by_lane_phase.get(key)
        dependencies[run_id] = [previous] if previous is not None else []
        previous_by_lane_phase[key] = run_id
    return dependencies


def _lane_plan_entry(
    run: Mapping[str, Any],
    dependencies: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    stage = str(run["stage"])
    gate_key = None
    if stage == "stage5_cross_pair_multiseed":
        gate_key = "conditional"
    elif stage != "stage1_reproduce_b6":
        gate_key = "reproduction"
    return {
        "run_id": run["id"],
        "pair": run["pair"],
        "variant": run["variant"],
        "seed": run["seed"],
        "stage": stage,
        "conditional": run["conditional"],
        "gate_key": gate_key,
        "depends_on_runs": list(dependencies[str(run["id"])]),
        "execution_policy": run["execution_policy"],
        "training": {
            "required": run["training"]["required"],
            "config": run["training"].get("config"),
            "selected_checkpoint": run["training"].get("selected_checkpoint"),
            "checkpoint_directory_sha256": run["training"].get(
                "checkpoint_directory_sha256"
            ),
            "checkpoint_provenance": copy.deepcopy(
                run["training"].get("checkpoint_provenance")
            ),
        },
        "evaluation": {
            "configs": copy.deepcopy(run["evaluation"]["configs"]),
            "output_dirs": copy.deepcopy(run["evaluation"]["output_dirs"]),
        },
        "gate_diagnostics": copy.deepcopy(run["gate_diagnostics"]),
    }


def _lane_submit_command(
    lane: str,
    phase: str,
    plan_ref: str,
    gate_ref: str,
    state_ref: str,
) -> list[str]:
    inner = [
        "python",
        "script/analysis/route1_identifiability_suite.py",
        "run-lane",
        "--plan",
        plan_ref,
        "--gate-file",
        gate_ref,
        "--state-dir",
        state_ref,
        "--reuse-complete",
        "--dependency-timeout-seconds",
        "259200",
    ]
    return _k8s_submit_command(f"r1id-{lane}-{phase}", inner)


def _write_lane_plans(
    runs: Sequence[Mapping[str, Any]],
    output_root: Path,
    repo_root: Path,
) -> tuple[dict[str, dict[str, str]], list[dict[str, Any]]]:
    dependencies = _dependency_map(runs)
    state_ref = _repo_reference(output_root / "lane_state", repo_root)
    gate_path = output_root / "gates.json"
    gate_template_path = output_root / "gates.template.json"
    gate_template = {"reproduction": "pending", "conditional": "pending"}
    _write_json(gate_template_path, gate_template)
    if not gate_path.exists():
        _write_json(gate_path, gate_template)
    gate_ref = _repo_reference(gate_path, repo_root)

    plan_refs: dict[str, dict[str, str]] = {}
    commands: list[dict[str, Any]] = []
    for lane in LANE_HARDWARE:
        plan_refs[lane] = {}
        for phase, is_conditional in (("phase1", False), ("conditional", True)):
            selected = [
                run
                for run in runs
                if run["pipeline_lane"] == lane
                and bool(run["conditional"]) is is_conditional
            ]
            plan = {
                "schema_version": 1,
                "suite": "route1_v22_identifiability",
                "lane": lane,
                "phase": phase,
                "hardware": copy.deepcopy(LANE_HARDWARE[lane]),
                "state_dir": state_ref,
                "gate_contract": {
                    "reproduction": "pass|pending|fail",
                    "conditional": "pass|pending|fail",
                },
                "runs": [_lane_plan_entry(run, dependencies) for run in selected],
            }
            plan_path = output_root / "lanes" / f"{lane}.{phase}.json"
            _write_json(plan_path, plan)
            plan_ref = _repo_reference(plan_path, repo_root)
            plan_refs[lane][phase] = plan_ref
            command = _lane_submit_command(
                lane,
                phase,
                plan_ref,
                gate_ref,
                state_ref,
            )
            commands.append(
                {
                    "lane": lane,
                    "phase": phase,
                    "conditional": is_conditional,
                    "plan": plan_ref,
                    "command": command,
                    "shell_command": shlex.join(command),
                }
            )
    return plan_refs, commands


def _current_b6_reuse_override() -> dict[str, Any]:
    return _read_json(DEFAULT_STEP1_REUSE_OVERRIDE)


def generate_suite(
    template_path: Path = DEFAULT_TEMPLATE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    repo_root: Path = REPO_ROOT,
    reuse_overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Generate recipes, eval triplets, manifest, and staged dependency plan."""
    template_path = template_path.resolve()
    output_root = output_root.resolve()
    repo_root = repo_root.resolve()
    if (
        template_path == DEFAULT_TEMPLATE.resolve()
        and _sha256(template_path) != FROZEN_TEMPLATE_SHA256
    ):
        raise ValueError(
            "Frozen v2.2 template hash mismatch; restore "
            f"{DEFAULT_TEMPLATE.relative_to(REPO_ROOT)}"
        )
    template = _read_json(template_path)
    _ensure_template_is_v22(template, template_path)
    reuse_overrides = reuse_overrides or {}
    git_commit = _git_commit_sha(repo_root)
    revision_namespace = f"rev_{git_commit[:12]}"

    runs_out: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    stage_job_ids: dict[str, list[str]] = {
        stage["id"]: [] for stage in _stage_definitions()
    }

    for planned in _assign_pipeline_lanes(_planned_runs()):
        pair_key = planned["pair"]
        variant_key = planned["variant"]
        seed = int(planned["seed"])
        stage = planned["stage"]
        conditional = bool(planned.get("conditional", False))
        run_id = _run_id(pair_key, variant_key, seed)
        override = dict(reuse_overrides.get(run_id, {}))
        execution_policy = str(override.get("mode", "run_or_reuse_complete"))
        if execution_policy not in {
            "run_or_reuse_complete",
            "reuse_required",
            "checkpoint_reuse_required",
        }:
            raise ValueError(
                f"Unsupported reuse mode for {run_id}: {execution_policy!r}"
            )
        if variant_key == "b0" and execution_policy != "run_or_reuse_complete":
            raise ValueError("B0 reuse is inferred from completed evaluation artifacts")
        pipeline_lane = str(planned["pipeline_lane"])
        run: dict[str, Any] = {
            "id": run_id,
            "pair": pair_key,
            "variant": variant_key,
            "seed": seed,
            "stage": stage,
            "reused_in": planned.get("reused_in", []),
            "conditional": conditional,
            "pipeline_lane": pipeline_lane,
            "execution_policy": execution_policy,
        }

        train_job_id: str | None = None
        train_config: dict[str, Any] | None = None
        if variant_key != "b0":
            checkpoint_root = (
                f"local/checkpoints/route1_identifiability/{revision_namespace}/"
                f"{pair_key}/"
                f"{_variant_slug(VARIANTS[variant_key]['label'])}/seed_{seed}"
            )
            default_final_checkpoint = f"{checkpoint_root}/final"
            final_checkpoint = str(
                override.get("checkpoint_dir", default_final_checkpoint)
            )
            train_config = _build_train_config(
                template,
                pair_key,
                variant_key,
                seed,
                checkpoint_root,
            )
            train_path = (
                output_root / "train" / pair_key / variant_key / f"seed_{seed}.json"
            )
            _write_json(train_path, train_config)
            checkpoint_provenance = _checkpoint_provenance_contract(
                run_id=run_id,
                train_config_path=train_path,
                train_config=train_config,
                repo_root=repo_root,
                git_commit=git_commit,
            )
            train_ref = _repo_reference(train_path, repo_root)
            train_inner = [
                "python",
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node=4",
                "script/train/SFT_train.py",
                "--config",
                train_ref,
            ]
            train_command = _k8s_submit_command(
                _job_base_name(pair_key, variant_key, seed, "train"),
                train_inner,
            )
            train_job_id = f"train::{run_id}"
            jobs.append(
                _job_record(
                    train_job_id,
                    "train",
                    stage,
                    run_id,
                    pipeline_lane,
                    train_command,
                    conditional=conditional,
                )
            )
            stage_job_ids[stage].append(train_job_id)
            run["training"] = {
                "required": True,
                "config": train_ref,
                "checkpoint_dir": checkpoint_root,
                "selected_checkpoint": final_checkpoint,
                "num_processes": 4,
                "k8s_job_id": train_job_id,
                "inner_command": train_inner,
                "checkpoint_provenance": checkpoint_provenance,
            }
            expected_checkpoint_sha256 = override.get("checkpoint_directory_sha256")
            if expected_checkpoint_sha256 is not None:
                expected_checkpoint_sha256 = str(expected_checkpoint_sha256).lower()
                if re.fullmatch(r"[0-9a-f]{64}", expected_checkpoint_sha256) is None:
                    raise ValueError(
                        f"checkpoint_directory_sha256 for {run_id} must be 64 hex chars"
                    )
                run["training"]["checkpoint_directory_sha256"] = (
                    expected_checkpoint_sha256
                )
            if override:
                run["training"]["reuse_override"] = copy.deepcopy(override)
        else:
            final_checkpoint = None
            run["training"] = {"required": False}

        eval_paths: dict[str, str] = {}
        eval_output_dirs: dict[str, str] = {}
        override_output_dirs = override.get("evaluation_output_dirs", {})
        if not isinstance(override_output_dirs, Mapping):
            raise ValueError(f"evaluation_output_dirs for {run_id} must be an object")
        for dataset, layout in EVAL_LAYOUT.items():
            default_result_root = (
                f"local/final_results/route1_identifiability/{revision_namespace}/"
                f"{pair_key}/"
                f"{variant_key}/seed_{seed}/{dataset}"
            )
            result_root = str(override_output_dirs.get(dataset, default_result_root))
            eval_output_dirs[dataset] = result_root
            eval_config = _build_eval_config(
                dataset=dataset,
                gpu_ids=layout["gpu_ids"],
                output_dir=result_root,
                train_config=train_config,
                checkpoint_dir=final_checkpoint,
            )
            eval_path = output_root / "eval" / run_id / f"{dataset}.yaml"
            _write_yaml(eval_path, eval_config)
            eval_paths[dataset] = _repo_reference(eval_path, repo_root)

        triplet_inner = [
            "python",
            "script/analysis/route1_identifiability_suite.py",
            "run-triplet",
            "--arc-config",
            eval_paths["ai2-arc"],
            "--openbookqa-config",
            eval_paths["openbookqa"],
            "--mmlu-config",
            eval_paths["mmlu-redux"],
        ]
        eval_command = _k8s_submit_command(
            _job_base_name(pair_key, variant_key, seed, "eval"),
            triplet_inner,
        )
        eval_job_id = f"eval::{run_id}"
        eval_dependencies = [train_job_id] if train_job_id else []
        jobs.append(
            _job_record(
                eval_job_id,
                "eval_triplet",
                stage,
                run_id,
                pipeline_lane,
                eval_command,
                depends_on_jobs=(job for job in eval_dependencies if job),
                conditional=conditional,
            )
        )
        stage_job_ids[stage].append(eval_job_id)
        run["evaluation"] = {
            "configs": eval_paths,
            "output_dirs": eval_output_dirs,
            "gpu_layout": {
                dataset: layout["gpu_ids"] for dataset, layout in EVAL_LAYOUT.items()
            },
            "k8s_job_id": eval_job_id,
            "inner_command": triplet_inner,
        }
        gate_diagnostics_required = _requires_posthoc_gate_diagnostics(train_config)
        gate_diagnostics_output_dir = (
            f"local/final_results/route1_identifiability/{revision_namespace}/"
            f"{pair_key}/"
            f"{variant_key}/seed_{seed}/gate_posthoc"
        )
        gate_diagnostics_inner = [
            "python",
            "script/analysis/route1_confidence_gate_diagnostics.py",
            "--eval-config",
            eval_paths["mmlu-redux"],
            "--dataset-type",
            "MMLUChatDataset",
            "--dataset-split",
            "auxiliary_train",
            "--num-samples",
            "64",
            "--max-length",
            "1024",
            "--batch-size",
            "1",
            "--device",
            "cuda:0",
            "--output-dir",
            gate_diagnostics_output_dir,
        ]
        diagnostic_job_id = f"gate_diagnostics::{run_id}"
        if gate_diagnostics_required:
            diagnostic_command = _k8s_submit_command(
                _job_base_name(pair_key, variant_key, seed, "gate-diag"),
                gate_diagnostics_inner,
                gpus=1,
            )
            jobs.append(
                _job_record(
                    diagnostic_job_id,
                    "gate_diagnostics",
                    stage,
                    run_id,
                    pipeline_lane,
                    diagnostic_command,
                    depends_on_jobs=(train_job_id,) if train_job_id else (),
                    conditional=conditional,
                    gpus=1,
                )
            )
            stage_job_ids[stage].append(diagnostic_job_id)
        run["gate_diagnostics"] = {
            "required": gate_diagnostics_required,
            "mode": "single_gpu_posthoc_auxiliary64",
            "batch_size": 1,
            "num_samples": 64,
            "output_dir": gate_diagnostics_output_dir,
            "artifact": f"{gate_diagnostics_output_dir}/gate_diagnostics.json",
            "inner_command": gate_diagnostics_inner,
            "k8s_job_id": diagnostic_job_id if gate_diagnostics_required else None,
        }
        runs_out.append(run)

    stages = _stage_definitions()
    for stage in stages:
        stage["job_ids"] = stage_job_ids[stage["id"]]
        stage["train_job_ids"] = [
            job["id"]
            for job in jobs
            if job["stage"] == stage["id"] and job["kind"] == "train"
        ]
        stage["eval_job_ids"] = [
            job["id"]
            for job in jobs
            if job["stage"] == stage["id"] and job["kind"] == "eval_triplet"
        ]
        stage["gate_diagnostic_job_ids"] = [
            job["id"]
            for job in jobs
            if job["stage"] == stage["id"] and job["kind"] == "gate_diagnostics"
        ]
    stage2 = next(
        stage for stage in stages if stage["id"] == "stage2_single_seed_decomposition"
    )
    stage2["reused_job_ids"] = [
        "train::tinyllama__b6__seed_42",
        "eval::tinyllama__b6__seed_42",
    ]
    lane_plan_refs, lane_commands = _write_lane_plans(
        runs_out,
        output_root,
        repo_root,
    )
    reuse_example_path = output_root / "reuse_overrides.step1_b6.json"
    _write_json(reuse_example_path, _current_b6_reuse_override())

    manifest = {
        "schema_version": 1,
        "suite": "route1_v22_identifiability",
        "generator": _repo_reference(Path(__file__), repo_root),
        "git_commit": git_commit,
        "revision_namespace": revision_namespace,
        "template": {
            "path": _repo_reference(template_path, repo_root),
            "sha256": _sha256(template_path),
        },
        "fixed_conditions": {
            "receiver_model": RECEIVER_MODEL,
            "training_dataset": "MMLUChatDataset/auxiliary_train",
            "training_samples": 2048,
            "training_gpus": 4,
            "checkpoint_selection": "final",
            "seeds": [42, 43, 44],
            "evaluation_datasets": list(EVAL_LAYOUT),
            "evaluation_gpu_layout": {
                dataset: layout["gpu_ids"] for dataset, layout in EVAL_LAYOUT.items()
            },
            "training_parameters_except_seed": {
                key: copy.deepcopy(value)
                for key, value in template["training"].items()
                if key != "seed"
            },
        },
        "sharer_pairs": SHARER_PAIRS,
        "variants": {
            "b0": {
                "label": "B0",
                "description": "receiver-only; no cache transfer or training",
            },
            **VARIANTS,
        },
        "runs": runs_out,
        "jobs": jobs,
        "stages": stages,
        "scheduling": {
            "reproduction_parallelism": 1,
            "post_reproduction_parallel_lanes": 3,
            "gpus_per_active_job": 4,
            "lane_rule": (
                "Keep each run's train and eval-triplet jobs on the same lane; "
                "run at most one four-GPU job per lane. All lanes share the same "
                "/netdisk workspace, state, frozen data, model copies, checkpoints, "
                "and results. Llama-3.2 keeps lane_a affinity so conditional pair "
                "work maps cleanly one pair per lane; remaining phase work is balanced."
            ),
            "model_availability_constraints": {
                "llama32_1b": "lane_a",
                "conditional_pair_lanes": copy.deepcopy(CONDITIONAL_PAIR_LANES),
            },
            "physical_layout": (
                "lane_a uses one 24gx4 node. lane_b and lane_c are separate "
                "four-GPU Pods co-located on the same 24gx8 node."
            ),
            "placement_note": (
                "Generated commands request four GPUs but intentionally contain no "
                "node selector; the infrastructure layer owns actual A/B/C placement."
            ),
            "lane_plans": lane_plan_refs,
            "lanes": [
                {
                    "id": lane,
                    "hardware": copy.deepcopy(LANE_HARDWARE[lane]),
                    "assigned_run_ids": [
                        run["id"] for run in runs_out if run["pipeline_lane"] == lane
                    ],
                }
                for lane in LANE_HARDWARE
            ],
        },
        "summary": {
            "run_count": len(runs_out),
            "train_run_count": sum(
                bool(run["training"]["required"]) for run in runs_out
            ),
            "eval_triplet_count": len(runs_out),
            "posthoc_gate_diagnostic_count": sum(
                bool(run["gate_diagnostics"]["required"]) for run in runs_out
            ),
            "conditional_run_count": sum(bool(run["conditional"]) for run in runs_out),
            "note": (
                "Recommended commands launch lane runners. Each runner serializes "
                "train, eval, then required single-GPU post-hoc diagnostics and "
                "respects gates/dependencies; generation submits nothing."
            ),
        },
        "reuse_override_example": _repo_reference(reuse_example_path, repo_root),
        "tracked_step1_reuse_override": _repo_reference(
            DEFAULT_STEP1_REUSE_OVERRIDE,
            repo_root,
        ),
        "reuse_override_policy": (
            "Never applied by default. A checkpoint-only override may be applied "
            "explicitly after strict checkpoint/hash verification; canonical suite "
            "evaluations are rerun with the current evaluator."
        ),
    }

    manifest_path = output_root / "manifest.json"
    stage_path = output_root / "stage_dependencies.json"
    commands_path = output_root / "recommended_commands.jsonl"
    analysis_path = output_root / "analysis_manifest.json"
    manifest["analysis_manifest"] = _repo_reference(analysis_path, repo_root)
    _write_json(manifest_path, manifest)
    _write_json(
        stage_path,
        {
            "schema_version": 1,
            "suite": manifest["suite"],
            "stages": stages,
        },
    )
    _ensure_directory(commands_path.parent)
    commands_path.write_text(
        "".join(
            json.dumps(command, ensure_ascii=False) + "\n" for command in lane_commands
        ),
        encoding="utf-8",
    )
    analysis_manifest = {
        "schema_version": 1,
        "suite": manifest["suite"],
        "report_contract": {
            "required_pairs": list(SHARER_PAIRS),
            "required_seeds": [42, 43, 44],
            "expected_task_rows": {
                task: int(layout["expected_rows"])
                for task, layout in EVAL_LAYOUT.items()
            },
            "required_methods": ["B2", "B5", "B6"],
        },
        "receiver_baseline_run_id": "receiver__b0__seed_42",
        "artifact_selection": (
            "Resolve exactly one completed file per glob; timestamped evaluator "
            "filenames make the output directory the stable identifier."
        ),
        "expected_prediction_format": "unified_evaluator *_cot.csv",
        "component_comparisons": [
            {"question": "soft_candidates", "candidate": "b3", "control": "b2"},
            {"question": "static_entropy", "candidate": "b4", "control": "b3"},
            {
                "question": "gate_capacity",
                "candidate": "b5",
                "control": "b2_constant",
            },
            {
                "question": "gate_capacity_static_scale_confounded",
                "candidate": "b5",
                "control": "b2",
            },
            {"question": "soft_gate_interaction", "candidate": "b6", "control": "b4"},
            {"question": "gate_soft_interaction", "candidate": "b6", "control": "b5"},
            {
                "question": "entropy_constant_counterfactual",
                "candidate": "b6",
                "control": "b6_constant",
            },
            {
                "question": "entropy_shuffle_counterfactual",
                "candidate": "b6",
                "control": "b6_shuffle",
            },
        ],
        "runs": [
            {
                "run_id": run["id"],
                "pair": run["pair"],
                "variant": run["variant"],
                "seed": run["seed"],
                "conditional": run["conditional"],
                "pipeline_lane": run["pipeline_lane"],
                "posthoc_gate_diagnostics": copy.deepcopy(
                    run["gate_diagnostics"]
                ),
                "datasets": {
                    dataset: {
                        "output_dir": output_dir,
                        "prediction_glob": f"{output_dir}/*_cot.csv",
                        "summary_glob": f"{output_dir}/*_summary.json",
                        "length_glob": f"{output_dir}/*_length.json",
                        "gate_diagnostics_glob": (
                            f"{output_dir}/*_gate_diagnostics.json"
                        ),
                    }
                    for dataset, output_dir in run["evaluation"]["output_dirs"].items()
                },
            }
            for run in runs_out
        ],
    }
    _write_json(analysis_path, analysis_manifest)
    return manifest


def _validate_triplet_config(
    path: Path,
    expected_dataset: str,
    expected_gpu_ids: list[int],
) -> None:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    try:
        eval_config = config["eval"]
    except (TypeError, KeyError) as error:
        raise ValueError(f"Invalid evaluation config: {path}") from error
    if eval_config.get("dataset") != expected_dataset:
        raise ValueError(
            f"{path} uses dataset={eval_config.get('dataset')!r}; "
            f"expected {expected_dataset!r}"
        )
    if eval_config.get("gpu_ids") != expected_gpu_ids:
        raise ValueError(
            f"{path} uses gpu_ids={eval_config.get('gpu_ids')!r}; "
            f"expected {expected_gpu_ids!r}"
        )


def _stop_processes(processes: dict[str, Any]) -> None:
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    for process in processes.values():
        if process.poll() is not None:
            continue
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def run_eval_triplet(
    arc_config: Path,
    openbookqa_config: Path,
    mmlu_config: Path,
    python_executable: str = sys.executable,
    evaluator_path: Path = REPO_ROOT / "script/evaluation/unified_evaluator.py",
    popen_factory: Callable[..., Any] = subprocess.Popen,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Run the three development evaluations concurrently in one four-GPU Pod."""
    specs = [
        ("ARC", arc_config.resolve(), "ai2-arc", [0]),
        ("OpenBookQA", openbookqa_config.resolve(), "openbookqa", [1]),
        ("MMLU-Redux", mmlu_config.resolve(), "mmlu-redux", [2, 3]),
    ]
    for _name, config_path, dataset, gpu_ids in specs:
        _validate_triplet_config(config_path, dataset, gpu_ids)

    processes: dict[str, Any] = {}
    try:
        for name, config_path, _dataset, _gpu_ids in specs:
            command = [
                python_executable,
                str(evaluator_path.resolve()),
                "--config",
                str(config_path),
            ]
            print(f"[{name}] starting: {shlex.join(command)}", flush=True)
            processes[name] = popen_factory(command, cwd=str(REPO_ROOT))

        pending = set(processes)
        while pending:
            for name in list(pending):
                return_code = processes[name].poll()
                if return_code is None:
                    continue
                pending.remove(name)
                if return_code != 0:
                    print(
                        f"[{name}] failed with exit code {return_code}; "
                        "terminating sibling evaluations.",
                        file=sys.stderr,
                        flush=True,
                    )
                    _stop_processes(processes)
                    return 1
                print(f"[{name}] completed successfully.", flush=True)
            if pending:
                sleep_fn(0.2)
    except BaseException:
        _stop_processes(processes)
        raise
    return 0


def _resolve_runtime_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _projector_references(config: Any) -> tuple[int, int, set[int]]:
    """Return terminal layer mappings and referenced projector IDs."""
    terminal_mappings = 0
    reference_count = 0
    projector_ids: set[int] = set()

    def visit(value: Any) -> None:
        nonlocal reference_count, terminal_mappings
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child)
            return
        if not isinstance(value, list):
            return
        is_terminal = bool(value) and all(
            isinstance(pair, list)
            and len(pair) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) for item in pair)
            for pair in value
        )
        if is_terminal:
            terminal_mappings += 1
            reference_count += len(value)
            projector_ids.update(int(pair[1]) for pair in value)
            return
        for child in value:
            visit(child)

    visit(config)
    return terminal_mappings, reference_count, projector_ids


def _checkpoint_directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(path.iterdir(), key=lambda candidate: candidate.name):
        if not item.is_file():
            continue
        digest.update(item.name.encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_complete(
    checkpoint_dir: str | Path | None,
    expected_directory_sha256: str | None = None,
) -> bool:
    if not checkpoint_dir:
        return False
    path = _resolve_runtime_path(checkpoint_dir)
    config_path = path / "projector_config.json"
    if not path.is_dir() or not config_path.is_file() or config_path.stat().st_size <= 0:
        return False

    try:
        projector_config = _read_json(config_path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    terminal_mappings, reference_count, referenced_ids = _projector_references(
        projector_config
    )
    expected_ids = set(range(EXPECTED_PROJECTOR_COUNT))
    if (
        terminal_mappings != EXPECTED_PROJECTOR_COUNT
        or reference_count != EXPECTED_PROJECTOR_COUNT
        or referenced_ids != expected_ids
    ):
        return False

    state_paths = {
        int(item.stem.removeprefix("projector_")): item
        for item in path.glob("projector_[0-9]*.pt")
        if item.is_file() and item.stem.removeprefix("projector_").isdigit()
    }
    projector_config_paths = {
        int(item.stem.removeprefix("projector_")): item
        for item in path.glob("projector_[0-9]*.json")
        if item.is_file() and item.name != "projector_config.json"
        and item.stem.removeprefix("projector_").isdigit()
    }
    if set(state_paths) != expected_ids or set(projector_config_paths) != expected_ids:
        return False

    for projector_id in range(EXPECTED_PROJECTOR_COUNT):
        state_path = state_paths[projector_id]
        projector_path = projector_config_paths[projector_id]
        if state_path.stat().st_size <= 0 or projector_path.stat().st_size <= 0:
            return False
        try:
            projector_metadata = _read_json(projector_path)
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        except Exception:
            return False
        if (
            not isinstance(projector_metadata.get("class"), str)
            or not isinstance(projector_metadata.get("init_args"), Mapping)
            or not isinstance(state, Mapping)
            or not state
            or not all(isinstance(key, str) for key in state)
            or not any(
                isinstance(value, torch.Tensor) and value.numel() > 0
                for value in state.values()
            )
        ):
            return False
        del state
    if expected_directory_sha256 is not None:
        try:
            if _checkpoint_directory_sha256(path) != expected_directory_sha256:
                return False
        except OSError:
            return False
    return True


def _checkpoint_provenance_path(checkpoint_dir: str | Path) -> Path:
    return _resolve_runtime_path(checkpoint_dir) / CHECKPOINT_PROVENANCE_FILENAME


def _checkpoint_provenance_matches(
    checkpoint_dir: str | Path | None,
    expected: Mapping[str, Any] | None,
) -> bool:
    if not checkpoint_dir or not expected:
        return False
    path = _checkpoint_provenance_path(checkpoint_dir)
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    try:
        observed = _read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return observed == dict(expected)


def _checkpoint_reusable(
    checkpoint_dir: str | Path | None,
    expected_directory_sha256: str | None,
    expected_provenance: Mapping[str, Any] | None,
) -> bool:
    if not _checkpoint_complete(checkpoint_dir, expected_directory_sha256):
        return False
    if expected_directory_sha256 is not None:
        return True
    if expected_provenance is None:
        return True
    return _checkpoint_provenance_matches(checkpoint_dir, expected_provenance)


def _write_checkpoint_provenance(
    checkpoint_dir: str | Path,
    provenance: Mapping[str, Any],
) -> None:
    _write_json(_checkpoint_provenance_path(checkpoint_dir), dict(provenance))


def _summary_total_samples(summary: Mapping[str, Any], group: str) -> int | None:
    length_statistics = summary.get("length_statistics")
    if not isinstance(length_statistics, Mapping):
        return None
    rows = length_statistics.get(group)
    if not isinstance(rows, Mapping) or not rows:
        return None
    total = 0
    for value in rows.values():
        if not isinstance(value, Mapping):
            return None
        sample_count = value.get("total_samples")
        if (
            not isinstance(sample_count, int)
            or isinstance(sample_count, bool)
            or sample_count < 0
        ):
            return None
        total += sample_count
    return total


def _prediction_csv_metrics(path: Path) -> tuple[int, float] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"subject", "question_id", "is_correct"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                return None
            seen: set[tuple[str, str]] = set()
            count = 0
            correct = 0
            for row in reader:
                subject = str(row.get("subject", "")).strip()
                question_id = str(row.get("question_id", "")).strip()
                if not subject or not question_id or (subject, question_id) in seen:
                    return None
                seen.add((subject, question_id))
                value = str(row.get("is_correct", "")).strip().lower()
                if value in {"true", "1"}:
                    correct += 1
                elif value not in {"false", "0"}:
                    return None
                count += 1
    except (OSError, csv.Error, UnicodeError):
        return None
    return count, (correct / count if count else math.nan)


def _artifact_attempts(path: Path, suffix: str) -> dict[str, Path]:
    return {
        item.name[: -len(suffix)]: item
        for item in path.glob(f"*{suffix}")
        if item.is_file() and item.name.endswith(suffix)
    }


def _evaluation_attempt_complete(
    dataset: str,
    prediction_path: Path,
    summary_path: Path,
    gate_path: Path,
    gate_required: bool = False,
) -> bool:
    try:
        if prediction_path.stat().st_size <= 0 or summary_path.stat().st_size <= 0:
            return False
        expected_rows = int(EVAL_LAYOUT[dataset]["expected_rows"])
        gate_artifact = _read_json(gate_path)
        gate_status = gate_artifact.get("status")
        if gate_required:
            counts = gate_artifact.get("counts", {})
            if (
                gate_status not in {"ok", "compact"}
                or not isinstance(counts, Mapping)
                or int(counts.get("examples_seen", 0)) != expected_rows
                or int(counts.get("examples_with_gate", 0)) != expected_rows
            ):
                return False
        elif gate_status not in {"ok", "compact", "unavailable"}:
            return False
        metrics = _prediction_csv_metrics(prediction_path)
        if metrics is None:
            return False
        row_count, csv_accuracy = metrics
        if row_count != expected_rows:
            return False
        summary = _read_json(summary_path)
        if summary.get("dataset") != dataset:
            return False
        try:
            summary_accuracy = float(summary["overall_accuracy"])
        except (KeyError, TypeError, ValueError):
            return False
        if (
            not math.isfinite(summary_accuracy)
            or not 0.0 <= summary_accuracy <= 1.0
            or not math.isclose(summary_accuracy, csv_accuracy, abs_tol=1e-12)
        ):
            return False
        summary_rows = _summary_total_samples(
            summary, str(EVAL_LAYOUT[dataset]["length_group"])
        )
        if summary_rows != expected_rows:
            return False
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return True


def _latest_complete_evaluation_attempt(
    dataset: str, output_dir: str | Path, gate_required: bool = False
) -> dict[str, Path] | None:
    path = _resolve_runtime_path(output_dir)
    if not path.is_dir():
        return None
    predictions = _artifact_attempts(path, "_cot.csv")
    summaries = _artifact_attempts(path, "_summary.json")
    gates = _artifact_attempts(path, "_gate_diagnostics.json")
    common_attempts = set(predictions) & set(summaries) & set(gates)

    def order_key(stem: str) -> tuple[str, int, str]:
        timestamp = re.search(r"(\d{8}_\d{6})$", stem)
        return (
            timestamp.group(1) if timestamp is not None else "",
            max(
                predictions[stem].stat().st_mtime_ns,
                summaries[stem].stat().st_mtime_ns,
                gates[stem].stat().st_mtime_ns,
            ),
            stem,
        )

    ordered_attempts = sorted(
        common_attempts,
        key=order_key,
        reverse=True,
    )
    for stem in ordered_attempts:
        if _evaluation_attempt_complete(
            dataset,
            predictions[stem],
            summaries[stem],
            gates[stem],
            gate_required,
        ):
            return {
                "stem": Path(stem),
                "prediction": predictions[stem],
                "summary": summaries[stem],
                "gate_diagnostics": gates[stem],
            }
    return None


def _evaluation_complete(
    output_dirs: Mapping[str, str], gate_required: bool = False
) -> bool:
    if set(output_dirs) != set(EVAL_LAYOUT):
        return False
    for dataset, output_dir in output_dirs.items():
        if (
            _latest_complete_evaluation_attempt(
                dataset, output_dir, gate_required=gate_required
            )
            is None
        ):
            return False
    return True


def _posthoc_gate_diagnostics_complete(diagnostics: Mapping[str, Any]) -> bool:
    if not bool(diagnostics.get("required", False)):
        return True
    artifact_value = diagnostics.get("artifact")
    if not artifact_value:
        return False
    artifact_path = _resolve_runtime_path(str(artifact_value))
    if not artifact_path.is_file() or artifact_path.stat().st_size <= 0:
        return False
    try:
        artifact = _read_json(artifact_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    counts = artifact.get("counts", {})
    metadata = artifact.get("metadata", {})
    if not isinstance(counts, Mapping) or not isinstance(metadata, Mapping):
        return False
    try:
        expected_samples = int(diagnostics.get("num_samples", 0))
        processed_samples = int(metadata.get("processed_samples", 0))
        examples_seen = int(counts.get("examples_seen", 0))
        examples_with_gate = int(counts.get("examples_with_gate", 0))
        gate_projectors = int(counts.get("token_head_gate_projectors", 0))
    except (TypeError, ValueError):
        return False
    return (
        expected_samples > 0
        and artifact.get("status") == "ok"
        and processed_samples == expected_samples
        and examples_seen == expected_samples
        and examples_with_gate == expected_samples
        and gate_projectors > 0
        and len(artifact.get("by_layer", [])) == gate_projectors
        and bool(artifact.get("by_stage"))
        and bool(artifact.get("by_layer_head"))
        and bool(artifact.get("by_relative_token_bin"))
    )


def _completion_marker(state_dir: Path, run_id: str) -> Path:
    return state_dir / "completed" / f"{run_id}.json"


def _failure_marker(state_dir: Path, run_id: str) -> Path:
    return state_dir / "failed" / f"{run_id}.json"


def _write_completion_marker(
    state_dir: Path,
    plan: Mapping[str, Any],
    run: Mapping[str, Any],
) -> None:
    _failure_marker(state_dir, str(run["run_id"])).unlink(missing_ok=True)
    _write_json(
        _completion_marker(state_dir, str(run["run_id"])),
        {
            "status": "complete",
            "lane": plan["lane"],
            "phase": plan["phase"],
            "run_id": run["run_id"],
            "completed_at_unix": time.time(),
        },
    )


def _write_failure_marker(
    state_dir: Path,
    plan: Mapping[str, Any],
    run: Mapping[str, Any],
    reason: str,
) -> None:
    _completion_marker(state_dir, str(run["run_id"])).unlink(missing_ok=True)
    _write_json(
        _failure_marker(state_dir, str(run["run_id"])),
        {
            "status": "failed",
            "lane": plan["lane"],
            "phase": plan["phase"],
            "run_id": run["run_id"],
            "reason": reason,
            "failed_at_unix": time.time(),
        },
    )


def _gate_passed(gate_file: Path | None, gate_key: str | None) -> bool:
    if gate_key is None:
        return True
    if gate_file is None or not gate_file.is_file():
        return False
    data = _read_json(gate_file)
    gates = data.get("gates", data)
    if not isinstance(gates, Mapping):
        return False
    value = gates.get(gate_key)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"pass", "passed", "true", "1", "yes"}


def _wait_for_dependencies(
    state_dir: Path,
    dependencies: Sequence[str],
    timeout_seconds: float,
    poll_seconds: float,
    sleep_fn: Callable[[float], None],
) -> tuple[bool, str]:
    deadline = time.monotonic() + max(0.0, timeout_seconds)
    while True:
        missing = [
            run_id
            for run_id in dependencies
            if not _completion_marker(state_dir, run_id).is_file()
        ]
        if not missing:
            return True, ""
        failed_dependencies = [
            run_id
            for run_id in missing
            if _failure_marker(state_dir, run_id).is_file()
        ]
        if failed_dependencies:
            return False, f"dependency failed: {failed_dependencies}"
        if time.monotonic() >= deadline:
            return False, f"dependency timeout; missing={missing}"
        sleep_fn(max(0.01, poll_seconds))


def _run_return_code(result: Any) -> int:
    if isinstance(result, int):
        return result
    return int(getattr(result, "returncode", 1))


def run_lane_plan(
    plan_path: Path,
    gate_file: Path | None = None,
    state_dir: Path | None = None,
    reuse_complete: bool = False,
    dependency_timeout_seconds: float = 259200.0,
    dependency_poll_seconds: float = 10.0,
    python_executable: str = sys.executable,
    run_command: Callable[..., Any] = subprocess.run,
    triplet_runner: Callable[..., int] = run_eval_triplet,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Execute one lane serially while honoring gates and lane-local dependencies."""
    plan = _read_json(plan_path.resolve())
    if not isinstance(plan.get("runs"), list):
        raise ValueError(f"Lane plan has no runs list: {plan_path}")
    resolved_state_dir = (
        state_dir.resolve()
        if state_dir is not None
        else _resolve_runtime_path(str(plan["state_dir"]))
    )
    _ensure_directory(resolved_state_dir)
    resolved_gate_file = gate_file.resolve() if gate_file is not None else None

    for run in plan["runs"]:
        run_id = str(run["run_id"])
        gate_key = run.get("gate_key")
        if not _gate_passed(resolved_gate_file, gate_key):
            print(
                f"[{plan['lane']}] blocked: gate {gate_key!r} is not pass "
                f"before {run_id}",
                file=sys.stderr,
                flush=True,
            )
            return 3

        checkpoint_dir = run["training"].get("selected_checkpoint")
        expected_checkpoint_sha256 = run["training"].get(
            "checkpoint_directory_sha256"
        )
        expected_checkpoint_provenance = run["training"].get(
            "checkpoint_provenance"
        )
        output_dirs = run["evaluation"]["output_dirs"]
        gate_diagnostics = run.get("gate_diagnostics", {"required": False})
        marker = _completion_marker(resolved_state_dir, run_id)
        marker_complete = False
        if marker.is_file():
            checkpoint_ok = not run["training"]["required"] or _checkpoint_reusable(
                checkpoint_dir,
                expected_checkpoint_sha256,
                expected_checkpoint_provenance,
            )
            marker_complete = (
                checkpoint_ok
                and _evaluation_complete(
                    output_dirs,
                    gate_required=bool(gate_diagnostics.get("required", False)),
                )
                and _posthoc_gate_diagnostics_complete(gate_diagnostics)
            )
            if not marker_complete:
                marker.unlink(missing_ok=True)

        # A lane invocation that explicitly includes this run is a retry. Clear only
        # its stale failure marker; unrelated lanes remain isolated, while true
        # dependants continue to wait for this run's completion marker.
        _failure_marker(resolved_state_dir, run_id).unlink(missing_ok=True)

        dependencies_ok, reason = _wait_for_dependencies(
            resolved_state_dir,
            [str(item) for item in run.get("depends_on_runs", [])],
            dependency_timeout_seconds,
            dependency_poll_seconds,
            sleep_fn,
        )
        if not dependencies_ok:
            _write_failure_marker(resolved_state_dir, plan, run, reason)
            print(f"[{plan['lane']}] {run_id}: {reason}", file=sys.stderr, flush=True)
            return 4

        if marker_complete:
            print(f"[{plan['lane']}] {run_id}: already complete; skip", flush=True)
            continue

        policy = str(run.get("execution_policy", "run_or_reuse_complete"))
        if policy == "reuse_required":
            checkpoint_ok = not run["training"]["required"] or _checkpoint_reusable(
                checkpoint_dir,
                expected_checkpoint_sha256,
                expected_checkpoint_provenance,
            )
            if (
                not checkpoint_ok
                or not _evaluation_complete(
                    output_dirs,
                    gate_required=bool(gate_diagnostics.get("required", False)),
                )
                or not _posthoc_gate_diagnostics_complete(gate_diagnostics)
            ):
                reason = "reuse_required artifacts are incomplete"
                _write_failure_marker(resolved_state_dir, plan, run, reason)
                print(f"[{plan['lane']}] {run_id}: {reason}", file=sys.stderr)
                return 1
            print(f"[{plan['lane']}] {run_id}: verified explicit reuse", flush=True)
            _write_completion_marker(resolved_state_dir, plan, run)
            continue

        checkpoint_reuse_required = policy == "checkpoint_reuse_required"
        if checkpoint_reuse_required and not _checkpoint_reusable(
            checkpoint_dir,
            expected_checkpoint_sha256,
            expected_checkpoint_provenance,
        ):
            reason = "checkpoint_reuse_required checkpoint is incomplete or mismatched"
            _write_failure_marker(resolved_state_dir, plan, run, reason)
            print(f"[{plan['lane']}] {run_id}: {reason}", file=sys.stderr)
            return 1

        trained_now = False
        if run["training"]["required"]:
            if checkpoint_reuse_required:
                print(
                    f"[{plan['lane']}] {run_id}: verified required checkpoint reuse",
                    flush=True,
                )
            elif reuse_complete and _checkpoint_reusable(
                checkpoint_dir,
                expected_checkpoint_sha256,
                expected_checkpoint_provenance,
            ):
                print(
                    f"[{plan['lane']}] {run_id}: reuse complete checkpoint", flush=True
                )
            else:
                train_config = _resolve_runtime_path(run["training"]["config"])
                command = [
                    python_executable,
                    "-m",
                    "torch.distributed.run",
                    "--standalone",
                    "--nproc_per_node=4",
                    "script/train/SFT_train.py",
                    "--config",
                    str(train_config),
                ]
                print(
                    f"[{plan['lane']}] train {run_id}: {shlex.join(command)}",
                    flush=True,
                )
                result = run_command(command, cwd=str(REPO_ROOT))
                return_code = _run_return_code(result)
                if return_code != 0:
                    reason = f"training exited with code {return_code}"
                    _write_failure_marker(resolved_state_dir, plan, run, reason)
                    return return_code if return_code > 0 else 1
                if not _checkpoint_complete(
                    checkpoint_dir, expected_checkpoint_sha256
                ):
                    reason = "training exited successfully but final checkpoint is incomplete"
                    _write_failure_marker(resolved_state_dir, plan, run, reason)
                    print(f"[{plan['lane']}] {run_id}: {reason}", file=sys.stderr)
                    return 1
                if (
                    expected_checkpoint_provenance is not None
                    and expected_checkpoint_sha256 is None
                ):
                    _write_checkpoint_provenance(
                        checkpoint_dir, expected_checkpoint_provenance
                    )
                if not _checkpoint_reusable(
                    checkpoint_dir,
                    expected_checkpoint_sha256,
                    expected_checkpoint_provenance,
                ):
                    reason = "training checkpoint provenance is missing or mismatched"
                    _write_failure_marker(resolved_state_dir, plan, run, reason)
                    print(f"[{plan['lane']}] {run_id}: {reason}", file=sys.stderr)
                    return 1
                trained_now = True

        if reuse_complete and not trained_now and _evaluation_complete(
            output_dirs,
            gate_required=bool(gate_diagnostics.get("required", False)),
        ):
            print(f"[{plan['lane']}] {run_id}: reuse complete evaluation", flush=True)
        else:
            configs = run["evaluation"]["configs"]
            return_code = triplet_runner(
                arc_config=_resolve_runtime_path(configs["ai2-arc"]),
                openbookqa_config=_resolve_runtime_path(configs["openbookqa"]),
                mmlu_config=_resolve_runtime_path(configs["mmlu-redux"]),
                python_executable=python_executable,
            )
            if return_code != 0:
                reason = f"evaluation triplet exited with code {return_code}"
                _write_failure_marker(resolved_state_dir, plan, run, reason)
                return return_code if return_code > 0 else 1
            if not _evaluation_complete(
                output_dirs,
                gate_required=bool(gate_diagnostics.get("required", False)),
            ):
                reason = "evaluation exited successfully but expected artifacts are incomplete"
                _write_failure_marker(resolved_state_dir, plan, run, reason)
                print(f"[{plan['lane']}] {run_id}: {reason}", file=sys.stderr)
                return 1

        if bool(gate_diagnostics.get("required", False)):
            if reuse_complete and not trained_now and _posthoc_gate_diagnostics_complete(
                gate_diagnostics
            ):
                print(
                    f"[{plan['lane']}] {run_id}: reuse complete post-hoc gate diagnostics",
                    flush=True,
                )
            else:
                stored_command = list(gate_diagnostics.get("inner_command", []))
                if not stored_command:
                    reason = "required post-hoc gate diagnostics command is missing"
                    _write_failure_marker(resolved_state_dir, plan, run, reason)
                    print(f"[{plan['lane']}] {run_id}: {reason}", file=sys.stderr)
                    return 1
                stored_command[0] = python_executable
                print(
                    f"[{plan['lane']}] gate diagnostics {run_id}: "
                    f"{shlex.join(stored_command)}",
                    flush=True,
                )
                result = run_command(stored_command, cwd=str(REPO_ROOT))
                return_code = _run_return_code(result)
                if return_code != 0:
                    reason = f"post-hoc gate diagnostics exited with code {return_code}"
                    _write_failure_marker(resolved_state_dir, plan, run, reason)
                    return return_code if return_code > 0 else 1
                if not _posthoc_gate_diagnostics_complete(gate_diagnostics):
                    reason = (
                        "post-hoc gate diagnostics exited successfully but required "
                        "K/V layer/head/token artifact is incomplete"
                    )
                    _write_failure_marker(resolved_state_dir, plan, run, reason)
                    print(f"[{plan['lane']}] {run_id}: {reason}", file=sys.stderr)
                    return 1

        _write_completion_marker(resolved_state_dir, plan, run)
        print(f"[{plan['lane']}] {run_id}: complete", flush=True)
    return 0


def _resolve_prediction_glob(pattern: str) -> list[Path]:
    path = Path(pattern)
    resolved_pattern = str(path if path.is_absolute() else REPO_ROOT / path)
    return sorted(
        (Path(item).resolve() for item in glob_module.glob(resolved_pattern)),
        key=lambda item: (item.stat().st_mtime, item.name),
    )


def _materialized_evaluation_attempt(
    task: str, artifacts: Mapping[str, Any], gate_required: bool = False
) -> dict[str, Path] | None:
    output_dir = artifacts.get("output_dir")
    if output_dir and artifacts.get("gate_diagnostics_glob"):
        return _latest_complete_evaluation_attempt(
            task, str(output_dir), gate_required=gate_required
        )
    # Backwards-compatible path for hand-authored/legacy manifests that predate
    # timestamp-grouped summary and gate artifacts.
    matches = _resolve_prediction_glob(str(artifacts["prediction_glob"]))
    if not matches:
        return None
    return {"prediction": matches[-1]}


def _method_label(variant: str) -> str:
    if variant == "b0":
        return "B0"
    return str(VARIANTS[variant]["label"])


def materialize_analysis_manifest(
    analysis_manifest_path: Path,
    output_path: Path,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Resolve timestamped evaluator globs into the report script's input schema."""
    analysis = _read_json(analysis_manifest_path.resolve())
    receiver_by_task: dict[str, Path] = {}
    missing: list[dict[str, str]] = []

    for run in analysis["runs"]:
        if run["variant"] != "b0":
            continue
        for task, artifacts in run["datasets"].items():
            attempt = _materialized_evaluation_attempt(task, artifacts)
            if attempt is not None:
                receiver_by_task[task] = attempt["prediction"]
            else:
                missing.append({"run_id": run["run_id"], "task": task})

    if set(receiver_by_task) != set(EVAL_LAYOUT) and not allow_missing:
        raise FileNotFoundError(
            f"Receiver B0 predictions are incomplete: missing={missing}"
        )

    report_runs: list[dict[str, Any]] = []
    synthetic_receivers: dict[tuple[str, int, str], dict[str, Any]] = {}
    for run in analysis["runs"]:
        if run["variant"] == "b0":
            continue
        posthoc = run.get("posthoc_gate_diagnostics", {})
        if not isinstance(posthoc, Mapping):
            posthoc = {}
        posthoc_required = bool(posthoc.get("required", False))
        posthoc_complete = _posthoc_gate_diagnostics_complete(posthoc)
        if posthoc_required and not posthoc_complete:
            missing.append(
                {"run_id": run["run_id"], "task": "gate_diagnostics_posthoc"}
            )
            if allow_missing:
                continue
            raise FileNotFoundError(str(posthoc.get("artifact", "")))
        posthoc_artifact = (
            _resolve_runtime_path(str(posthoc["artifact"]))
            if posthoc_required and posthoc_complete
            else None
        )
        for task, artifacts in run["datasets"].items():
            attempt = _materialized_evaluation_attempt(
                task, artifacts, gate_required=posthoc_required
            )
            if attempt is None:
                missing.append({"run_id": run["run_id"], "task": task})
                if allow_missing:
                    continue
                raise FileNotFoundError(artifacts["prediction_glob"])
            receiver_csv = receiver_by_task.get(task)
            if receiver_csv is None:
                if allow_missing:
                    continue
                raise FileNotFoundError(f"Missing B0 receiver CSV for {task}")
            pair = str(run["pair"])
            seed = int(run["seed"])
            synthetic_receivers.setdefault(
                (pair, seed, task),
                {
                    "method": "B0",
                    "pair": pair,
                    "seed": seed,
                    "task": task,
                    "csv": str(receiver_csv),
                },
            )
            report_runs.append(
                {
                    "method": _method_label(str(run["variant"])),
                    "pair": pair,
                    "seed": seed,
                    "task": task,
                    "csv": str(attempt["prediction"]),
                    "receiver_csv": str(receiver_csv),
                    "receiver_method": "B0",
                    "source_run_id": run["run_id"],
                    "gate_diagnostics": (
                        str(attempt["gate_diagnostics"])
                        if "gate_diagnostics" in attempt
                        else None
                    ),
                    "gate_diagnostics_posthoc": (
                        str(posthoc_artifact)
                        if posthoc_artifact is not None
                        else None
                    ),
                }
            )

    report_runs.extend(synthetic_receivers.values())
    report_runs.sort(
        key=lambda row: (row["pair"], row["method"], int(row["seed"]), row["task"])
    )
    conditional_expected = {
        (str(run["run_id"]), str(task))
        for run in analysis["runs"]
        if bool(run.get("conditional", False))
        for task in run["datasets"]
    }
    conditional_materialized = {
        (str(row["source_run_id"]), str(row["task"]))
        for row in report_runs
        if row.get("source_run_id") is not None
    }
    conditional_complete = (
        conditional_expected.issubset(conditional_materialized)
        if conditional_expected
        else bool(analysis.get("conditional_complete", False))
    )
    result = {
        "schema_version": 1,
        "receiver_method": "B0",
        "source_analysis_manifest": str(analysis_manifest_path.resolve()),
        "report_contract": copy.deepcopy(analysis.get("report_contract", {})),
        "conditional_complete": conditional_complete,
        "comparisons": [
            {"name": name, "baseline": baseline, "candidate": candidate}
            for name, baseline, candidate in REPORT_COMPARISONS
        ],
        "runs": report_runs,
        "missing": missing,
    }
    _write_json(output_path.resolve(), result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and run the Route-1 v2.2 identifiability suite"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate recipes and a run plan without submitting any jobs",
    )
    generate_parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    generate_parser.add_argument(
        "--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT
    )
    generate_parser.add_argument(
        "--reuse-override",
        type=Path,
        default=None,
        help="Optional JSON mapping run IDs to reuse_required artifact paths",
    )

    triplet_parser = subparsers.add_parser(
        "run-triplet",
        help="Run ARC/OBQA/MMLU concurrently inside one four-GPU Pod",
    )
    triplet_parser.add_argument("--arc-config", type=Path, required=True)
    triplet_parser.add_argument("--openbookqa-config", type=Path, required=True)
    triplet_parser.add_argument("--mmlu-config", type=Path, required=True)
    triplet_parser.add_argument("--python", default=sys.executable)
    triplet_parser.add_argument(
        "--evaluator",
        type=Path,
        default=REPO_ROOT / "script/evaluation/unified_evaluator.py",
    )

    lane_parser = subparsers.add_parser(
        "run-lane",
        help="Execute one generated lane plan serially on a four-GPU Pod",
    )
    lane_parser.add_argument("--plan", type=Path, required=True)
    lane_parser.add_argument("--gate-file", type=Path, default=None)
    lane_parser.add_argument("--state-dir", type=Path, default=None)
    lane_parser.add_argument("--reuse-complete", action="store_true")
    lane_parser.add_argument(
        "--dependency-timeout-seconds", type=float, default=259200.0
    )
    lane_parser.add_argument("--dependency-poll-seconds", type=float, default=10.0)
    lane_parser.add_argument("--python", default=sys.executable)

    materialize_parser = subparsers.add_parser(
        "materialize-analysis",
        help="Resolve evaluator CSV globs into a report input manifest",
    )
    materialize_parser.add_argument("--analysis-manifest", type=Path, required=True)
    materialize_parser.add_argument("--output", type=Path, default=None)
    materialize_parser.add_argument("--allow-missing", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "generate":
        reuse_overrides: Mapping[str, Mapping[str, Any]] | None = None
        if args.reuse_override is not None:
            raw_overrides = _read_json(args.reuse_override.resolve())
            candidate = raw_overrides.get("runs", raw_overrides)
            if not isinstance(candidate, Mapping):
                raise ValueError("Reuse override must be an object keyed by run ID")
            reuse_overrides = candidate
        manifest = generate_suite(
            args.template,
            args.output_root,
            reuse_overrides=reuse_overrides,
        )
        output_root = args.output_root.resolve()
        print(f"Generated {manifest['summary']['run_count']} runs in {output_root}")
        print(f"Manifest: {output_root / 'manifest.json'}")
        print("No Kubernetes jobs were submitted.")
        return 0
    if args.command == "run-triplet":
        return run_eval_triplet(
            arc_config=args.arc_config,
            openbookqa_config=args.openbookqa_config,
            mmlu_config=args.mmlu_config,
            python_executable=args.python,
            evaluator_path=args.evaluator,
        )
    if args.command == "run-lane":
        return run_lane_plan(
            plan_path=args.plan,
            gate_file=args.gate_file,
            state_dir=args.state_dir,
            reuse_complete=args.reuse_complete,
            dependency_timeout_seconds=args.dependency_timeout_seconds,
            dependency_poll_seconds=args.dependency_poll_seconds,
            python_executable=args.python,
        )
    output_path = args.output
    if output_path is None:
        output_path = (
            args.analysis_manifest.resolve().parent / "report_input_manifest.json"
        )
    result = materialize_analysis_manifest(
        analysis_manifest_path=args.analysis_manifest,
        output_path=output_path,
        allow_missing=args.allow_missing,
    )
    print(f"Materialized {len(result['runs'])} report rows to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
