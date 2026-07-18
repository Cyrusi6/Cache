"""Generate and run Phase 1.5 same-checkpoint evaluation interventions.

Only six non-native conditions are materialized. Native B2/B3/B6 predictions are
referenced from the completed Phase-1 analysis manifest and are never re-evaluated.
"""

from __future__ import annotations

import argparse
import copy
import glob
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rosetta.utils.eval_interventions import apply_eval_intervention_to_config


DATASETS = ("ai2-arc", "openbookqa", "mmlu-redux")
GPU_LAYOUT = {"ai2-arc": [0], "openbookqa": [1], "mmlu-redux": [0, 1]}
PAIRS = ("tinyllama", "qwen3_1p7b", "qwen25_0p5b", "llama32_1b")
SEEDS = (42, 43, 44)
DEFAULT_CONSTANT_CONFIDENCE = 0.93

INTERVENTIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "b2_eval_k4",
        "trained_variant": "b2",
        "native_variant": "b2",
        "ambiguity_variant": "b3",
        "override": {"top_k": 4},
        "contrast": "B2 train-k1: eval-k4 minus native eval-k1",
    },
    {
        "name": "b3_eval_k1",
        "trained_variant": "b3",
        "native_variant": "b3",
        "ambiguity_variant": "b3",
        "override": {"top_k": 1},
        "contrast": "B3 train-k4: eval-k1 minus native eval-k4",
    },
    {
        "name": "b6_entropy_constant",
        "trained_variant": "b6",
        "native_variant": "b6",
        "ambiguity_variant": "b6",
        "override": {
            "entropy_mode": "constant",
            "entropy_constant_value": DEFAULT_CONSTANT_CONFIDENCE,
            "gate_mode": "learned",
        },
        "contrast": "B6 constant confidence minus native entropy",
    },
    {
        "name": "b6_entropy_shuffled",
        "trained_variant": "b6",
        "native_variant": "b6",
        "ambiguity_variant": "b6",
        "override": {"entropy_mode": "shuffled", "gate_mode": "learned"},
        "contrast": "B6 shuffled entropy minus native entropy",
    },
    {
        "name": "b6_gate_static",
        "trained_variant": "b6",
        "native_variant": "b6",
        "ambiguity_variant": "b6",
        "override": {"entropy_mode": "native", "gate_mode": "static"},
        "contrast": "B6 static entropy scalar minus learned token/head gate",
    },
    {
        "name": "b6_gate_forced_on",
        "trained_variant": "b6",
        "native_variant": "b6",
        "ambiguity_variant": "b6",
        "override": {"entropy_mode": "native", "gate_mode": "forced_on"},
        "contrast": (
            "B6 forced-on alignment-confidence and legacy scalar K/V gates "
            "minus fully checkpoint-native learned gating"
        ),
    },
)

ANOMALY_INTERVENTIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "b6_gate_alignment_forced_on",
        "trained_variant": "b6",
        "native_variant": "b6",
        "ambiguity_variant": "b6",
        "override": {
            "entropy_mode": "native",
            "gate_mode": "alignment_forced_on",
        },
        "contrast": (
            "Qwen2.5 B6 seed44 alignment-confidence forced on with legacy "
            "scalar K/V gates checkpoint-native"
        ),
    },
    {
        "name": "b6_gate_legacy_forced_on",
        "trained_variant": "b6",
        "native_variant": "b6",
        "ambiguity_variant": "b6",
        "override": {
            "entropy_mode": "native",
            "gate_mode": "legacy_forced_on",
        },
        "contrast": (
            "Qwen2.5 B6 seed44 legacy scalar K/V gates forced on with "
            "alignment-confidence checkpoint-learned"
        ),
    },
)


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def _repo_reference(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _resolve_under(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def _resolve_single_csv(root: Path, pattern: str) -> Path:
    resolved_pattern = str(_resolve_under(root, pattern))
    matches = sorted(Path(value).resolve() for value in glob.glob(resolved_pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected exactly one Phase-1 prediction CSV for {resolved_pattern}, "
            f"found {len(matches)}"
        )
    return matches[0]


def _index_runs(manifest: Mapping[str, Any]) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (str(run["pair"]), str(run["variant"]), int(run["seed"])): run
        for run in manifest["runs"]
    }


def _analysis_prediction_csvs(
    analysis_run: Mapping[str, Any], artifact_root: Path
) -> dict[str, str]:
    return {
        dataset: str(
            _resolve_single_csv(
                artifact_root,
                analysis_run["datasets"][dataset]["prediction_glob"],
            )
        )
        for dataset in DATASETS
    }


def _base_eval_config(
    phase1_run: Mapping[str, Any], dataset: str, artifact_root: Path
) -> tuple[Path, dict[str, Any]]:
    config_path = _resolve_under(
        artifact_root, phase1_run["evaluation"]["configs"][dataset]
    )
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    checkpoint = _resolve_under(
        artifact_root, phase1_run["training"]["selected_checkpoint"]
    )
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"selected checkpoint does not exist: {checkpoint}")
    if not any(checkpoint.glob("projector_*.pt")):
        raise FileNotFoundError(
            f"selected checkpoint has no projector_*.pt files: {checkpoint}"
        )
    config["model"]["rosetta_config"]["checkpoints_dir"] = str(checkpoint)
    config["eval"]["gpu_ids"] = GPU_LAYOUT[dataset]
    return config_path, config


def generate_manifest(
    *,
    phase1_manifest_path: Path,
    phase1_analysis_manifest_path: Path,
    phase1_artifact_root: Path,
    output_root: Path,
    results_root: Path,
    recommended_shards: int = 7,
    shard_results_roots: Optional[Mapping[int, Path]] = None,
    intervention_definitions: tuple[dict[str, Any], ...] = INTERVENTIONS,
    pairs: tuple[str, ...] = PAIRS,
    seeds: tuple[int, ...] = SEEDS,
    suite_name: str = "route1_phase1_5_same_checkpoint_interventions",
) -> dict[str, Any]:
    if recommended_shards <= 0:
        raise ValueError("recommended_shards must be positive")
    phase1_manifest = _read_json(phase1_manifest_path.resolve())
    phase1_analysis = _read_json(phase1_analysis_manifest_path.resolve())
    phase1_runs = _index_runs(phase1_manifest)
    analysis_runs = {
        (str(run["pair"]), str(run["variant"]), int(run["seed"])): run
        for run in phase1_analysis["runs"]
    }

    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    results_root = results_root.resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    resolved_shard_roots = {
        int(index): Path(path).resolve()
        for index, path in (shard_results_roots or {}).items()
    }
    invalid_shards = sorted(
        index for index in resolved_shard_roots if not 0 <= index < recommended_shards
    )
    if invalid_shards:
        raise ValueError(f"invalid shard result root indices: {invalid_shards}")
    runs: list[dict[str, Any]] = []
    for pair in pairs:
        for seed in seeds:
            for definition in intervention_definitions:
                variant = definition["trained_variant"]
                source_key = (pair, variant, seed)
                if source_key not in phase1_runs:
                    raise KeyError(f"Phase-1 manifest is missing {source_key}")
                phase1_run = phase1_runs[source_key]
                native_run = analysis_runs[(pair, definition["native_variant"], seed)]
                ambiguity_run = analysis_runs[
                    (pair, definition["ambiguity_variant"], seed)
                ]
                native_csvs = _analysis_prediction_csvs(
                    native_run, phase1_artifact_root
                )
                ambiguity_csvs = _analysis_prediction_csvs(
                    ambiguity_run, phase1_artifact_root
                )
                run_id = f"{pair}__{definition['name']}__seed_{seed}"
                config_paths: dict[str, str] = {}
                output_dirs: dict[str, str] = {}
                base_configs: dict[str, str] = {}
                checkpoint_path: Optional[str] = None
                logical_shard = len(runs) % recommended_shards
                run_results_root = resolved_shard_roots.get(
                    logical_shard, results_root
                )
                for dataset in DATASETS:
                    base_path, config = _base_eval_config(
                        phase1_run, dataset, phase1_artifact_root
                    )
                    base_configs[dataset] = str(base_path)
                    intervention = copy.deepcopy(definition["override"])
                    intervention["id"] = definition["name"]
                    if intervention.get("entropy_mode") == "shuffled":
                        intervention["entropy_shuffle_seed"] = seed
                    config["eval"]["intervention"] = intervention
                    normalized = apply_eval_intervention_to_config(config)
                    assert normalized is not None
                    dataset_output = run_results_root / pair / variant / f"seed_{seed}" / definition["name"] / dataset
                    # Pre-create the complete output tree serially. Concurrent
                    # evaluators on NFS can otherwise race while recursively
                    # creating a shared ancestor even with exist_ok=True.
                    if logical_shard not in resolved_shard_roots:
                        dataset_output.mkdir(parents=True, exist_ok=True)
                    config["output"]["output_dir"] = str(dataset_output)
                    config_path = output_root / "eval" / run_id / f"{dataset}.yaml"
                    config_path.parent.mkdir(parents=True, exist_ok=True)
                    with config_path.open("w", encoding="utf-8") as handle:
                        yaml.safe_dump(config, handle, sort_keys=False)
                    config_paths[dataset] = _repo_reference(config_path)
                    output_dirs[dataset] = str(dataset_output)
                    current_checkpoint = config["model"]["rosetta_config"][
                        "checkpoints_dir"
                    ]
                    if checkpoint_path is None:
                        checkpoint_path = current_checkpoint
                    elif checkpoint_path != current_checkpoint:
                        raise ValueError(f"inconsistent checkpoint in {run_id}")

                command = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "run-triplet",
                    "--arc-config",
                    config_paths["ai2-arc"],
                    "--openbookqa-config",
                    config_paths["openbookqa"],
                    "--mmlu-config",
                    config_paths["mmlu-redux"],
                ]
                runs.append(
                    {
                        "id": run_id,
                        "pair": pair,
                        "seed": seed,
                        "trained_variant": variant,
                        "intervention": normalized,
                        "contrast": definition["contrast"],
                        "checkpoint": {
                            "path": checkpoint_path,
                            "directory_sha256": phase1_run["training"].get(
                                "checkpoint_directory_sha256"
                            ),
                            "same_checkpoint_no_training": True,
                        },
                        "base_eval_configs": base_configs,
                        "eval_configs": config_paths,
                        "output_dirs": output_dirs,
                        "native_comparator": {
                            "run_id": native_run["run_id"],
                            "variant": definition["native_variant"],
                            "prediction_csv": native_csvs,
                        },
                        "ambiguity_source": {
                            "run_id": ambiguity_run["run_id"],
                            "variant": definition["ambiguity_variant"],
                            "fixed_across_contrast": True,
                            "prediction_csv": ambiguity_csvs,
                        },
                        "command": command,
                    }
                )

    expected_run_count = len(pairs) * len(seeds) * len(intervention_definitions)
    if len(runs) != expected_run_count:
        raise AssertionError(
            f"expected {expected_run_count} intervention triplets, got {len(runs)}"
        )
    manifest_path = output_root / "manifest.json"
    shard_commands = []
    for shard_index in range(recommended_shards):
        inner = [
            "python",
            _repo_reference(Path(__file__)),
            "run-shard",
            "--manifest",
            _repo_reference(manifest_path),
            "--shard-index",
            str(shard_index),
            "--num-shards",
            str(recommended_shards),
        ]
        shard_commands.append(
            {
                "shard_index": shard_index,
                "run_count": len(
                    [index for index in range(len(runs)) if index % recommended_shards == shard_index]
                ),
                "gpus": 2,
                "inner_command": inner,
                "scheduler_contract": {
                    "job_name": f"r1-p15-int-s{shard_index + 1:02d}",
                    "nvidia_gpus": 2,
                    "command": inner,
                    "note": (
                        "Submit with the multi-node Kubernetes launcher; "
                        "bash/k8s/gpu_job.sh is hostPath-limited to 4090-24gx4."
                    ),
                },
            }
        )
    manifest = {
        "schema_version": 1,
        "suite": suite_name,
        "source_commit": "0d308525860d27897bde6d558798e468cf113281",
        "constraints": {
            "training_required": False,
            "query_time_transport": False,
            "new_router_or_gate_or_loss": False,
            "native_phase1_reused": True,
        },
        "source": {
            "phase1_manifest": str(phase1_manifest_path.resolve()),
            "phase1_analysis_manifest": str(
                phase1_analysis_manifest_path.resolve()
            ),
            "phase1_artifact_root": str(phase1_artifact_root.resolve()),
        },
        "summary": {
            "new_triplet_count": len(runs),
            "new_dataset_eval_count": len(runs) * len(DATASETS),
            "native_triplet_count_reused": len(
                {
                    (pair, seed, definition["native_variant"])
                    for pair in pairs
                    for seed in seeds
                    for definition in intervention_definitions
                }
            ),
            "pair_count": len(pairs),
            "seed_count": len(seeds),
            "intervention_count": len(intervention_definitions),
        },
        "runs": runs,
        "scheduling": {
            "recommended_shards": recommended_shards,
            "gpu_per_shard": 2,
            "triplet_gpu_schedule": (
                "ARC on GPU 0 and OpenBookQA on GPU 1 concurrently, then "
                "MMLU-Redux on GPUs [0, 1]"
            ),
            "commands": shard_commands,
            "shard_results_roots": {
                str(index): str(path)
                for index, path in sorted(resolved_shard_roots.items())
            },
        },
    }
    _write_json(manifest_path, manifest)
    return manifest


def generate_qwen25_seed44_anomaly_manifest(
    *,
    phase1_manifest_path: Path,
    phase1_analysis_manifest_path: Path,
    phase1_artifact_root: Path,
    output_root: Path,
    results_root: Path,
) -> dict[str, Any]:
    """Generate two optional gate-isolation triplets outside the main 72."""
    return generate_manifest(
        phase1_manifest_path=phase1_manifest_path,
        phase1_analysis_manifest_path=phase1_analysis_manifest_path,
        phase1_artifact_root=phase1_artifact_root,
        output_root=output_root,
        results_root=results_root,
        recommended_shards=1,
        intervention_definitions=ANOMALY_INTERVENTIONS,
        pairs=("qwen25_0p5b",),
        seeds=(44,),
        suite_name="route1_phase1_5_qwen25_seed44_gate_anomaly",
    )


def _triplet_complete(run: Mapping[str, Any]) -> bool:
    for dataset in DATASETS:
        output_dir = _resolve_under(REPO_ROOT, run["output_dirs"][dataset])
        if not _dataset_output_complete(output_dir):
            return False
    return True


def _dataset_output_complete(output_dir: Path) -> bool:
    return (
        len(list(output_dir.glob("*_cot.csv"))) == 1
        and len(list(output_dir.glob("*_summary.json"))) == 1
        and (output_dir / "eval_intervention_provenance.json").is_file()
    )


def run_triplet(
    *, arc_config: Path, openbookqa_config: Path, mmlu_config: Path
) -> int:
    specs = (
        ("ARC", arc_config, "ai2-arc", [0]),
        ("OpenBookQA", openbookqa_config, "openbookqa", [1]),
        ("MMLU-Redux", mmlu_config, "mmlu-redux", [0, 1]),
    )
    completed: dict[str, bool] = {}
    for name, path, dataset, gpu_ids in specs:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        if config.get("eval", {}).get("dataset") != dataset:
            raise ValueError(f"{name} config has the wrong dataset: {path}")
        if config.get("eval", {}).get("gpu_ids") != gpu_ids:
            raise ValueError(
                f"{name} config must use gpu_ids={gpu_ids}, got "
                f"{config.get('eval', {}).get('gpu_ids')}"
            )
        output_dir = _resolve_under(REPO_ROOT, config["output"]["output_dir"])
        completed[name] = _dataset_output_complete(output_dir)
        if completed[name]:
            print(f"[{name}] complete; skipping", flush=True)

    evaluator = REPO_ROOT / "script/evaluation/unified_evaluator.py"

    def command(path: Path) -> list[str]:
        return [sys.executable, str(evaluator), "--config", str(path.resolve())]

    parallel = {}
    if not completed["ARC"]:
        parallel["ARC"] = subprocess.Popen(command(arc_config), cwd=REPO_ROOT)
    if not completed["OpenBookQA"]:
        parallel["OpenBookQA"] = subprocess.Popen(
            command(openbookqa_config), cwd=REPO_ROOT
        )
    pending = set(parallel)
    while pending:
        for name in list(pending):
            return_code = parallel[name].poll()
            if return_code is None:
                continue
            pending.remove(name)
            if return_code != 0:
                for sibling in parallel.values():
                    if sibling.poll() is None:
                        sibling.terminate()
                for sibling in parallel.values():
                    sibling.wait()
                return return_code
        if pending:
            time.sleep(0.2)

    if completed["MMLU-Redux"]:
        return 0
    return subprocess.run(command(mmlu_config), cwd=REPO_ROOT, check=False).returncode


def run_shard(manifest_path: Path, shard_index: int, num_shards: int) -> int:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("require num_shards > 0 and 0 <= shard_index < num_shards")
    manifest = _read_json(manifest_path.resolve())
    selected = [
        run
        for index, run in enumerate(manifest["runs"])
        if index % num_shards == shard_index
    ]
    for run in selected:
        if _triplet_complete(run):
            print(f"[{run['id']}] complete; skipping", flush=True)
            continue
        print(f"[{run['id']}] starting", flush=True)
        configs = run["eval_configs"]
        return_code = run_triplet(
            arc_config=_resolve_under(REPO_ROOT, configs["ai2-arc"]),
            openbookqa_config=_resolve_under(REPO_ROOT, configs["openbookqa"]),
            mmlu_config=_resolve_under(REPO_ROOT, configs["mmlu-redux"]),
        )
        if return_code != 0:
            print(f"[{run['id']}] failed with {return_code}", file=sys.stderr)
            return return_code
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--phase1-manifest", type=Path, required=True)
    generate.add_argument("--phase1-analysis-manifest", type=Path, required=True)
    generate.add_argument("--phase1-artifact-root", type=Path, required=True)
    generate.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "local/tmp/phase1_5_causal_diagnostics",
    )
    generate.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "local/final_results/phase1_5_causal_diagnostics/rev_0d30852"
        ),
    )
    generate.add_argument("--recommended-shards", type=int, default=7)
    generate.add_argument(
        "--shard-results-root",
        action="append",
        default=[],
        metavar="SHARD=PATH",
        help=(
            "Place one logical shard under a node-isolated shared result root; "
            "repeat for multiple shards"
        ),
    )

    anomaly = subparsers.add_parser("generate-anomaly")
    anomaly.add_argument("--phase1-manifest", type=Path, required=True)
    anomaly.add_argument("--phase1-analysis-manifest", type=Path, required=True)
    anomaly.add_argument("--phase1-artifact-root", type=Path, required=True)
    anomaly.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT
        / "local/tmp/phase1_5_causal_diagnostics/qwen25_seed44_gate_anomaly",
    )
    anomaly.add_argument(
        "--results-root",
        type=Path,
        default=Path(
            "local/final_results/phase1_5_causal_diagnostics/"
            "rev_0d30852/qwen25_seed44_gate_anomaly"
        ),
    )

    triplet = subparsers.add_parser("run-triplet")
    triplet.add_argument("--arc-config", type=Path, required=True)
    triplet.add_argument("--openbookqa-config", type=Path, required=True)
    triplet.add_argument("--mmlu-config", type=Path, required=True)

    shard = subparsers.add_parser("run-shard")
    shard.add_argument("--manifest", type=Path, required=True)
    shard.add_argument("--shard-index", type=int, required=True)
    shard.add_argument("--num-shards", type=int, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.command == "generate":
        shard_results_roots = {}
        for raw in args.shard_results_root:
            index_text, separator, path_text = raw.partition("=")
            if not separator or not path_text.strip():
                raise ValueError("--shard-results-root requires SHARD=PATH")
            index = int(index_text)
            if index in shard_results_roots:
                raise ValueError(f"duplicate shard result root: {index}")
            shard_results_roots[index] = Path(path_text)
        manifest = generate_manifest(
            phase1_manifest_path=args.phase1_manifest,
            phase1_analysis_manifest_path=args.phase1_analysis_manifest,
            phase1_artifact_root=args.phase1_artifact_root,
            output_root=args.output_root,
            results_root=args.results_root,
            recommended_shards=args.recommended_shards,
            shard_results_roots=shard_results_roots,
        )
        print(
            f"Generated {manifest['summary']['new_triplet_count']} triplets at "
            f"{args.output_root.resolve() / 'manifest.json'}"
        )
        return 0
    if args.command == "generate-anomaly":
        manifest = generate_qwen25_seed44_anomaly_manifest(
            phase1_manifest_path=args.phase1_manifest,
            phase1_analysis_manifest_path=args.phase1_analysis_manifest,
            phase1_artifact_root=args.phase1_artifact_root,
            output_root=args.output_root,
            results_root=args.results_root,
        )
        print(
            f"Generated {manifest['summary']['new_triplet_count']} anomaly "
            f"triplets at {args.output_root.resolve() / 'manifest.json'}"
        )
        return 0
    if args.command == "run-triplet":
        return run_triplet(
            arc_config=args.arc_config,
            openbookqa_config=args.openbookqa_config,
            mmlu_config=args.mmlu_config,
        )
    return run_shard(args.manifest, args.shard_index, args.num_shards)


if __name__ == "__main__":
    raise SystemExit(main())
