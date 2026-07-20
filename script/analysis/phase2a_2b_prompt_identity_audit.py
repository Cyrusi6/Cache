#!/usr/bin/env python3
"""CPU-only tokenizer environment and historical prompt-identity audit.

This script reads tokenizer assets, evaluation configs, artifact filenames,
execution manifests, and provenance only.  It never opens prediction CSV
contents, correctness fields, sealed outcomes, or model checkpoints.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import locale as locale_module
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import yaml
from transformers import AutoTokenizer
import transformers.utils.chat_template_utils as chat_template_utils

from rosetta.utils.prompt_identity import (
    audit_tokenizer_paths,
    build_prompt_identity_record,
    prompt_identity_record_sha256,
    sha256_file,
    tokenizer_identity,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_ROOT = Path("/netdisk/lijunsi/c2c-route1-identifiability")
SOURCE_WORKSPACE = SOURCE_ROOT / "workspace/Cache"
MODEL_ROOT = SOURCE_ROOT / "models"
PHASE1_MANIFEST = (
    SOURCE_WORKSPACE / "local/tmp/route1_identifiability_suite/manifest.json"
)
PHASE15_MANIFEST = (
    SOURCE_WORKSPACE / "local/tmp/phase1_5_causal_diagnostics/manifest.json"
)
PHASE2A0_MANIFEST = (
    REPO_ROOT / "recipe/eval_recipe/phase2a_0/opportunity_audit_manifest.json"
)
PHASE2A1_FREEZE = REPO_ROOT / "recipe/eval_recipe/phase2a_1/code_and_design_freeze.json"
PHASE2A2_EXECUTION = (
    REPO_ROOT
    / "recipe/eval_recipe/phase2a_2a_equivalence_debug/execution_manifest.json"
)
PHASE2A2_EQUIVALENCE = (
    REPO_ROOT / "recipe/eval_recipe/phase2a_2a_equivalence_debug/aggregate.json"
)
TIMESTAMP = re.compile(r"_(20[0-9]{6})_[0-9]{6}_cot\.csv$")
CLASSIFICATIONS = {"safe", "ambiguous", "invalid"}


class AuditError(RuntimeError):
    """Prompt identity audit contract violation."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _artifact_date(path: Path) -> str | None:
    match = TIMESTAMP.search(path.name)
    return None if match is None else match.group(1)


def _single_prediction_path(output_dir: Path) -> Path | None:
    candidates = sorted(output_dir.glob("*_cot.csv"))
    return candidates[0] if len(candidates) == 1 else None


def _resolve_workspace_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else SOURCE_WORKSPACE / path


def _prompt_config_view(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    model = config.get("model", {})
    rosetta = model.get("rosetta_config", {})
    evaluation = config.get("eval", {})
    return {
        "model_name": model.get("model_name"),
        "base_model": rosetta.get("base_model"),
        "teacher_model": rosetta.get("teacher_model"),
        "is_do_alignment": rosetta.get("is_do_alignment", model.get("is_do_alignment")),
        "dataset": evaluation.get("dataset"),
        "answer_method": evaluation.get("answer_method"),
        "response_text": evaluation.get("response_text"),
        "use_cot": evaluation.get("use_cot"),
        "use_template": evaluation.get("use_template"),
        "sample_interval": evaluation.get("sample_interval"),
        "limit": evaluation.get("limit"),
        "subjects": evaluation.get("subjects"),
    }


def _entry(**values: Any) -> dict[str, Any]:
    classification = str(values["classification"])
    if classification not in CLASSIFICATIONS:
        raise AuditError(f"invalid classification: {classification}")
    return values


def _scan_models() -> list[dict[str, Any]]:
    model_paths = sorted(
        path
        for path in MODEL_ROOT.iterdir()
        if path.is_dir() and (path / "tokenizer_config.json").is_file()
    )

    def loader(path: Path):
        return AutoTokenizer.from_pretrained(path, local_files_only=True)

    return audit_tokenizer_paths(model_paths, loader)


def _fixed_environment_probe() -> dict[str, Any]:
    """Render Llama with one fixed date under three ambient environments."""

    model_path = MODEL_ROOT / "Llama-3.2-1B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer_record = tokenizer_identity(tokenizer, model_path)
    original_datetime = chat_template_utils.datetime
    original_locale = locale_module.setlocale(locale_module.LC_ALL)
    original_timezone = os.environ.get("TZ")
    cases = (
        ("2026-07-17T01:02:03", "UTC", "C"),
        ("2031-12-31T23:59:59", "Asia/Shanghai", "zh_CN.utf8"),
        ("2042-02-03T04:05:06", "America/New_York", "en_US.utf8"),
    )
    outputs = []
    try:
        for ambient_date, timezone_name, locale_name in cases:
            fixed = datetime.fromisoformat(ambient_date)

            class AmbientDatetime(datetime):
                @classmethod
                def now(cls, tz=None):
                    return cls(
                        fixed.year,
                        fixed.month,
                        fixed.day,
                        fixed.hour,
                        fixed.minute,
                        fixed.second,
                        tzinfo=tz,
                    )

            os.environ["TZ"] = timezone_name
            time.tzset()
            locale_module.setlocale(locale_module.LC_ALL, locale_name)
            chat_template_utils.datetime = AmbientDatetime
            record = build_prompt_identity_record(
                messages=[{"role": "user", "content": "Determinism probe."}],
                tokenizers={"sender": tokenizer},
                model_paths={"sender": model_path},
                tokenizer_records={"sender": tokenizer_record},
                add_generation_prompt=True,
                enable_thinking=False,
                remove_last_suffix=False,
                template_kwargs={"date_string": "17 Jul 2026"},
            )
            role = record["roles"]["sender"]
            outputs.append(
                {
                    "ambient_date": ambient_date,
                    "timezone": timezone_name,
                    "locale": locale_name,
                    "record_sha256": prompt_identity_record_sha256(record),
                    "rendered_prompt_sha256": role["rendered_prompt_sha256"],
                    "input_ids_sha256": role["input_ids_sha256"],
                    "token_count": role["token_count"],
                }
            )
    finally:
        chat_template_utils.datetime = original_datetime
        locale_module.setlocale(locale_module.LC_ALL, original_locale)
        if original_timezone is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_timezone
        time.tzset()
    identity_fields = (
        "record_sha256",
        "rendered_prompt_sha256",
        "input_ids_sha256",
        "token_count",
    )
    return {
        "fixed_date_string": "17 Jul 2026",
        "cases": outputs,
        "all_hashes_and_counts_identical": all(
            len({row[field] for row in outputs}) == 1 for field in identity_fields
        ),
    }


def _phase1_entries() -> list[dict[str, Any]]:
    manifest = _load_json(PHASE1_MANIFEST)
    llama_runs = [
        run for run in manifest.get("runs", []) if run.get("pair") == "llama32_1b"
    ]
    dates_by_seed_task: dict[tuple[int, str], set[str]] = {}
    paths_by_seed_task: dict[tuple[int, str], list[str]] = {}
    for run in llama_runs:
        seed = int(run["seed"])
        for task, output_value in (
            run.get("evaluation", {}).get("output_dirs", {}).items()
        ):
            output_dir = _resolve_workspace_path(str(output_value))
            artifact = _single_prediction_path(output_dir)
            if artifact is None:
                continue
            date = _artifact_date(artifact)
            if date is not None:
                dates_by_seed_task.setdefault((seed, str(task)), set()).add(date)
            paths_by_seed_task.setdefault((seed, str(task)), []).append(str(artifact))
    entries = []
    for (seed, task), dates in sorted(dates_by_seed_task.items()):
        entries.append(
            _entry(
                phase="Phase 1 source suite",
                scope="within-seed component contrasts",
                pair="llama32_1b",
                seed=seed,
                task=task,
                classification="safe" if len(dates) == 1 else "ambiguous",
                comparator_date=next(iter(dates)) if len(dates) == 1 else None,
                intervention_date=next(iter(dates)) if len(dates) == 1 else None,
                evidence=(
                    "All B1/B2/B3/B5/B6 arms for this seed/task share one UTC "
                    "artifact date and the same base prompt config."
                    if len(dates) == 1
                    else "Component arms span multiple dates without prompt hashes."
                ),
                artifact_paths=sorted(paths_by_seed_task[(seed, task)]),
            )
        )
    all_dates = sorted(
        {date for dates in dates_by_seed_task.values() for date in dates}
    )
    entries.append(
        _entry(
            phase="Phase 1 source suite",
            scope="cross-seed Llama3.2 aggregate",
            pair="llama32_1b",
            seed="42/43/44",
            task="all",
            classification="ambiguous" if len(all_dates) > 1 else "safe",
            comparator_date=all_dates[0] if all_dates else None,
            intervention_date=all_dates[-1] if all_dates else None,
            evidence=(
                "Seed 42 artifacts use 20260717 while seeds 43/44 use 20260718; "
                "no per-example input hashes were recorded, so prompt date is "
                "confounded with seed in aggregate variance."
            ),
            artifact_paths=[],
        )
    )
    return entries


def _phase15_entries() -> list[dict[str, Any]]:
    manifest = _load_json(PHASE15_MANIFEST)
    entries = []
    for run in manifest.get("runs", []):
        if run.get("pair") != "llama32_1b":
            continue
        base_configs = run.get("base_eval_configs", {})
        intervention_configs = run.get("eval_configs", {})
        for task, comparator_value in (
            run.get("native_comparator", {}).get("prediction_csv", {}).items()
        ):
            comparator = Path(str(comparator_value))
            output_dir = Path(str(run.get("output_dirs", {}).get(task, "")))
            intervention = _single_prediction_path(output_dir)
            comparator_date = _artifact_date(comparator)
            intervention_date = (
                None if intervention is None else _artifact_date(intervention)
            )
            base_config = _resolve_workspace_path(str(base_configs.get(task, "")))
            intervention_config = _resolve_workspace_path(
                str(intervention_configs.get(task, ""))
            )
            prompt_configs_equal = (
                base_config.is_file()
                and intervention_config.is_file()
                and _prompt_config_view(base_config)
                == _prompt_config_view(intervention_config)
            )
            if (
                comparator_date
                and intervention_date
                and comparator_date != intervention_date
            ):
                classification = "invalid"
                evidence = (
                    "The native and intervention arms use different UTC dates; the "
                    "Llama3.2 template is confirmed to emit different input IDs across dates."
                )
            elif comparator_date and intervention_date and prompt_configs_equal:
                classification = "safe"
                evidence = (
                    "Paired arms use the same prompt-relevant config and UTC date; "
                    "the intervention changes only the registered evaluation control."
                )
            else:
                classification = "ambiguous"
                evidence = (
                    "No per-example input hashes and the date/config equivalence could "
                    "not be established from provenance."
                )
            entries.append(
                _entry(
                    phase="Phase 1.5 causal diagnostics",
                    scope=str(run["id"]),
                    pair="llama32_1b",
                    seed=int(run["seed"]),
                    task=str(task),
                    classification=classification,
                    comparator_date=comparator_date,
                    intervention_date=intervention_date,
                    prompt_configs_equal=prompt_configs_equal,
                    evidence=evidence,
                    comparator_path=str(comparator),
                    intervention_path=(
                        None if intervention is None else str(intervention)
                    ),
                    base_config=str(base_config),
                    intervention_config=str(intervention_config),
                )
            )
    return entries


def _source_artifact_paths(manifest: Mapping[str, Any]) -> list[Path]:
    values = manifest.get("inputs", {}).get("source_artifacts", [])
    paths = [
        Path(str(value["path"]))
        for value in values
        if value.get("pair") == "llama32_1b" and value.get("path")
    ]
    if paths:
        return paths
    # Phase 2A-0 freezes its Phase-1 suite manifest SHA rather than expanding
    # every source path. Recover only the Llama B6 provenance from that manifest;
    # prediction CSV contents are never opened.
    phase1 = _load_json(PHASE1_MANIFEST)
    for run in phase1.get("runs", []):
        if run.get("pair") != "llama32_1b" or run.get("variant") != "b6":
            continue
        for output_value in run.get("evaluation", {}).get("output_dirs", {}).values():
            artifact = _single_prediction_path(
                _resolve_workspace_path(str(output_value))
            )
            if artifact is not None:
                paths.append(artifact)
    return paths


def _phase2a_cpu_entry(phase: str, manifest_path: Path) -> list[dict[str, Any]]:
    manifest = _load_json(manifest_path)
    paths = _source_artifact_paths(manifest)
    dates = sorted({date for path in paths if (date := _artifact_date(path))})
    return [
        _entry(
            phase=phase,
            scope="CPU analysis over immutable stored artifacts",
            pair="llama32_1b",
            seed="42/43/44",
            task="all",
            classification="safe",
            comparator_date=None,
            intervention_date=None,
            evidence=(
                "The CPU stage does not render tokenizers and binds source artifacts "
                "by path/SHA; reproducing the stored-artifact computation is safe."
            ),
            artifact_paths=[str(path) for path in paths],
        ),
        _entry(
            phase=phase,
            scope="Llama3.2 cross-seed/generalization interpretation",
            pair="llama32_1b",
            seed="42/43/44",
            task="all",
            classification="ambiguous" if len(dates) > 1 else "safe",
            comparator_date=dates[0] if dates else None,
            intervention_date=dates[-1] if dates else None,
            evidence=(
                "The immutable Llama3.2 source artifacts span UTC dates "
                f"{dates} without per-example input hashes; inherited prompt-date "
                "confounding cannot be separated from seed/pair effects."
            ),
            artifact_paths=[str(path) for path in paths],
        ),
    ]


def _phase2a2_entries() -> list[dict[str, Any]]:
    execution = _load_json(PHASE2A2_EXECUTION)
    equivalence = _load_json(PHASE2A2_EQUIVALENCE)
    created = str(execution.get("created_at", ""))
    current_date = created[:10].replace("-", "") if len(created) >= 10 else None
    entries = []
    for task, reference in execution.get("assets", {}).get("references", {}).items():
        reference_path = Path(str(reference["path"]))
        entries.append(
            _entry(
                phase="Phase 2A-2a cache geometry",
                scope="Llama3.2 frozen-reference Gate 1",
                pair="llama32_1b",
                seed=42,
                task=str(task),
                classification="invalid",
                comparator_date=_artifact_date(reference_path),
                intervention_date=current_date,
                evidence=(
                    "Historical reference and instrumented run use different dates; "
                    "the equivalence debug confirmed identical current OFF/ON outputs "
                    "and different date-token IDs."
                ),
                comparator_path=str(reference_path),
                intervention_path=str(execution.get("results_root")),
                superseding_classification=equivalence.get("classification"),
            )
        )
    entries.append(
        _entry(
            phase="Phase 2A-2a cache geometry",
            scope="geometry predictivity",
            pair="three heterogeneous pairs",
            seed=42,
            task="all",
            classification="ambiguous",
            comparator_date=None,
            intervention_date=None,
            evidence=(
                "The preregistered pilot stopped at Gate 1; geometry/outcome join and "
                "predictivity gates were never evaluated. This is untested, not negative."
            ),
            artifact_paths=[],
        )
    )
    return entries


def build_audit() -> dict[str, Any]:
    models = _scan_models()
    entries = [
        *_phase1_entries(),
        *_phase15_entries(),
        *_phase2a_cpu_entry("Phase 2A-0 opportunity audit", PHASE2A0_MANIFEST),
        *_phase2a_cpu_entry("Phase 2A-1 selector kill-test", PHASE2A1_FREEZE),
        *_phase2a2_entries(),
    ]
    counts = {
        classification: sum(row["classification"] == classification for row in entries)
        for classification in sorted(CLASSIFICATIONS)
    }
    return {
        "schema_version": 1,
        "role": "phase2a2_prompt_identity_audit",
        "created_at": _utc_now(),
        "constraints": {
            "gpu_used": False,
            "prediction_csv_contents_opened": False,
            "correctness_or_labels_read": False,
            "sealed_test_read": False,
        },
        "sources": {
            "model_root": str(MODEL_ROOT),
            "phase1_manifest": {
                "path": str(PHASE1_MANIFEST),
                "sha256": sha256_file(PHASE1_MANIFEST),
            },
            "phase15_manifest": {
                "path": str(PHASE15_MANIFEST),
                "sha256": sha256_file(PHASE15_MANIFEST),
            },
            "phase2a0_manifest": {
                "path": str(PHASE2A0_MANIFEST),
                "sha256": sha256_file(PHASE2A0_MANIFEST),
            },
            "phase2a1_freeze": {
                "path": str(PHASE2A1_FREEZE),
                "sha256": sha256_file(PHASE2A1_FREEZE),
            },
            "phase2a2_execution": {
                "path": str(PHASE2A2_EXECUTION),
                "sha256": sha256_file(PHASE2A2_EXECUTION),
            },
        },
        "model_audit": models,
        "cpu_environment_matrix": _fixed_environment_probe(),
        "classification_counts": counts,
        "affected_experiments": entries,
    }


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "phase",
        "scope",
        "pair",
        "seed",
        "task",
        "classification",
        "comparator_date",
        "intervention_date",
        "prompt_configs_equal",
        "evidence",
        "comparator_path",
        "intervention_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    result = build_audit()
    _atomic_json(args.output_json, result)
    write_csv(args.output_csv, result["affected_experiments"])
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_csv": str(args.output_csv),
                "models": len(result["model_audit"]),
                "entries": len(result["affected_experiments"]),
                "classification_counts": result["classification_counts"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
