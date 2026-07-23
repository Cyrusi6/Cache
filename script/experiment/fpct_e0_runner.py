from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

import yaml


E0_UID = "fpct-e0-20260722-v1"
E0_CONTAINER_ROOT = Path("/fpct-e0")
IMAGE = "docker.io/library/fpct-gpu-r2m:80fb295@sha256:0ea40657a38dbe8778ef492474f28ef2360156f4c9b0f8287e2bb0988d695851"
IMAGE_DIGEST = "sha256:0ea40657a38dbe8778ef492474f28ef2360156f4c9b0f8287e2bb0988d695851"
SCIENTIFIC_COMMIT = "80fb295542ad298fae4cddb1273517b401bbcd17"
SIDECAR_SHA256 = "48caee80b31925a6074c9c5304bd861163f4e2e21adb55ebec9bf00237e2d990"
SIDECAR_CONTAINER_PATH = "/fpct-assets/mmlu_auxiliary_train_2048_certified.pt"
RUNTIME_ASSET_FILES = {
    "/models/c2c/Qwen3-0.6B/config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "/models/c2c/Qwen3-0.6B/tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "/models/c2c/Qwen3-0.6B/model.safetensors": "f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b",
    "/models/c2c/TinyLlama-1.1B-Chat-v1.0/config.json": "486bedda3a6988332e60d9638a09ca4b260d34ebcf1b19e22cf3b140b63d8fe9",
    "/models/c2c/TinyLlama-1.1B-Chat-v1.0/tokenizer.json": "bcd04f0eadf90287bd26e1a183ac487d8a141b09b06aecb7725bbdd343640f2e",
    "/models/c2c/TinyLlama-1.1B-Chat-v1.0/tokenizer.model": "9e556afd44213b6bd1be2b850ebbbd98f5481437a8021afaf58ee7fb1818d347",
    "/models/c2c/TinyLlama-1.1B-Chat-v1.0/model.safetensors": "6e6001da2106d4757498752a021df6c2bdc332c650aae4bae6b0c004dcf14933",
}
SEED_ORDER = {
    2026072201: ("c_post", "f"),
    2026072202: ("f", "c_post"),
    2026072203: ("c_post", "f"),
}
TASK_LIMITS = {"ai2-arc": 128, "openbookqa": 70, "mmlu-redux": 128}
SUPPORT_EXECUTION = "7aecf2370df8a544b553baa6a7a58b24191e02ef"


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(item.relative_to(path).as_posix().encode() + b"\0")
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any, *, overwrite: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _jsonable(value: Any) -> Any:
    try:
        import torch
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except ImportError:
        pass
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _prompt(question: str, choices: Iterable[str]) -> str:
    from rosetta.utils.evaluate import build_prompt
    choice_text = "".join(f"{chr(65 + index)}. {value}\n" for index, value in enumerate(choices))
    return build_prompt(
        dataset="mmlu-redux", locale="", question=question, choices=choice_text,
        use_cot=False, use_template=True,
    )


def prepare_dev_manifest(
    repo_root: Path,
    shared_root: Path,
    support_root: Path,
    output: Path,
) -> dict[str, Any]:
    os.environ.setdefault("C2C_MODEL_ROOT", str(shared_root / "models"))
    audit = _load_module(
        "fpct_e0_projection",
        repo_root / "script/analysis/fpct_1b_structural_support_audit.py",
    )
    samples = audit.load_projected_samples(shared_root)
    audit.validate_canonical_samples(samples)
    by_group: dict[tuple[str, str], list[Any]] = {}
    for sample in samples:
        by_group.setdefault((sample.task, sample.content_group_sha256), []).append(sample)

    from transformers import AutoTokenizer
    from rosetta.model.aligner import AlignmentStrategy, TokenAligner
    from rosetta.utils.evaluate import set_default_chat_template
    from rosetta.utils.model_loading import resolve_model_path

    receiver_name = "Qwen/Qwen3-0.6B"
    sender_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    receiver = AutoTokenizer.from_pretrained(resolve_model_path(receiver_name))
    sender = AutoTokenizer.from_pretrained(resolve_model_path(sender_name))
    set_default_chat_template(receiver, receiver_name)
    set_default_chat_template(sender, sender_name)
    aligner = TokenAligner(
        slm_tokenizer=receiver,
        llm_tokenizer=sender,
        strategy=AlignmentStrategy("soft_span_overlap_v2"),
        soft_alignment_score_mode="uniform",
        soft_alignment_boundary_bonus=0.5,
        soft_alignment_boundary_tolerance=1,
        soft_alignment_min_weight=0.0,
        soft_alignment_confidence_mode="entropy",
        soft_alignment_confidence_alpha=0.5,
        soft_alignment_confidence_floor=0.5,
        soft_alignment_fallback_confidence=0.25,
    )

    selected_groups: dict[str, list[str]] = {}
    ledgers: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    for task, limit in TASK_LIMITS.items():
        ledger = support_root / f"tinyllama__{task}" / "group_support.csv"
        geometry = support_root / f"tinyllama__{task}" / "r1_parent_geometry.csv"
        ledgers[task] = {
            "group_support_sha256": sha256_file(ledger),
            "parent_geometry_sha256": sha256_file(geometry),
        }
        with ledger.open(newline="", encoding="utf-8") as handle:
            candidates = [
                row for row in csv.DictReader(handle)
                if row["split"] == "calibration"
                and row["has_certified_m2"] == "1"
                and row["member_consistent"] == "1"
            ]
        hashes = sorted(row["content_group_sha256"] for row in candidates)[:limit]
        if len(hashes) < 30:
            raise RuntimeError(f"{task} has only {len(hashes)} certified calibration groups")
        selected_groups[task] = hashes
        task_samples = sorted(
            (sample for group in hashes for sample in by_group[(task, group)]),
            key=lambda sample: (sample.subject, sample.content_group_sha256, sample.sample_key_sha256),
        )
        subject_position: dict[str, int] = {}
        for task_position, sample in enumerate(task_samples):
            prompt = _prompt(sample.question, sample.choices)
            details = aligner.align_chat_messages_soft(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                return_details=True,
                enable_thinking=False,
                remove_last_surfix=False,
                top_k=4,
            )
            details = aligner.sanitize_fpct_soft_alignment(
                details,
                target_length=len(details["slm_ids"]),
                source_length=len(details["llm_ids"]),
            )
            soft = _jsonable(details["soft_alignment"])
            indices = soft["source_indices"]
            weights = soft["source_weights"]
            candidate_count = max(
                (sum(1 for index, weight in zip(row_i, row_w) if int(index) >= 0 and float(weight) > 0)
                 for row_i, row_w in zip(indices, weights)),
                default=0,
            )
            if candidate_count < 2:
                raise RuntimeError(f"selected certified group lost support: {sample.content_group_sha256}")
            if task in {"ai2-arc", "openbookqa"}:
                split_point = len(task_samples) // 2
                evaluation_subject = "SPLIT_0_OF_2" if task_position < split_point else "SPLIT_1_OF_2"
                evaluation_question_id = task_position
            else:
                evaluation_subject = sample.subject
                evaluation_question_id = subject_position.get(sample.subject, 0)
                subject_position[sample.subject] = evaluation_question_id + 1
            alignment_payload = {
                "slm_ids": details["slm_ids"],
                "llm_ids": details["llm_ids"],
                "source_indices": indices,
                "source_weights": weights,
            }
            rows.append({
                "task": task,
                "source_split": "canonical_calibration_from_frozen_test_projection",
                "subject": sample.subject,
                "source_row_id": sample.question_id,
                "sample_key_sha256": sample.sample_key_sha256,
                "content_group_sha256": sample.content_group_sha256,
                "rendered_prompt_sha256": sha256_bytes(details["slm_text"].encode("utf-8")),
                "alignment_sha256": sha256_bytes(canonical_bytes(alignment_payload)),
                "candidate_count": candidate_count,
                "eligibility": "certified_m_ge_2",
                "evaluation_subject": evaluation_subject,
                "evaluation_question_id": evaluation_question_id,
            })
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_e0_exploratory_dev_manifest_v1",
        "run_uid": E0_UID,
        "pair": "TinyLlama-1.1B-Chat-v1.0_to_Qwen3-0.6B",
        "source_support_execution": SUPPORT_EXECUTION,
        "selection": {
            "source_partition": "calibration",
            "label_free": True,
            "ordering": "content_group_sha256_ascending",
            "limits": TASK_LIMITS,
            "uses_logits_attention_or_correctness": False,
        },
        "ledgers": ledgers,
        "group_counts": {task: len(values) for task, values in selected_groups.items()},
        "row_counts": {task: sum(row["task"] == task for row in rows) for task in TASK_LIMITS},
        "rows": rows,
        "confirmatory_model_selection_or_test_accessed": False,
    }
    atomic_json(output, payload)
    return payload


def _training_config(repo_root: Path, seed: int, arm: str) -> dict[str, Any]:
    runner = _load_module(
        f"fpct_e0_confirmatory_config_{seed}_{arm}",
        repo_root / "script/experiment/fpct_confirmatory_runner.py",
    )
    lock_like = {
        "run_uid": E0_UID,
        "assets": {
            "training_alignment_sidecar_2048": {
                "container_path": SIDECAR_CONTAINER_PATH,
            }
        },
    }
    output = E0_CONTAINER_ROOT / "seeds" / str(seed) / "active" / arm
    return runner.training_config(lock_like, seed, arm, output, examples=2048, optimizer_steps=64)


def _eval_config(seed: int, trained: str, inference: str, task: str, subjects: list[str]) -> dict[str, Any]:
    checkpoint = E0_CONTAINER_ROOT / "seeds" / str(seed) / "active" / trained / "final"
    cell = {("c_post", "c_post"): "Y_CC", ("c_post", "f"): "Y_CF", ("f", "c_post"): "Y_FC", ("f", "f"): "Y_FF"}[(trained, inference)]
    config: dict[str, Any] = {
        "model": {
            "model_name": "Rosetta",
            "rosetta_config": {
                "base_model": "Qwen/Qwen3-0.6B",
                "teacher_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "checkpoints_dir": str(checkpoint),
                "attn_implementation": "eager",
                "is_do_alignment": True,
                "alignment_strategy": "soft_span_overlap_v2",
                "soft_alignment_top_k": 4,
                "soft_alignment_score_mode": "uniform",
                "soft_alignment_boundary_bonus": 0.5,
                "soft_alignment_boundary_tolerance": 1,
                "soft_alignment_min_weight": 0.0,
                "soft_alignment_confidence_mode": "entropy",
                "soft_alignment_confidence_alpha": 0.5,
                "soft_alignment_confidence_floor": 0.5,
                "soft_alignment_fallback_confidence": 0.25,
                "fpct_alignment_sanitizer": "certified_slot0_v1",
                "fpct_operator": inference,
                "include_response": False,
                "fpct_instrumentation": False,
            },
            "generation_config": {"do_sample": False, "max_new_tokens": 64},
        },
        "output": {"output_dir": str(E0_CONTAINER_ROOT / "seeds" / str(seed) / "active" / "eval" / cell / task)},
        "eval": {
            "dataset": task,
            "data_root": str(E0_CONTAINER_ROOT / "dev_data"),
            "gpu_ids": [0, 1],
            "answer_method": "generate",
            "use_cot": False,
            "use_template": True,
            "sample_interval": 1,
            "debug_dump_bad_samples": False,
            "gate_diagnostics": False,
            "math_grading_method": "comprehensive",
        },
    }
    if task == "mmlu-redux":
        config["eval"]["subjects"] = subjects
    return config


def sidecar_semantic_hash(sidecar: Path) -> dict[str, Any]:
    import torch
    payload = torch.load(sidecar, map_location="cpu", weights_only=False)
    if payload.get("examples") != 2048 or len(payload.get("items", [])) != 2048:
        raise RuntimeError("unexpected training sidecar length")
    digest = hashlib.sha256()
    prompt_mask_ok = True
    for item in payload["items"]:
        labels = list(item["labels"])
        supervised = [index for index, value in enumerate(labels) if int(value) != -100]
        prompt_mask_ok &= bool(supervised) and all(int(value) == -100 for value in labels[: supervised[0]])
        projected = {
            "messages": _jsonable(item["messages"]),
            "input_ids": _jsonable(item["input_ids"]),
            "labels": labels,
        }
        digest.update(canonical_bytes(projected))
    if not prompt_mask_ok:
        raise RuntimeError("training sidecar violates response-only label mask")
    return {"rendered_training_sha256": digest.hexdigest(), "prompt_labels_all_minus_100": True}


def render_configs(repo_root: Path, dev_manifest_path: Path, output_dir: Path, sidecar: Path) -> dict[str, Any]:
    dev = json.loads(dev_manifest_path.read_text())
    subjects = sorted({row["subject"] for row in dev["rows"] if row["task"] == "mmlu-redux"})
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for seed, order in SEED_ORDER.items():
        for arm in order:
            name = f"train_{seed}_{arm}.json"
            path = output_dir / name
            atomic_json(path, _training_config(repo_root, seed, arm))
            records.append({"kind": "training", "seed": seed, "arm": arm, "filename": name, "sha256": sha256_file(path)})
        for trained, inference in (("c_post", "c_post"), ("c_post", "f"), ("f", "c_post"), ("f", "f")):
            cell = {("c_post", "c_post"): "Y_CC", ("c_post", "f"): "Y_CF", ("f", "c_post"): "Y_FC", ("f", "f"): "Y_FF"}[(trained, inference)]
            for task in TASK_LIMITS:
                name = f"eval_{seed}_{cell}_{task}.yaml"
                path = output_dir / name
                path.write_text(yaml.safe_dump(_eval_config(seed, trained, inference, task, subjects), sort_keys=True), encoding="utf-8")
                records.append({"kind": "evaluation", "seed": seed, "cell": cell, "task": task, "filename": name, "sha256": sha256_file(path)})
    sidecar_record = sidecar_semantic_hash(sidecar)
    payload = {
        "schema_version": 1,
        "protocol_id": "fpct_e0_rendered_config_index_v1",
        "run_uid": E0_UID,
        "records": records,
        "record_count": len(records),
        "bundle_sha256": sha256_bytes(canonical_bytes(records)),
        "alignment_sidecar_sha256": sha256_file(sidecar),
        **sidecar_record,
    }
    if payload["alignment_sidecar_sha256"] != SIDECAR_SHA256:
        raise RuntimeError("alignment sidecar SHA mismatch")
    atomic_json(output_dir / "config_index.json", payload)
    return payload


def materialize_dev(manifest_path: Path, source_root: Path, output_root: Path) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.ipc as ipc
    import pyarrow.parquet as pq

    manifest = json.loads(manifest_path.read_text())
    if output_root.exists():
        raise FileExistsError(output_root)
    temporary = output_root.with_name(f".{output_root.name}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    rows_by_task = {task: [row for row in manifest["rows"] if row["task"] == task] for task in TASK_LIMITS}
    temporary.mkdir(parents=True)

    arc_source = source_root / "ai2_arc/ARC-Challenge/test-00000-of-00001.parquet"
    arc_raw = pq.read_table(arc_source).to_pylist()
    arc_rows = [arc_raw[int(row["source_row_id"])] for row in rows_by_task["ai2-arc"]]
    arc_out = temporary / "ai2_arc/ARC-Challenge/test-00000-of-00001.parquet"
    arc_out.parent.mkdir(parents=True); pq.write_table(pa.Table.from_pylist(arc_rows), arc_out)

    obqa_source = source_root / "openbookqa/main/test-00000-of-00001.parquet"
    obqa_raw = pq.read_table(obqa_source).to_pylist()
    obqa_rows = [obqa_raw[int(row["source_row_id"])] for row in rows_by_task["openbookqa"]]
    obqa_out = temporary / "openbookqa/main/test-00000-of-00001.parquet"
    obqa_out.parent.mkdir(parents=True); pq.write_table(pa.Table.from_pylist(obqa_rows), obqa_out)

    mmlu_rows = rows_by_task["mmlu-redux"]
    for subject in sorted({row["subject"] for row in mmlu_rows}):
        source = source_root / "mmlu-redux-2.0" / subject / "data-00000-of-00001.arrow"
        with source.open("rb") as handle:
            raw = ipc.open_stream(handle).read_all().to_pylist()
        selected = [raw[int(row["source_row_id"])] for row in mmlu_rows if row["subject"] == subject]
        out = temporary / "mmlu-redux-2.0" / subject / "test-00000-of-00001.parquet"
        out.parent.mkdir(parents=True, exist_ok=True); pq.write_table(pa.Table.from_pylist(selected), out)
    os.replace(temporary, output_root)
    payload = {"schema_version": 1, "row_counts": {task: len(rows) for task, rows in rows_by_task.items()}, "tree_sha256": tree_sha256(output_root)}
    atomic_json(output_root.parent / "dev_data_manifest.json", payload)
    return payload


def _copy_configs(config_map_root: Path, run_root: Path, index: dict[str, Any]) -> None:
    target = run_root / "configs"
    target.mkdir(parents=True, exist_ok=True)
    for record in index["records"]:
        source = config_map_root / record["filename"]
        if sha256_file(source) != record["sha256"]:
            raise RuntimeError(f"ConfigMap config SHA mismatch: {source}")
        destination = target / record["filename"]
        if destination.exists():
            if sha256_file(destination) != record["sha256"]:
                raise RuntimeError(f"existing config differs: {destination}")
        else:
            shutil.copyfile(source, destination)


def _verify_runtime_inputs(manifest: dict[str, Any], config_map_root: Path, run_root: Path) -> dict[str, str]:
    expected_files = {
        "config_index.json": manifest["hashes"]["config_index_sha256"],
        "exploratory_dev_manifest.json": manifest["hashes"]["development_group_manifest_sha256"],
        "fpct_e0_runner.py": manifest["hashes"]["runner_sha256"],
        "fpct_e0_formula_oracle.py": manifest["hashes"]["formula_oracle_sha256"],
        "fpct_e0_effect_report.py": manifest["hashes"]["effect_report_sha256"],
    }
    observed: dict[str, str] = {}
    for name, expected in expected_files.items():
        observed[name] = sha256_file(config_map_root / name)
        if observed[name] != expected:
            raise RuntimeError(f"ConfigMap payload SHA mismatch: {name}")
    for raw_path, expected in RUNTIME_ASSET_FILES.items():
        path = Path(raw_path)
        observed[raw_path] = sha256_file(path)
        if observed[raw_path] != expected:
            raise RuntimeError(f"mounted model/tokenizer SHA mismatch: {raw_path}")
    observed[SIDECAR_CONTAINER_PATH] = sha256_file(Path(SIDECAR_CONTAINER_PATH))
    if observed[SIDECAR_CONTAINER_PATH] != SIDECAR_SHA256:
        raise RuntimeError("mounted alignment sidecar SHA mismatch")
    dev_data = json.loads((run_root / "dev_data_manifest.json").read_text())
    if dev_data["tree_sha256"] != manifest["development_data"]["materialized_tree_sha256"]:
        raise RuntimeError("materialized development data SHA mismatch")
    observed["dev_data_tree_sha256"] = dev_data["tree_sha256"]
    return observed


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _training_command(seed: int, arm: str, config: Path, attestation: Path) -> list[str]:
    return [
        "/opt/conda/bin/python3.11", "-m", "torch.distributed.run", "--standalone",
        "--nproc-per-node=2", "--no-python", "/opt/conda/bin/python3.11", "-I",
        "/opt/fpct/script/runtime/fpct_bootstrap.py", "--repo-root", "/opt/fpct",
        "--target", "/opt/fpct/script/train/SFT_train.py", "--include-gpu-closure",
        "--attestation-out", str(attestation), "--", "--config", str(config),
    ]


def _verify_matched(seed: int, attempt: Path, order: tuple[str, str]) -> dict[str, Any]:
    records = {}
    for arm in order:
        path = attempt / arm / "fpct_formal_integrity.json"
        records[arm] = json.loads(path.read_text())
        if records[arm]["optimizer_steps"] != 64 or not records[arm]["checkpoint_reload_equal"]:
            raise RuntimeError(f"incomplete training integrity: {seed}/{arm}")
    equal_fields = [
        "step0_trainable_sha256", "trainable_keys_sha256",
        "rng_state_before_training_sha256", "data_order_sha256",
        "training_examples", "optimizer_class", "optimizer_group_count",
        "optimizer_learning_rates", "optimizer_weight_decay", "scheduler_class",
        "scheduler_initial_state_sha256",
    ]
    mismatches = [field for field in equal_fields if records[order[0]].get(field) != records[order[1]].get(field)]
    if mismatches:
        raise RuntimeError(f"matched arm identity mismatch: {mismatches}")
    payload = {"schema_version": 1, "seed": seed, "arm_order": list(order), "equal_fields": equal_fields, "records": records, "status": "GO"}
    atomic_json(attempt / "matched_integrity.json", payload)
    return payload


def mechanism_probe(seed: int, run_root: Path, config_root: Path, dev_manifest_path: Path) -> dict[str, Any]:
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, "/opt/fpct")
    evaluator_module = _load_module("fpct_e0_unified_evaluator", Path("/opt/fpct/script/evaluation/unified_evaluator.py"))
    from rosetta.utils.dataset_loading import load_c2c_dataset
    from rosetta.utils.evaluate import load_rosetta_model, set_default_chat_template
    from rosetta.utils.model_loading import resolve_model_path

    config = yaml.safe_load((config_root / f"eval_{seed}_Y_FF_ai2-arc.yaml").read_text())
    config["model"]["rosetta_config"]["fpct_instrumentation"] = True
    config["output"]["output_dir"] = str(run_root / "seeds" / str(seed) / "active" / "mechanism_probe_setup")
    evaluator = evaluator_module.UnifiedEvaluator(config)
    device = torch.device("cuda:0")
    model, tokenizer = load_rosetta_model(config["model"], config["eval"], device, config["model"]["generation_config"])
    teacher_name = config["model"]["rosetta_config"]["teacher_model"]
    llm_tokenizer = AutoTokenizer.from_pretrained(resolve_model_path(teacher_name))
    set_default_chat_template(llm_tokenizer, teacher_name)
    dev = json.loads(dev_manifest_path.read_text())
    records = []
    for task in TASK_LIMITS:
        row = next(value for value in dev["rows"] if value["task"] == task)
        evaluator.dataset_name = task; evaluator.dataset_config = evaluator_module.DATASET_CONFIGS[task]
        if task == "ai2-arc": config_name = "ARC-Challenge"
        elif task == "openbookqa": config_name = "main"
        else: config_name = row["subject"]
        data = load_c2c_dataset(evaluator.dataset_config["dataset_name"], config_name=config_name, split="test", data_root_path=str(run_root / "dev_data"))
        index = int(row["evaluation_question_id"])
        example = data[index]
        prompt = evaluator.format_example(example, use_cot=False)
        prepared = evaluator.prepare_model_inputs(prompt, tokenizer, device, "rosetta", llm_tokenizer, "generate")
        with torch.no_grad():
            model.generate(**prepared["inputs"], do_sample=False, max_new_tokens=2)
        metrics = {key: float(value.detach().cpu()) for key, value in model._fpct_mechanism_metrics.items()}
        records.append({"task": task, "content_group_sha256": row["content_group_sha256"], "metrics": metrics})
    keys = sorted({key for record in records for key in record["metrics"]})
    aggregate = {key: sum(record["metrics"].get(key, 0.0) for record in records) / len(records) for key in keys}
    signal_keys = ("gamma_kl_prior", "jensen_gap", "gamma_query_variance", "candidate_logit_range", "d_k", "d_v", "output_delta_l2")
    nonzero = any(aggregate.get(key, 0.0) > 0 for key in signal_keys)
    payload = {"schema_version": 1, "seed": seed, "records": records, "aggregate": aggregate, "nonzero_activation": nonzero}
    atomic_json(run_root / "seeds" / str(seed) / "active" / "mechanism_diagnostics.json", payload)
    return payload


def run_seed(seed: int, attempt_number: int, config_map_root: Path, run_root: Path) -> dict[str, Any]:
    if seed not in SEED_ORDER:
        raise ValueError(seed)
    manifest = json.loads((config_map_root / "e0_manifest.json").read_text())
    index = json.loads((config_map_root / "config_index.json").read_text())
    if manifest["run_uid"] != E0_UID or manifest["exact_image"] != IMAGE:
        raise RuntimeError("E0 manifest identity mismatch")
    asset_sha256 = _verify_runtime_inputs(manifest, config_map_root, run_root)
    _copy_configs(config_map_root, run_root, index)
    if not (run_root / "dev_data_manifest.json").is_file():
        raise RuntimeError("E0 dev data was not materialized before execution")
    parity = json.loads((run_root / "parity" / "formula_production_parity.json").read_text())
    if parity.get("status") != "GO":
        raise RuntimeError("E0 production parity is not GO")
    seed_root = run_root / "seeds" / str(seed)
    seed_root.mkdir(parents=True, exist_ok=True)
    attempt = seed_root / f"attempt_{attempt_number}"
    if attempt.exists():
        raise FileExistsError(attempt)
    attempt.mkdir()
    active = seed_root / "active"
    if active.is_symlink():
        active.unlink()
    elif active.exists():
        raise RuntimeError("active seed path is not a symlink")
    active.symlink_to(attempt.name)
    provenance = {
        "schema_version": 1,
        "run_uid": E0_UID,
        "seed": seed,
        "attempt": attempt_number,
        "pod": os.environ.get("E0_POD_NAME"),
        "node": os.environ.get("E0_NODE_NAME"),
        "image": IMAGE,
        "image_provenance": json.loads(Path("/opt/fpct/.fpct_image_provenance.json").read_text()),
        "configmap_tree_sha256": tree_sha256(config_map_root),
        "sidecar_sha256": sha256_file(Path(SIDECAR_CONTAINER_PATH)),
        "mounted_asset_sha256": asset_sha256,
    }
    try:
        import torch
        provenance["torch"] = torch.__version__
        provenance["cuda"] = torch.version.cuda
        provenance["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    except Exception as error:
        provenance["gpu_probe_error"] = str(error)
    atomic_json(attempt / "runtime_provenance.json", provenance)
    order = SEED_ORDER[seed]
    for arm in order:
        config = run_root / "configs" / f"train_{seed}_{arm}.json"
        attestation = attempt / "attestations" / f"{seed}-{arm}-rank_{{rank}}.json"
        attestation.parent.mkdir(parents=True, exist_ok=True)
        _run(_training_command(seed, arm, config, attestation), cwd=Path("/opt/fpct"))
    integrity = _verify_matched(seed, attempt, order)
    for cell in ("Y_CC", "Y_CF", "Y_FC", "Y_FF"):
        for task in TASK_LIMITS:
            config = run_root / "configs" / f"eval_{seed}_{cell}_{task}.yaml"
            _run(["/opt/conda/bin/python3.11", "/opt/fpct/script/evaluation/unified_evaluator.py", "--config", str(config)], cwd=Path("/opt/fpct"))
    mechanism = mechanism_probe(seed, run_root, run_root / "configs", config_map_root / "exploratory_dev_manifest.json")
    payload = {"schema_version": 1, "seed": seed, "attempt": attempt_number, "status": "COMPLETE", "integrity": integrity["status"], "mechanism_nonzero": mechanism["nonzero_activation"]}
    atomic_json(attempt / "seed_complete.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare-dev")
    prep.add_argument("--repo-root", type=Path, required=True)
    prep.add_argument("--shared-root", type=Path, required=True)
    prep.add_argument("--support-root", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    render = sub.add_parser("render-configs")
    render.add_argument("--repo-root", type=Path, required=True)
    render.add_argument("--dev-manifest", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--sidecar", type=Path, required=True)
    materialize = sub.add_parser("materialize-dev")
    materialize.add_argument("--manifest", type=Path, required=True)
    materialize.add_argument("--source-root", type=Path, required=True)
    materialize.add_argument("--output-root", type=Path, required=True)
    seed_parser = sub.add_parser("run-seed")
    seed_parser.add_argument("--seed", type=int, required=True)
    seed_parser.add_argument("--attempt", type=int, default=1)
    seed_parser.add_argument("--config-map-root", type=Path, default=Path("/opt/fpct-e0"))
    seed_parser.add_argument("--run-root", type=Path, default=E0_CONTAINER_ROOT)
    args = parser.parse_args()
    if args.command == "prepare-dev":
        payload = prepare_dev_manifest(args.repo_root.resolve(), args.shared_root.resolve(), args.support_root.resolve(), args.output.resolve())
    elif args.command == "render-configs":
        payload = render_configs(args.repo_root.resolve(), args.dev_manifest.resolve(), args.output_dir.resolve(), args.sidecar.resolve())
    elif args.command == "materialize-dev":
        payload = materialize_dev(args.manifest.resolve(), args.source_root.resolve(), args.output_root.resolve())
    else:
        payload = run_seed(args.seed, args.attempt, args.config_map_root.resolve(), args.run_root.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
