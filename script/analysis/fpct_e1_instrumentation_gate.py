#!/usr/bin/env python3
"""Export outcome-free FPCT-E1 instrumentation hard-gate evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.cache_utils import DynamicCache

from rosetta.model.fpct_attention import (
    FPCTSidecarSegment,
    fpct_mechanism_diagnostics,
    pack_fpct_memory,
)
from rosetta.model.fpct_instrumentation import (
    FPCTCaptureAccumulator,
    teacher_forced_query_mask,
)
from rosetta.model.wrapper import RosettaModel


SCHEMA_VERSION = 1
SEED = 20260726

# The v1 artifacts are historical evidence.  They predate the A4 streaming
# implementation and must never be rewritten in-place.  The compatibility
# gate id remains unchanged because execution-plan consumers intentionally
# require it; ``attestation_id`` distinguishes the prospective successor.
CONSOLIDATED_GATE_ID = "fpct_e1_instrumentation_hard_gate_v1"
A4_ATTESTATION_ID = "fpct_e1_a4_instrumentation_reattestation_v1"
A4_SCHEMA_VERSION = 2
A4_PARITY_PROTOCOL_ID = "fpct_e1_instrumentation_parity_a4_v1"
A4_SYNTHETIC_PROTOCOL_ID = "fpct_e1_synthetic_query_variance_a4_v1"
A4_PARITY_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_instrumentation_parity_a4.json"
)
A4_SYNTHETIC_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance_a4.json"
)
A4_HARD_GATE_RELATIVE = Path(
    "recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate_a4.json"
)
LEGACY_V1_RELATIVES = frozenset(
    {
        Path("recipe/eval_recipe/fpct_e1/e1_instrumentation_parity.json"),
        Path("recipe/eval_recipe/fpct_e1/e1_synthetic_query_variance.json"),
        Path("recipe/eval_recipe/fpct_e1/e1_instrumentation_hard_gate.json"),
    }
)
REQUIRED_GATE_CHECKS = (
    "formula_oracles",
    "instrumentation_on_off_equivalent",
    "synthetic_query_variance_positive",
    "multistep_accumulation_preserved",
    "invalid_probability_gradient_exact_zero",
    "no_nan_inf",
)
A4_SOURCE_FILES = (
    "rosetta/model/fpct_instrumentation.py",
    "rosetta/model/fpct_attention.py",
    "rosetta/model/projector.py",
    "rosetta/model/wrapper.py",
    "script/analysis/fpct_e1_instrumentation_gate.py",
    "script/analysis/fpct_e1_mechanism_audit.py",
    "script/analysis/fpct_reference_operator.py",
    "script/experiment/fpct_e1_runtime_backend.py",
)
A4_TEST_FILES = (
    "test/test_fpct_instrumentation.py",
    "test/test_fpct_qwen_cpu_integration.py",
    "test/test_fpct_reference_operator.py",
    "test/test_fpct_production_path.py",
    "test/test_fpct_e1_runtime_capture_primitives.py",
    "test/test_fpct_e1_runtime_backend.py",
    "test/test_fpct_e1_instrumentation_gate_a4.py",
)


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(value))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def publish_json_transaction_no_overwrite(
    artifacts: tuple[tuple[Path, Mapping[str, Any]], ...],
    *,
    final_validator: Callable[[], bool] | None = None,
) -> None:
    """Publish an all-or-nothing immutable artifact set.

    Temporary files are fsynced in each destination directory.  Every linked
    destination is owned by this call, so a later failure can safely unlink
    only those newly published paths.  Pre-existing paths are never touched.
    """

    paths = tuple(path for path, _value in artifacts)
    if len(set(paths)) != len(paths):
        raise ValueError("transaction output paths must be distinct")
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError(f"prospective artifacts already exist: {existing}")
    temporary_paths: list[str] = []
    published: list[Path] = []
    encoded = [(path, canonical_bytes(value)) for path, value in artifacts]
    try:
        for path, payload in encoded:
            path.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{path.name}.", dir=path.parent
            )
            temporary_paths.append(temporary)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        for (path, payload), temporary in zip(encoded, temporary_paths):
            try:
                os.link(temporary, path)
            except FileExistsError as error:
                raise FileExistsError(
                    f"prospective artifact appeared during publication: {path}"
                ) from error
            published.append(path)
            if path.read_bytes() != payload:
                raise RuntimeError(f"published artifact byte verification failed: {path}")
        if final_validator is not None and not final_validator():
            raise RuntimeError("source/test closure changed during A4 publication")
    except BaseException:
        for path in reversed(published):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary in temporary_paths:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _all_finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return value == value and abs(value) != float("inf")
    if isinstance(value, Mapping):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _a4_mark(value: Mapping[str, Any], protocol_id: str) -> dict[str, Any]:
    marked = dict(value)
    marked.update(
        {
            "schema_version": A4_SCHEMA_VERSION,
            "protocol_id": protocol_id,
            "attestation_id": A4_ATTESTATION_ID,
            "prospective_a4_successor": True,
            "historical_v1_artifacts_overwritten": False,
        }
    )
    return marked


def _repo_relative(repo_root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(repo_root.resolve())
    except ValueError as error:
        raise ValueError(f"A4 artifact/evidence must be inside repo root: {path}") from error


def _record(repo_root: Path, path: Path, kind: str) -> dict[str, Any]:
    relative = _repo_relative(repo_root, path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "kind": kind,
        "logical_path": f"repo://{relative.as_posix()}",
        "sha256": sha256_file(path),
        "byte_size": path.stat().st_size,
    }


def _record_bytes(
    repo_root: Path, path: Path, kind: str, payload: bytes
) -> dict[str, Any]:
    relative = _repo_relative(repo_root, path)
    return {
        "kind": kind,
        "logical_path": f"repo://{relative.as_posix()}",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
    }


def _validate_a4_outputs(
    repo_root: Path,
    parity_output: Path,
    synthetic_output: Path,
    hard_gate_output: Path,
    *,
    canonical_paths_required: bool,
) -> None:
    outputs = (parity_output.resolve(), synthetic_output.resolve(), hard_gate_output.resolve())
    if len(set(outputs)) != 3:
        raise ValueError("A4 parity, synthetic, and hard-gate outputs must be distinct")
    relative = tuple(_repo_relative(repo_root, path) for path in outputs)
    if any(path in LEGACY_V1_RELATIVES for path in relative):
        raise ValueError("historical v1 instrumentation artifacts are immutable")
    expected = (A4_PARITY_RELATIVE, A4_SYNTHETIC_RELATIVE, A4_HARD_GATE_RELATIVE)
    if canonical_paths_required and relative != expected:
        raise ValueError(f"A4 successor outputs must use canonical paths: {expected}")
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"A4 successor artifacts are immutable and already exist: {existing}")


def run_a4_cpu_test_suite(repo_root: Path) -> dict[str, Any]:
    """Run the fixed outcome-free CPU/offline source-and-test closure."""

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--no-cov",
        "-p",
        "no:cacheprovider",
        *A4_TEST_FILES,
    ]
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    with tempfile.TemporaryDirectory(prefix="fpct-e1-a4-instrumentation-") as temporary:
        command.extend(("--basetemp", str(Path(temporary) / "pytest")))
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    matches = re.findall(r"(?:^|\s)(\d+) passed(?:,|\s|$)", completed.stdout)
    passed = int(matches[-1]) if matches else 0
    return {
        "runner": "python -m pytest",
        "test_files": list(A4_TEST_FILES),
        "passed": passed,
        "exit_code": completed.returncode,
        "cuda_visible_devices": "",
        "offline_environment": True,
        "status": "GO" if completed.returncode == 0 and passed > 0 else "BLOCKED",
    }


def _qwen_model(state: dict[str, Tensor] | None = None) -> Qwen3ForCausalLM:
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=True,
    )
    config._attn_implementation = "eager"
    torch.manual_seed(1701)
    model = Qwen3ForCausalLM(config)
    if state is not None:
        model.load_state_dict(state)
    model.eval()
    return model


def _add_sidecar(wrapper: RosettaModel) -> None:
    generator = torch.Generator().manual_seed(719)
    key = torch.randn(1, 2, 4, 2, 8, generator=generator)
    value = torch.randn(1, 2, 4, 2, 8, generator=generator)
    prior = torch.tensor([[[0.6, 0.4], [1.0, 0.0], [0.7, 0.3], [1.0, 0.0]]])
    wrapper._store_fpct_sidecar(0, 0, key, value, prior, prior > 0)


def _forward(
    wrapper: RosettaModel,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    cache: DynamicCache | None = None,
    labels: Tensor | None = None,
):
    return wrapper._base_model_forward_with_fpct(
        input_ids=input_ids,
        attention_mask=attention_mask,
        past_key_values=cache or DynamicCache(),
        labels=labels,
        use_cache=True,
        return_dict=True,
    )


def _cache_equal(left: DynamicCache, right: DynamicCache) -> bool:
    if left.get_seq_length() != right.get_seq_length():
        return False
    return all(
        torch.equal(a, b)
        for a, b in zip(
            [*left.key_cache, *left.value_cache],
            [*right.key_cache, *right.value_cache],
        )
    )


def parity_evidence() -> dict[str, Any]:
    state = _qwen_model().state_dict()
    baseline = RosettaModel([_qwen_model(state)], fpct_operator="f")
    captured = RosettaModel([_qwen_model(state)], fpct_operator="f")
    _add_sidecar(baseline)
    _add_sidecar(captured)
    ids = torch.tensor([[4, 5, 6, 7]])
    mask = torch.ones(1, 4, dtype=torch.long)
    labels = torch.tensor([[-100, -100, 6, 7]])

    baseline_teacher = _forward(baseline, ids, mask, labels=labels)
    captured.begin_fpct_capture(
        "teacher_forced_response",
        metadata={"gate": "fpct-e1", "sample": "random-qwen3"},
        query_mask=teacher_forced_query_mask(labels),
    )
    captured_teacher = _forward(captured, ids, mask, labels=labels)
    teacher_report = captured.end_fpct_capture()

    logits_equal = torch.equal(baseline_teacher.logits, captured_teacher.logits)
    loss_equal = torch.equal(baseline_teacher.loss, captured_teacher.loss)
    teacher_cache_equal = _cache_equal(
        baseline_teacher.past_key_values, captured_teacher.past_key_values
    )

    baseline_cache: DynamicCache | None = None
    captured_cache: DynamicCache | None = None
    baseline_tokens: list[int] = []
    captured_tokens: list[int] = []
    baseline_input = ids
    captured_input = ids.clone()
    current_mask = mask
    captured.begin_fpct_capture("greedy_decode")
    decode_logits_equal = True
    decode_cache_equal = True
    for step in range(3):
        baseline_output = _forward(
            baseline, baseline_input, current_mask, cache=baseline_cache
        )
        captured_output = _forward(
            captured, captured_input, current_mask, cache=captured_cache
        )
        decode_logits_equal &= torch.equal(
            baseline_output.logits, captured_output.logits
        )
        decode_cache_equal &= _cache_equal(
            baseline_output.past_key_values, captured_output.past_key_values
        )
        baseline_next = int(baseline_output.logits[:, -1].argmax(dim=-1))
        captured_next = int(captured_output.logits[:, -1].argmax(dim=-1))
        baseline_tokens.append(baseline_next)
        captured_tokens.append(captured_next)
        baseline_cache = baseline_output.past_key_values
        captured_cache = captured_output.past_key_values
        baseline_input = torch.tensor([[baseline_next]])
        captured_input = torch.tensor([[captured_next]])
        current_mask = torch.ones(1, ids.shape[1] + step + 1, dtype=torch.long)
    decode_report = captured.end_fpct_capture()

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "fpct_e1_instrumentation_parity_v1",
        "model": "Qwen3ForCausalLM random tiny config; no pretrained weights",
        "seed": SEED,
        "teacher_forced": {
            "logits_bitwise_equal": logits_equal,
            "loss_bitwise_equal": loss_equal,
            "cache_bitwise_equal": teacher_cache_equal,
            "eligible_query_count": teacher_report["layers"]["0"]["eligible_query_count"],
            "prompt_queries_excluded": True,
        },
        "greedy_decode": {
            "steps": 3,
            "logits_bitwise_equal_each_step": decode_logits_equal,
            "cache_bitwise_equal_each_step": decode_cache_equal,
            "baseline_tokens": baseline_tokens,
            "captured_tokens": captured_tokens,
            "tokens_equal": baseline_tokens == captured_tokens,
            "capture_forward_count": decode_report["forward_count"],
        },
        "capture_contract": {
            "stores_raw_kv": teacher_report["stores_raw_kv"],
            "parent_summaries_present": bool(
                teacher_report["layers"]["0"]["parent_summaries"]
            ),
            "json_serializable": True,
        },
    }
    result["status"] = "GO" if all(
        (
            logits_equal,
            loss_equal,
            teacher_cache_equal,
            decode_logits_equal,
            decode_cache_equal,
            baseline_tokens == captured_tokens,
            not teacher_report["stores_raw_kv"],
            result["capture_contract"]["parent_summaries_present"],
        )
    ) else "BLOCKED"
    return result


def _packed(key: Tensor, value: Tensor, prior: Tensor, query_length: int):
    sidecar = FPCTSidecarSegment(
        parent_start=0,
        key=key,
        value=value,
        prior=prior.float(),
        valid=prior > 0,
    )
    return pack_fpct_memory(
        key[..., 0, :].clone(),
        value[..., 0, :].clone(),
        torch.zeros(1, 1, query_length, key.shape[2]),
        [sidecar],
        query_length=query_length,
    )


def synthetic_evidence() -> dict[str, Any]:
    key = torch.tensor([[[[[2.0, 0.0], [-2.0, 0.0]]]]])
    value = torch.tensor([[[[[1.0, 0.0], [0.0, 1.0]]]]])
    prior = torch.tensor([[[0.5, 0.5]]])
    packed = _packed(key, value, prior, 1)
    capture = FPCTCaptureAccumulator(
        "teacher_forced_response",
        metadata={"case": "query-changing"},
        query_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    for query in (
        torch.tensor([[[[1.0, 0.0]]]]),
        torch.tensor([[[[-1.0, 0.0]]]]),
    ):
        metrics, payload = fpct_mechanism_diagnostics(
            query, packed, return_capture_payload=True
        )
        capture.update(0, metrics, payload)
    report = capture.finalize()
    parent = report["layers"]["0"]["parent_summaries"][0]

    identical_key = torch.tensor([[[[[1.0, -0.5], [1.0, -0.5]]]]])
    identical_value = torch.tensor([[[[[0.25, 2.0], [0.25, 2.0]]]]])
    identical_prior = torch.tensor([[[0.3, 0.7]]])
    identical_query = torch.tensor([[[[1.0, 0.0], [-1.0, 0.5]]]])
    identical_metrics = fpct_mechanism_diagnostics(
        identical_query,
        _packed(identical_key, identical_value, identical_prior, 2),
    )

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol_id": "fpct_e1_synthetic_query_variance_v1",
        "query_changing": {
            "forward_count": report["forward_count"],
            "eligible_query_count": report["layers"]["0"]["eligible_query_count"],
            "gamma_query_variance": report["metrics"]["gamma_query_variance"],
            "posterior_top1_any_change": report["metrics"]["posterior_top1_any_change"],
            "posterior_top1_change_rate": report["metrics"]["posterior_top1_change_rate"],
            "parent_query_count": parent["query_count"],
            "candidate_gamma_moments": parent["candidate_gamma_moments"],
        },
        "identical_candidates": {
            "gamma_kl_prior": float(identical_metrics["gamma_kl_prior"]),
            "gamma_tv_prior": float(identical_metrics["gamma_tv_prior"]),
            "jensen_gap": float(identical_metrics["jensen_gap"]),
        },
        "stores_raw_kv": report["stores_raw_kv"],
    }
    tolerance = 1e-7
    result["status"] = "GO" if (
        result["query_changing"]["gamma_query_variance"] > 0
        and result["query_changing"]["posterior_top1_any_change"]
        and result["query_changing"]["parent_query_count"] == 2
        and abs(result["identical_candidates"]["gamma_kl_prior"]) <= tolerance
        and abs(result["identical_candidates"]["gamma_tv_prior"]) <= tolerance
        and abs(result["identical_candidates"]["jensen_gap"]) <= tolerance
        and not result["stores_raw_kv"]
    ) else "BLOCKED"
    return result


def _test_summary_valid(summary: Mapping[str, Any]) -> bool:
    return (
        summary.get("status") == "GO"
        and summary.get("exit_code") == 0
        and isinstance(summary.get("passed"), int)
        and not isinstance(summary.get("passed"), bool)
        and summary["passed"] > 0
        and summary.get("cuda_visible_devices") == ""
        and summary.get("offline_environment") is True
        and summary.get("test_files") == list(A4_TEST_FILES)
    )


def _instrumentation_equivalent(parity: Mapping[str, Any]) -> bool:
    teacher = parity.get("teacher_forced", {})
    decode = parity.get("greedy_decode", {})
    return bool(
        parity.get("status") == "GO"
        and teacher.get("logits_bitwise_equal") is True
        and teacher.get("loss_bitwise_equal") is True
        and teacher.get("cache_bitwise_equal") is True
        and decode.get("logits_bitwise_equal_each_step") is True
        and decode.get("cache_bitwise_equal_each_step") is True
        and decode.get("tokens_equal") is True
    )


def _synthetic_variance_positive(synthetic: Mapping[str, Any]) -> bool:
    query = synthetic.get("query_changing", {})
    return bool(
        synthetic.get("status") == "GO"
        and isinstance(query.get("gamma_query_variance"), (int, float))
        and not isinstance(query.get("gamma_query_variance"), bool)
        and query["gamma_query_variance"] > 0
        and query.get("posterior_top1_any_change") is True
    )


def _multistep_preserved(
    parity: Mapping[str, Any], synthetic: Mapping[str, Any]
) -> bool:
    decode = parity.get("greedy_decode", {})
    query = synthetic.get("query_changing", {})
    return bool(
        decode.get("steps") == 3
        and decode.get("capture_forward_count") == 3
        and query.get("forward_count") == 2
        and query.get("parent_query_count") == 2
    )


def build_a4_successor_attestation(
    *,
    repo_root: Path,
    parity_output: Path,
    synthetic_output: Path,
    hard_gate_output: Path,
    test_runner: Callable[[Path], Mapping[str, Any]] = run_a4_cpu_test_suite,
    canonical_paths_required: bool = True,
) -> dict[str, Any]:
    """Build a no-overwrite A4 successor without touching historical v1 files.

    The source/test closure is hashed both before and after the subprocess test
    run.  A concurrently changing file therefore cannot receive a GO receipt.
    ``test_runner`` exists only to make the builder itself unit-testable; the
    command-line path always uses :func:`run_a4_cpu_test_suite`.
    """

    repo_root = repo_root.resolve()
    parity_output = parity_output.resolve()
    synthetic_output = synthetic_output.resolve()
    hard_gate_output = hard_gate_output.resolve()
    _validate_a4_outputs(
        repo_root,
        parity_output,
        synthetic_output,
        hard_gate_output,
        canonical_paths_required=canonical_paths_required,
    )

    source_paths = [repo_root / relative for relative in A4_SOURCE_FILES]
    test_paths = [repo_root / relative for relative in A4_TEST_FILES]
    closure_paths = source_paths + test_paths
    if len(set(closure_paths)) != len(closure_paths):
        raise ValueError("A4 instrumentation source/test closure contains duplicates")
    before = {str(path.relative_to(repo_root)): sha256_file(path) for path in closure_paths}
    test_summary = dict(test_runner(repo_root))
    parity = _a4_mark(parity_evidence(), A4_PARITY_PROTOCOL_ID)
    synthetic = _a4_mark(synthetic_evidence(), A4_SYNTHETIC_PROTOCOL_ID)
    after = {str(path.relative_to(repo_root)): sha256_file(path) for path in closure_paths}
    source_test_stable = before == after
    parity_bytes = canonical_bytes(parity)
    synthetic_bytes = canonical_bytes(synthetic)

    evidence = [
        _record_bytes(repo_root, parity_output, "generated_evidence", parity_bytes),
        _record_bytes(
            repo_root, synthetic_output, "generated_evidence", synthetic_bytes
        ),
        *(_record(repo_root, path, "source") for path in source_paths),
        *(_record(repo_root, path, "test") for path in test_paths),
    ]
    tests_go = _test_summary_valid(test_summary)
    checks = {
        "formula_oracles": tests_go,
        "instrumentation_on_off_equivalent": _instrumentation_equivalent(parity),
        "synthetic_query_variance_positive": _synthetic_variance_positive(synthetic),
        "multistep_accumulation_preserved": _multistep_preserved(parity, synthetic),
        # These two invariants are exercised in both the pure reference and
        # production-path files in the fixed, hashed CPU test closure.
        "invalid_probability_gradient_exact_zero": tests_go,
        "no_nan_inf": tests_go and _all_finite(parity) and _all_finite(synthetic),
    }
    if tuple(checks) != REQUIRED_GATE_CHECKS:
        raise AssertionError("A4 instrumentation gate check ordering drifted")
    go = all(checks.values()) and source_test_stable
    if not go:
        failed = [name for name, value in checks.items() if not value]
        if not source_test_stable:
            failed.append("source_test_stable_during_attestation")
        raise RuntimeError(
            "A4 instrumentation re-attestation failed before publication: "
            + ", ".join(failed)
        )
    gate = {
        "schema_version": A4_SCHEMA_VERSION,
        "gate_id": CONSOLIDATED_GATE_ID,
        "attestation_id": A4_ATTESTATION_ID,
        "status": "GO",
        "decision": "GO_PROSPECTIVE_A4_REATTESTATION",
        "logical_path": f"repo://{_repo_relative(repo_root, hard_gate_output).as_posix()}",
        "prospective_a4_successor": True,
        "historical_v1_artifacts": {
            "status": "immutable_historical_evidence",
            "overwritten": False,
            "superseded_for_a4_execution_only": True,
            "paths": [path.as_posix() for path in sorted(LEGACY_V1_RELATIVES)],
        },
        "checks": checks,
        "evidence": evidence,
        "evidence_closure": {
            "generated_evidence_count": 2,
            "source_count": len(source_paths),
            "test_count": len(test_paths),
            "all_declared_files_hashed": len(evidence)
            == 2 + len(source_paths) + len(test_paths),
            "source_test_stable_during_attestation": source_test_stable,
            "records_sha256": hashlib.sha256(canonical_bytes(evidence)).hexdigest(),
        },
        "test_summary": test_summary,
        "runtime_contract": {
            "device": "CPU",
            "cuda_visible_devices": "",
            "offline": True,
            "model": "Qwen3ForCausalLM random tiny config; no pretrained weights",
        },
        "firewall": {
            "natural_data_accessed": False,
            "pretrained_weights_or_checkpoint_loaded": False,
            "e1_pilot_consumed": False,
            "gpu_or_kubernetes_used": False,
            "training_used": False,
        },
    }
    publish_json_transaction_no_overwrite(
        (
            (parity_output, parity),
            (synthetic_output, synthetic),
            (hard_gate_output, gate),
        ),
        final_validator=lambda: {
            str(path.relative_to(repo_root)): sha256_file(path)
            for path in closure_paths
        }
        == after,
    )
    return gate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-output", type=Path, required=True)
    parser.add_argument("--synthetic-output", type=Path, required=True)
    parser.add_argument("--a4-hard-gate-output", type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    if args.a4_hard_gate_output is not None:
        gate = build_a4_successor_attestation(
            repo_root=args.repo_root,
            parity_output=args.parity_output,
            synthetic_output=args.synthetic_output,
            hard_gate_output=args.a4_hard_gate_output,
        )
        print(
            json.dumps(
                {
                    "attestation_id": gate["attestation_id"],
                    "status": gate["status"],
                    "parity": {
                        "sha256": sha256_file(args.parity_output),
                        "path": str(args.parity_output),
                    },
                    "synthetic": {
                        "sha256": sha256_file(args.synthetic_output),
                        "path": str(args.synthetic_output),
                    },
                    "hard_gate": {
                        "sha256": sha256_file(args.a4_hard_gate_output),
                        "path": str(args.a4_hard_gate_output),
                    },
                },
                sort_keys=True,
            )
        )
        return 0 if gate["status"] == "GO" else 1
    parity = parity_evidence()
    synthetic = synthetic_evidence()
    atomic_json(args.parity_output, parity)
    atomic_json(args.synthetic_output, synthetic)
    payload = {
        "parity": {"status": parity["status"], "sha256": sha256_file(args.parity_output)},
        "synthetic": {"status": synthetic["status"], "sha256": sha256_file(args.synthetic_output)},
    }
    print(json.dumps(payload, sort_keys=True))
    return 0 if parity["status"] == synthetic["status"] == "GO" else 1


if __name__ == "__main__":
    raise SystemExit(main())
