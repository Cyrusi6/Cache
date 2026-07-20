"""
Use SFT trainer to train rosetta model
"""

import gc
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.optimization import get_scheduler
from torch.optim import AdamW
from tqdm import tqdm
import os
import sys
import json
import yaml
import argparse
import shutil
import wandb
import torch.distributed as dist  # Added for Distributed Data Parallel support
from torch.nn.parallel import DistributedDataParallel  # For type checking
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional, Union
import math
import contextlib
import hashlib
from pathlib import Path

from rosetta.model.wrapper import RosettaModel
from rosetta.model.projector import create_projector, save_projector
from rosetta.train.dataset_adapters import (
    ChatDataset,
    AlignedChatDataset,
    RosettaDataCollator,
    create_dataset,
    BaselineDataCollator,
    BaselineChatDataset,
)
from rosetta.train.answer_prior import (
    compute_answer_prior_regularization,
    resolve_option_token_ids,
)
from rosetta.train.answer_margin import compute_answer_margin_routing_loss
from rosetta.model.aligner import TokenAligner, AlignmentStrategy
from rosetta.train.model_utils import k_nearest_sources, last_aligned_sources
from rosetta.model.projector import AllInOneProjector
from rosetta.utils.evaluate import set_default_chat_template
from rosetta.utils.model_loading import model_matches, resolve_model_path

# PEFT imports for LoRA (baseline mode)
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

torch.autograd.set_detect_anomaly(True)


def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # For distributed training
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def enable_full_determinism():
    """Enable stricter determinism settings for reproducibility."""
    # Must be set before CUDA context creation for cuBLAS determinism
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    # PyTorch deterministic algorithms (may raise if non-deterministic ops are used)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    # Disable TF32 to reduce numeric variability
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass


def _load_dataset_split_indices(
    path: Path,
    dataset_size: int,
    train_size: int,
    eval_size: int,
) -> Tuple[List[int], List[int]]:
    """Load and validate a frozen train/eval split manifest."""
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    manifest_dataset_size = int(payload.get("dataset_size", dataset_size))
    if manifest_dataset_size != dataset_size:
        raise ValueError(
            f"Frozen split {path} declares dataset_size={manifest_dataset_size}; "
            f"expected {dataset_size}"
        )
    train_indices = [int(index) for index in payload["train_indices"]]
    eval_indices = [int(index) for index in payload["eval_indices"]]
    if len(train_indices) != train_size or len(eval_indices) != eval_size:
        raise ValueError(
            f"Frozen split {path} has lengths "
            f"{len(train_indices)}/{len(eval_indices)}; expected "
            f"{train_size}/{eval_size}"
        )
    combined = train_indices + eval_indices
    if len(set(combined)) != dataset_size or set(combined) != set(range(dataset_size)):
        raise ValueError(
            f"Frozen split {path} must contain every dataset index exactly once"
        )
    expected_hashes = payload.get("indices_sha256", {})
    for name, indices in {
        "permutation": combined,
        "train": train_indices,
        "eval": eval_indices,
    }.items():
        expected = expected_hashes.get(name)
        if not expected:
            continue
        actual = hashlib.sha256(
            json.dumps(indices, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if actual != expected:
            raise ValueError(
                f"Frozen split {path} has invalid {name} index SHA256: "
                f"{actual} != {expected}"
            )
    return train_indices, eval_indices


def _write_dataset_split_indices(
    path: Path,
    train_indices: List[int],
    eval_indices: List[int],
    *,
    dataset_size: int,
    seed: int,
    split_mode: str,
) -> None:
    """Atomically record a split so later variants can share identical data order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset_size": dataset_size,
        "seed": seed,
        "split_mode": split_mode,
        "train_indices": train_indices,
        "eval_indices": eval_indices,
    }
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            existing = json.load(handle)
        if existing != payload:
            raise ValueError(
                f"Refusing to overwrite frozen dataset split with different content: {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def _create_dataset_split(
    dataset: Dataset,
    train_size: int,
    eval_size: int,
    *,
    seed: int,
    split_mode: str = "seeded",
    split_indices_path: Optional[Path] = None,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset, str]:
    """Create the requested split while preserving the seeded default behavior."""
    normalized_mode = str(split_mode).strip().lower()
    if split_indices_path is not None:
        train_indices, eval_indices = _load_dataset_split_indices(
            split_indices_path,
            len(dataset),
            train_size,
            eval_size,
        )
        return (
            torch.utils.data.Subset(dataset, train_indices),
            torch.utils.data.Subset(dataset, eval_indices),
            "frozen_indices",
        )
    if normalized_mode == "legacy_global_rng":
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, eval_size],
        )
        return train_dataset, eval_dataset, normalized_mode
    if normalized_mode == "seeded":
        split_generator = torch.Generator().manual_seed(seed)
        train_dataset, eval_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, eval_size],
            generator=split_generator,
        )
        return train_dataset, eval_dataset, normalized_mode
    raise ValueError(
        "data.split_mode must be 'seeded' or 'legacy_global_rng'; "
        f"got {normalized_mode!r}"
    )


def broadcast_decision_from_rank0(
    decision: bool, distributed: bool, device: str, rank: int
) -> bool:
    """Broadcast a boolean decision from rank 0 to all ranks so control flow matches."""
    if not distributed:
        return decision
    if rank == 0:
        tensor_flag = torch.tensor(
            [1 if decision else 0], device=device, dtype=torch.int
        )
    else:
        tensor_flag = torch.empty(1, device=device, dtype=torch.int)
    dist.broadcast(tensor_flag, src=0)
    return bool(tensor_flag.item())


def freeze_model(model: nn.Module):
    """Freeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_model(model: nn.Module):
    """Unfreeze all parameters in a model"""
    for param in model.parameters():
        param.requires_grad = True


def unfreeze_projectors(rosetta_model: RosettaModel):
    """Unfreeze only the projector parameters"""
    for projector in rosetta_model.projector_list:
        for param in projector.parameters():
            param.requires_grad = True


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def _projector_auxiliary_loss(model: nn.Module) -> Optional[torch.Tensor]:
    """Return the mean auxiliary projector loss from projectors used in this batch."""
    model_to_use = _unwrap_model(model)
    projector_list = getattr(model_to_use, "projector_list", [])
    losses = []
    for projector in projector_list:
        loss_fn = getattr(projector, "alignment_regularization_loss", None)
        if loss_fn is None or not callable(loss_fn):
            continue
        loss = loss_fn()
        if loss is not None:
            losses.append(loss)
    if not losses:
        return None
    return torch.stack(losses).mean()


def _collect_projector_diagnostics(model: nn.Module) -> Dict[str, float]:
    """Collect best-effort projector diagnostics for logging."""
    model_to_use = _unwrap_model(model)
    projector_list = getattr(model_to_use, "projector_list", [])
    attrs = {
        "last_alignment_aux_loss": "projector/alignment_aux_loss",
        "last_alignment_delta_l2": "projector/alignment_delta_l2",
        "last_alignment_regularized_delta_l2": "projector/alignment_regularized_delta_l2",
        "last_alignment_regularization_selected_rate": "projector/alignment_regularization_selected_rate",
        "last_alignment_key_delta_abs_mean": "projector/key_delta_abs_mean",
        "last_alignment_value_delta_abs_mean": "projector/value_delta_abs_mean",
        "last_alignment_key_delta_abs_max": "projector/key_delta_abs_max",
        "last_alignment_value_delta_abs_max": "projector/value_delta_abs_max",
        "last_alignment_key_confidence": "projector/key_confidence_mean",
        "last_alignment_value_confidence": "projector/value_confidence_mean",
        "last_alignment_key_confidence_std": "projector/key_confidence_std",
        "last_alignment_value_confidence_std": "projector/value_confidence_std",
        "last_alignment_key_layer_scale": "projector/key_layer_scale",
        "last_alignment_value_layer_scale": "projector/value_layer_scale",
        "last_alignment_key_residual_scale": "projector/key_residual_scale",
        "last_alignment_value_residual_scale": "projector/value_residual_scale",
        "last_alignment_residual_scale_delta_l2": "projector/residual_scale_delta_l2",
        "last_alignment_residual_scale_aux_loss": "projector/residual_scale_aux_loss",
        "last_alignment_weight_entropy": "projector/weight_entropy",
        "last_alignment_weight_calibrated_entropy": "projector/weight_calibrated_entropy",
        "last_alignment_weight_entropy_l2": "projector/weight_entropy_l2",
        "last_alignment_weight_delta_l2": "projector/weight_delta_l2",
        "last_alignment_weight_aux_loss": "projector/weight_aux_loss",
        "last_alignment_weight_calibration_selected_rate": "projector/weight_calibration_selected_rate",
        "last_alignment_weight_delta_abs_mean": "projector/weight_delta_abs_mean",
        "last_alignment_weight_delta_abs_max": "projector/weight_delta_abs_max",
        "last_alignment_weight_top1_mean": "projector/weight_top1_mean",
        "last_alignment_quality_confidence_mean": "projector/quality_confidence_mean",
        "last_alignment_quality_entropy_mean": "projector/quality_entropy_mean",
        "last_alignment_quality_top1_mean": "projector/quality_top1_mean",
        "last_alignment_quality_active_fraction_mean": "projector/quality_active_fraction_mean",
        "last_learned_alignment_key_entropy": "projector/learned_alignment_key_entropy",
        "last_learned_alignment_value_entropy": "projector/learned_alignment_value_entropy",
        "last_learned_alignment_key_top1_mean": "projector/learned_alignment_key_top1",
        "last_learned_alignment_value_top1_mean": "projector/learned_alignment_value_top1",
        "last_learned_alignment_key_anchor_mean": "projector/learned_alignment_key_anchor",
        "last_learned_alignment_value_anchor_mean": "projector/learned_alignment_value_anchor",
        "last_learned_alignment_valid_rate": "projector/learned_alignment_valid_rate",
        "last_learned_alignment_key_injection": "projector/learned_alignment_key_injection",
        "last_learned_alignment_value_injection": "projector/learned_alignment_value_injection",
        "last_learned_alignment_key_injection_std": "projector/learned_alignment_key_injection_std",
        "last_learned_alignment_value_injection_std": "projector/learned_alignment_value_injection_std",
        "last_learned_alignment_key_injection_delta_abs_mean": "projector/learned_alignment_key_injection_delta_abs_mean",
        "last_learned_alignment_value_injection_delta_abs_mean": "projector/learned_alignment_value_injection_delta_abs_mean",
        "last_learned_alignment_key_transfer_gate": "projector/learned_alignment_key_transfer_gate",
        "last_learned_alignment_value_transfer_gate": "projector/learned_alignment_value_transfer_gate",
        "last_learned_alignment_key_transfer_gate_std": "projector/learned_alignment_key_transfer_gate_std",
        "last_learned_alignment_value_transfer_gate_std": "projector/learned_alignment_value_transfer_gate_std",
        "last_learned_alignment_key_transfer_margin": "projector/learned_alignment_key_transfer_margin",
        "last_learned_alignment_value_transfer_margin": "projector/learned_alignment_value_transfer_margin",
        "last_learned_alignment_key_transfer_top1": "projector/learned_alignment_key_transfer_top1",
        "last_learned_alignment_value_transfer_top1": "projector/learned_alignment_value_transfer_top1",
        "last_learned_alignment_key_transfer_entropy": "projector/learned_alignment_key_transfer_entropy",
        "last_learned_alignment_value_transfer_entropy": "projector/learned_alignment_value_transfer_entropy",
        "last_learned_alignment_transfer_gate_selected_rate": "projector/learned_alignment_transfer_gate_selected_rate",
        "last_learned_alignment_prior_aux_loss": "projector/learned_alignment_prior_aux_loss",
        "last_learned_alignment_delta_l2": "projector/learned_alignment_delta_l2",
        "last_learned_alignment_prior_ce": "projector/learned_alignment_prior_ce",
        "last_learned_alignment_prior_selected_rate": "projector/learned_alignment_prior_selected_rate",
        "last_learned_alignment_prior_entropy": "projector/learned_alignment_prior_entropy",
        "last_learned_alignment_prior_top1": "projector/learned_alignment_prior_top1",
        "last_learned_alignment_aux_loss": "projector/learned_alignment_aux_loss",
        "last_learned_alignment_aux_ce": "projector/learned_alignment_aux_ce",
        "last_learned_alignment_aux_selected_rate": "projector/learned_alignment_aux_selected_rate",
        "last_learned_alignment_aux_target_entropy": "projector/learned_alignment_aux_target_entropy",
        "last_learned_alignment_aux_target_anchor": "projector/learned_alignment_aux_target_anchor",
        "last_learned_alignment_aux_target_top1": "projector/learned_alignment_aux_target_top1",
        "last_learned_alignment_aux_score_margin": "projector/learned_alignment_aux_score_margin",
        "last_learned_alignment_margin_rank_aux_loss": "projector/learned_alignment_margin_rank_aux_loss",
        "last_learned_alignment_margin_rank_loss": "projector/learned_alignment_margin_rank_loss",
        "last_learned_alignment_margin_rank_selected_rate": "projector/learned_alignment_margin_rank_selected_rate",
        "last_learned_alignment_margin_rank_pair_count": "projector/learned_alignment_margin_rank_pair_count",
        "last_learned_alignment_margin_rank_utility_margin": "projector/learned_alignment_margin_rank_utility_margin",
        "last_learned_alignment_margin_rank_utility_top1": "projector/learned_alignment_margin_rank_utility_top1",
        "last_learned_alignment_margin_rank_utility_anchor": "projector/learned_alignment_margin_rank_utility_anchor",
    }
    values: Dict[str, List[float]] = {log_name: [] for log_name in attrs.values()}
    for projector in projector_list:
        for attr_name, log_name in attrs.items():
            if not hasattr(projector, attr_name):
                continue
            value = getattr(projector, attr_name)
            if isinstance(value, torch.Tensor):
                value = value.detach().float().mean().cpu().item()
            try:
                values[log_name].append(float(value))
            except (TypeError, ValueError):
                continue
    return {
        log_name: sum(items) / len(items) for log_name, items in values.items() if items
    }


def _final_section_labels(
    labels: torch.Tensor,
    kv_cache_index: List[torch.Tensor],
    output_seq_len: int,
) -> torch.Tensor:
    """Return labels matching Rosetta's final output section."""
    if not kv_cache_index:
        return labels[:, -output_seq_len:]
    section_lengths = [section.shape[1] for section in kv_cache_index]
    final_start = sum(section_lengths[:-1])
    final_end = final_start + section_lengths[-1]
    section_labels = labels[:, final_start:final_end]
    if section_labels.size(1) != output_seq_len:
        section_labels = section_labels[:, -output_seq_len:]
    return section_labels


def _average_metric_window(
    metric_sums: Dict[str, float], metric_counts: Dict[str, int]
) -> Dict[str, float]:
    return {
        key: metric_sums[key] / max(1, metric_counts.get(key, 0)) for key in metric_sums
    }


def _accumulate_metric_window(
    metric_sums: Dict[str, float],
    metric_counts: Dict[str, int],
    metrics: Dict[str, float],
) -> None:
    for key, value in metrics.items():
        metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        metric_counts[key] = metric_counts.get(key, 0) + 1


def _projector_likelihood_auxiliary_loss(
    model: nn.Module,
    task_loss: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Build likelihood-aware router loss from gradients of task loss."""
    model_to_use = _unwrap_model(model)
    projector_list = getattr(model_to_use, "projector_list", [])
    losses = []
    for projector in projector_list:
        loss_fn = getattr(
            projector,
            "compute_learned_alignment_grad_auxiliary_loss",
            None,
        )
        if loss_fn is None or not callable(loss_fn):
            continue
        loss = loss_fn(task_loss)
        if loss is not None:
            losses.append(loss)
    if not losses:
        return None
    return torch.stack(losses).mean()


def _set_learned_alignment_forced_rank(
    model: nn.Module,
    rank: Optional[int],
) -> None:
    model_to_use = _unwrap_model(model)
    for projector in getattr(model_to_use, "projector_list", []):
        setter = getattr(projector, "set_learned_alignment_forced_rank", None)
        if setter is not None and callable(setter):
            setter(rank)


def _set_learned_alignment_replay_target(
    model: nn.Module,
    target: Optional[torch.Tensor],
) -> None:
    model_to_use = _unwrap_model(model)
    for projector in getattr(model_to_use, "projector_list", []):
        setter = getattr(projector, "set_learned_alignment_replay_target", None)
        if setter is not None and callable(setter):
            setter(target)


def _set_learned_alignment_replay_utility(
    model: nn.Module,
    utility: Optional[torch.Tensor],
) -> None:
    model_to_use = _unwrap_model(model)
    for projector in getattr(model_to_use, "projector_list", []):
        setter = getattr(projector, "set_learned_alignment_replay_utility", None)
        if setter is not None and callable(setter):
            setter(utility)


def _set_learned_alignment_replay_utility_valid(
    model: nn.Module,
    valid: Optional[torch.Tensor],
) -> None:
    model_to_use = _unwrap_model(model)
    for projector in getattr(model_to_use, "projector_list", []):
        setter = getattr(projector, "set_learned_alignment_replay_utility_valid", None)
        if setter is not None and callable(setter):
            setter(valid)


def _set_learned_alignment_replay_scoring_mode(
    model: nn.Module,
    enabled: bool,
) -> None:
    model_to_use = _unwrap_model(model)
    for projector in getattr(model_to_use, "projector_list", []):
        setter = getattr(projector, "set_learned_alignment_replay_scoring_mode", None)
        if setter is not None and callable(setter):
            setter(enabled)


def _normalized_candidate_replay_target(
    losses: torch.Tensor,
    temperature: float,
    normalize: bool,
    clip: float,
) -> torch.Tensor:
    scores = -losses.detach().float()
    if normalize and scores.numel() > 1:
        centered = scores - scores.mean()
        scores = centered / centered.square().mean().clamp_min(1e-8).sqrt()
    if clip > 0:
        scores = scores.clamp(-clip, clip)
    return torch.softmax(scores / max(float(temperature), 1e-6), dim=0)


def _last_valid_shifted_label_mask(
    valid_mask: torch.Tensor,
    max_positions: int,
) -> torch.Tensor:
    if max_positions <= 0:
        return valid_mask
    valid_int = valid_mask.to(dtype=torch.long)
    valid_counts = valid_int.sum(dim=1, keepdim=True)
    valid_seen = valid_int.cumsum(dim=1)
    rank_from_end = valid_counts - valid_seen
    return valid_mask & (rank_from_end < int(max_positions))


def _answer_suffix_shifted_label_mask(
    answer_mask: torch.Tensor,
    valid_mask: torch.Tensor,
    suffix_tokens: int,
) -> torch.Tensor:
    if answer_mask.numel() == 0 or not bool(answer_mask.any().item()):
        return torch.zeros_like(valid_mask)

    B, T = valid_mask.shape
    positions = torch.arange(T, device=valid_mask.device).unsqueeze(0).expand(B, T)
    sentinel = torch.full_like(positions, T)
    first_answer = torch.where(answer_mask, positions, sentinel).min(dim=1).values
    has_answer = first_answer < T
    span_len = max(1, int(suffix_tokens))
    return (
        has_answer[:, None]
        & valid_mask
        & (positions >= first_answer[:, None])
        & (positions < (first_answer[:, None] + span_len))
    )


def _candidate_replay_score_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_loss: torch.Tensor,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Score a forced candidate for replay-label construction."""
    score_mode = str(config.get("score_mode", "task_loss"))
    metrics: Dict[str, float] = {"candidate_replay/score_positions": 0.0}
    if score_mode == "task_loss":
        metrics["candidate_replay/score_task_loss_fallback"] = 1.0
        return task_loss.detach().float(), metrics

    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("Expected logits [batch, seq, vocab] and labels [batch, seq].")
    if labels.size(1) != logits.size(1):
        labels = labels[:, -logits.size(1) :]
    if logits.size(1) < 2:
        metrics["candidate_replay/score_task_loss_fallback"] = 1.0
        return task_loss.detach().float(), metrics

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels.ne(-100)
    option_ids = torch.tensor(
        [int(token_id) for token_id in config.get("option_token_ids", [])],
        dtype=torch.long,
        device=labels.device,
    )
    option_mask = (
        valid_mask & shifted_labels.unsqueeze(-1).eq(option_ids).any(dim=-1)
        if option_ids.numel() > 0
        else torch.zeros_like(valid_mask)
    )

    if score_mode == "answer_token_ce":
        score_mask = option_mask
    elif score_mode == "answer_suffix_ce":
        score_mask = _answer_suffix_shifted_label_mask(
            answer_mask=option_mask,
            valid_mask=valid_mask,
            suffix_tokens=int(config.get("answer_suffix_tokens", 3)),
        )
    elif score_mode == "suffix_ce":
        score_mask = _last_valid_shifted_label_mask(
            valid_mask=valid_mask,
            max_positions=int(config.get("fallback_suffix_tokens", 4)),
        )
    else:
        raise ValueError(
            f"Unsupported candidate replay score_mode {score_mode!r}; expected "
            "'task_loss', 'answer_token_ce', 'answer_suffix_ce', or 'suffix_ce'."
        )

    min_positions = int(config.get("min_score_positions", 1))
    num_positions = int(score_mask.sum().detach().cpu().item())
    fallback_used = 0.0
    if num_positions < min_positions:
        fallback_mode = str(config.get("fallback_score_mode", "suffix_ce"))
        if fallback_mode == "suffix_ce":
            score_mask = _last_valid_shifted_label_mask(
                valid_mask=valid_mask,
                max_positions=int(config.get("fallback_suffix_tokens", 4)),
            )
            num_positions = int(score_mask.sum().detach().cpu().item())
            fallback_used = 1.0
        elif fallback_mode == "task_loss":
            metrics["candidate_replay/score_task_loss_fallback"] = 1.0
            metrics["candidate_replay/score_fallback_used"] = 1.0
            return task_loss.detach().float(), metrics
        else:
            raise ValueError(
                f"Unsupported candidate replay fallback_score_mode {fallback_mode!r}; "
                "expected 'suffix_ce' or 'task_loss'."
            )

    if num_positions <= 0:
        metrics["candidate_replay/score_task_loss_fallback"] = 1.0
        metrics["candidate_replay/score_fallback_used"] = 1.0
        return task_loss.detach().float(), metrics

    selected_logits = shifted_logits[score_mask]
    selected_labels = shifted_labels[score_mask]
    score_loss = F.cross_entropy(selected_logits, selected_labels)
    metrics["candidate_replay/score_positions"] = float(num_positions)
    metrics["candidate_replay/score_fallback_used"] = fallback_used
    metrics["candidate_replay/score_task_loss_fallback"] = 0.0
    return score_loss.detach().float(), metrics


def _candidate_replay_answer_margin_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    task_loss: torch.Tensor,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Score a forced candidate by correct-option margin over wrong options."""
    metrics: Dict[str, float] = {
        "candidate_replay/score_positions": 0.0,
        "candidate_replay/score_task_loss_fallback": 0.0,
        "candidate_replay/score_fallback_used": 0.0,
    }
    if logits.dim() != 3 or labels.dim() != 2:
        raise ValueError("Expected logits [batch, seq, vocab] and labels [batch, seq].")
    if labels.size(1) != logits.size(1):
        labels = labels[:, -logits.size(1) :]
    if logits.size(1) < 2:
        metrics["candidate_replay/score_task_loss_fallback"] = 1.0
        metrics["candidate_replay/score_fallback_used"] = 1.0
        return -task_loss.detach().float(), metrics

    option_ids = torch.tensor(
        [int(token_id) for token_id in config.get("option_token_ids", [])],
        dtype=torch.long,
        device=labels.device,
    )
    if option_ids.numel() < 2:
        metrics["candidate_replay/score_task_loss_fallback"] = 1.0
        metrics["candidate_replay/score_fallback_used"] = 1.0
        return -task_loss.detach().float(), metrics

    shifted_logits = logits[:, :-1, :].float()
    shifted_labels = labels[:, 1:]
    valid_mask = shifted_labels.ne(-100)
    option_mask = valid_mask & shifted_labels.unsqueeze(-1).eq(option_ids).any(dim=-1)
    if not bool(option_mask.any().detach().cpu().item()):
        metrics["candidate_replay/score_task_loss_fallback"] = 1.0
        metrics["candidate_replay/score_fallback_used"] = 1.0
        return -task_loss.detach().float(), metrics

    selected_logits = shifted_logits[option_mask]
    selected_labels = shifted_labels[option_mask]
    option_log_probs = F.log_softmax(selected_logits, dim=-1).index_select(
        dim=-1,
        index=option_ids,
    )
    correct_option_mask = selected_labels.unsqueeze(-1).eq(option_ids)
    correct_log_probs = option_log_probs[correct_option_mask]
    wrong_log_probs = option_log_probs.masked_fill(
        correct_option_mask,
        -torch.finfo(option_log_probs.dtype).max,
    )
    wrong_logsumexp = torch.logsumexp(wrong_log_probs, dim=-1)
    margins = correct_log_probs - wrong_logsumexp
    utility = margins.mean().detach().float()

    metrics["candidate_replay/score_positions"] = float(
        option_mask.sum().detach().cpu().item()
    )
    metrics["candidate_replay/answer_margin"] = float(utility.cpu().item())
    metrics["candidate_replay/answer_margin_min"] = float(
        margins.min().detach().float().cpu().item()
    )
    metrics["candidate_replay/answer_margin_max"] = float(
        margins.max().detach().float().cpu().item()
    )
    return utility, metrics


def _candidate_replay_target_from_forced_ranks(
    model: nn.Module,
    kv_cache_index: List[torch.Tensor],
    input_ids: Union[torch.Tensor, List[torch.Tensor]],
    attention_mask: Union[torch.Tensor, List[torch.Tensor]],
    position_ids: torch.Tensor,
    labels: torch.Tensor,
    soft_alignment: Optional[List[Dict[str, torch.Tensor]]],
    config: Dict[str, Any],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
    if not config or not config.get("enabled", False):
        return None, None, {}

    num_candidates = int(config.get("num_candidates", 0))
    if num_candidates <= 0:
        return None, None, {}

    was_training = model.training
    rank_losses: List[torch.Tensor] = []
    rank_utilities: List[torch.Tensor] = []
    rank_score_metrics: List[Dict[str, float]] = []
    try:
        model.eval()
        _set_learned_alignment_replay_scoring_mode(model, True)
        with torch.no_grad():
            for rank in range(num_candidates):
                _set_learned_alignment_forced_rank(model, rank)
                outputs = model.forward(
                    kv_cache_index=kv_cache_index,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    labels=labels,
                    soft_alignment=soft_alignment,
                    use_cache=True,
                )
                score_mode = str(config.get("score_mode", "task_loss"))
                if score_mode == "task_loss":
                    score_loss = outputs.loss.detach().float()
                    score_metrics = {
                        "candidate_replay/score_positions": 0.0,
                        "candidate_replay/score_fallback_used": 0.0,
                        "candidate_replay/score_task_loss_fallback": 1.0,
                    }
                    rank_utilities.append(-score_loss)
                else:
                    final_labels = _final_section_labels(
                        labels=labels,
                        kv_cache_index=kv_cache_index,
                        output_seq_len=outputs.logits.size(1),
                    )
                    if score_mode == "answer_margin":
                        utility, score_metrics = (
                            _candidate_replay_answer_margin_from_logits(
                                logits=outputs.logits,
                                labels=final_labels,
                                task_loss=outputs.loss,
                                config=config,
                            )
                        )
                        score_loss = -utility
                        rank_utilities.append(utility)
                    else:
                        score_loss, score_metrics = (
                            _candidate_replay_score_loss_from_logits(
                                logits=outputs.logits,
                                labels=final_labels,
                                task_loss=outputs.loss,
                                config=config,
                            )
                        )
                        rank_utilities.append(-score_loss)
                if score_mode == "answer_margin":
                    score_metrics["candidate_replay/utility"] = float(
                        rank_utilities[-1].detach().float().cpu().item()
                    )
                elif "candidate_replay/utility" not in score_metrics:
                    score_metrics["candidate_replay/utility"] = float(
                        (-score_loss).detach().float().cpu().item()
                    )
                rank_losses.append(score_loss)
                rank_score_metrics.append(score_metrics)
    finally:
        _set_learned_alignment_replay_scoring_mode(model, False)
        _set_learned_alignment_forced_rank(model, None)
        if was_training:
            model.train()
        else:
            model.eval()

    if not rank_losses:
        return None, None, {}

    losses = torch.stack(rank_losses)
    utilities = torch.stack(rank_utilities) if rank_utilities else -losses
    target = _normalized_candidate_replay_target(
        losses=losses,
        temperature=float(config.get("target_temperature", 1.0)),
        normalize=bool(config.get("score_normalize", True)),
        clip=float(config.get("score_clip", 3.0)),
    )

    top2 = torch.topk(target, k=min(2, target.numel())).values
    margin = (
        float((top2[0] - top2[1]).detach().cpu().item()) if top2.numel() > 1 else 0.0
    )
    entropy = -(target * target.clamp_min(torch.finfo(target.dtype).eps).log()).sum()
    denom = math.log(max(2, target.numel()))
    sorted_losses = losses.topk(k=min(2, losses.numel()), largest=False).values
    loss_margin = (
        float((sorted_losses[1] - sorted_losses[0]).cpu().item())
        if sorted_losses.numel() > 1
        else 0.0
    )
    metrics = {
        "candidate_replay/loss_min": float(losses.min().cpu().item()),
        "candidate_replay/loss_max": float(losses.max().cpu().item()),
        "candidate_replay/loss_margin": loss_margin,
        "candidate_replay/utility_min": float(utilities.min().cpu().item()),
        "candidate_replay/utility_max": float(utilities.max().cpu().item()),
        "candidate_replay/utility_margin": (
            float(
                (
                    utilities.topk(k=min(2, utilities.numel())).values[0]
                    - utilities.topk(k=min(2, utilities.numel())).values[-1]
                )
                .cpu()
                .item()
            )
            if utilities.numel() > 1
            else 0.0
        ),
        "candidate_replay/target_anchor": float(target[0].cpu().item()),
        "candidate_replay/target_top1": float(target.max().cpu().item()),
        "candidate_replay/target_entropy": float((entropy / denom).cpu().item()),
        "candidate_replay/target_margin": margin,
        "candidate_replay/best_rank": float(target.argmax().cpu().item()),
    }
    if rank_score_metrics:
        metric_keys = sorted(
            {key for metric in rank_score_metrics for key in metric.keys()}
        )
        for key in metric_keys:
            metrics[key] = sum(
                metric.get(key, 0.0) for metric in rank_score_metrics
            ) / len(rank_score_metrics)
    for rank, loss_value in enumerate(losses):
        metrics[f"candidate_replay/loss_rank_{rank}"] = float(loss_value.cpu().item())
    for rank, utility_value in enumerate(utilities):
        metrics[f"candidate_replay/utility_rank_{rank}"] = float(
            utility_value.cpu().item()
        )

    return target, utilities, metrics


def _cached_candidate_replay_target_from_batch(
    batch: Dict[str, Any],
    device: str,
    config: Dict[str, Any],
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Dict[str, float],
]:
    cached = batch.get("candidate_replay")
    if not cached:
        return None, None, None, {}

    cache_hit = cached.get("cache_hit")
    if cache_hit is None:
        return None, None, None, {}
    cache_hit = cache_hit.to(device=device, dtype=torch.bool)
    target = cached["target"].to(device=device, dtype=torch.float32)
    utility = cached["utility"].to(device=device, dtype=torch.float32)
    utility_valid = cached.get("utility_valid")
    if utility_valid is None:
        utility_valid = torch.ones_like(utility, dtype=torch.bool)
    else:
        utility_valid = utility_valid.to(device=device, dtype=torch.bool)

    valid_rows = cache_hit & (target.sum(dim=-1) > 0)
    if not bool(valid_rows.any().detach().cpu().item()):
        return (
            None,
            None,
            None,
            {
                "candidate_replay/cache_hit_rate": float(
                    cache_hit.float().mean().detach().cpu().item()
                ),
                "candidate_replay/cache_selected_rate": 0.0,
            },
        )

    target = torch.where(valid_rows[:, None], target, torch.zeros_like(target))
    utility = torch.where(valid_rows[:, None], utility, torch.zeros_like(utility))
    utility_valid = utility_valid & valid_rows[:, None]

    top2 = torch.topk(target, k=min(2, target.size(-1)), dim=-1).values
    target_margin = (
        top2[:, 0] - top2[:, 1] if top2.size(-1) > 1 else torch.zeros_like(top2[:, 0])
    )
    target_entropy = -(
        target * target.clamp_min(float(config.get("target_eps", 1e-8))).log()
    ).sum(dim=-1)
    target_entropy = target_entropy / math.log(max(2, target.size(-1)))

    masked_utility = utility.masked_fill(
        ~utility_valid, -torch.finfo(utility.dtype).max
    )
    utility_top2 = torch.topk(
        masked_utility,
        k=min(2, masked_utility.size(-1)),
        dim=-1,
    ).values
    utility_margin = (
        utility_top2[:, 0] - utility_top2[:, 1]
        if utility_top2.size(-1) > 1
        else torch.zeros_like(utility_top2[:, 0])
    )
    selected = valid_rows
    min_target_margin = float(config.get("min_target_margin", 0.0))
    min_utility_margin = float(config.get("min_utility_margin", 0.0))
    if min_target_margin > 0:
        selected = selected & (target_margin >= min_target_margin)
    if min_utility_margin > 0:
        selected = selected & (utility_margin >= min_utility_margin)

    target = torch.where(selected[:, None], target, torch.zeros_like(target))
    utility = torch.where(selected[:, None], utility, torch.zeros_like(utility))
    utility_valid = utility_valid & selected[:, None]

    selected_float = selected.float()
    selected_denom = selected_float.sum().clamp_min(1.0)
    metrics = {
        "candidate_replay/cache_hit_rate": float(
            cache_hit.float().mean().detach().cpu().item()
        ),
        "candidate_replay/cache_selected_rate": float(
            selected.float().mean().detach().cpu().item()
        ),
        "candidate_replay/target_anchor": float(
            ((target[:, 0] * selected_float).sum() / selected_denom)
            .detach()
            .cpu()
            .item()
        ),
        "candidate_replay/target_top1": float(
            ((target.max(dim=-1).values * selected_float).sum() / selected_denom)
            .detach()
            .cpu()
            .item()
        ),
        "candidate_replay/target_entropy": float(
            ((target_entropy * selected_float).sum() / selected_denom)
            .detach()
            .cpu()
            .item()
        ),
        "candidate_replay/target_margin": float(
            ((target_margin * selected_float).sum() / selected_denom)
            .detach()
            .cpu()
            .item()
        ),
        "candidate_replay/utility_margin": float(
            ((utility_margin * selected_float).sum() / selected_denom)
            .detach()
            .cpu()
            .item()
        ),
        "candidate_replay/best_rank": float(
            ((target.argmax(dim=-1).float() * selected_float).sum() / selected_denom)
            .detach()
            .cpu()
            .item()
        ),
    }
    return target, utility, utility_valid, metrics


def detect_training_mode(model_config: Dict[str, Any]) -> str:
    """Detect whether to use baseline or Rosetta training based on config"""
    if "baseline_model" in model_config and "base_model" not in model_config:
        return "baseline"
    elif "base_model" in model_config and "teacher_model" in model_config:
        return "rosetta"
    else:
        raise ValueError(
            "Invalid model configuration. Provide either 'baseline_model' for baseline training "
            "or both 'base_model' and 'teacher_model' for Rosetta training."
        )


def setup_lora_model(model: nn.Module, lora_config: Dict[str, Any]) -> nn.Module:
    """Setup LoRA for the model"""
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library is required for LoRA training. Install with: pip install peft"
        )

    # Default LoRA configuration
    default_config = {
        "r": 16,  # LoRA rank
        "lora_alpha": 32,  # LoRA scaling parameter
        "target_modules": [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }

    # Update with user config
    config = {**default_config, **lora_config}

    # Create LoRA config
    peft_config = LoraConfig(
        r=config["r"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
        bias=config["bias"],
        task_type=getattr(TaskType, config["task_type"]),
    )

    # Apply LoRA to model
    model = get_peft_model(model, peft_config)

    return model


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file based on file extension"""
    file_ext = os.path.splitext(config_path)[1].lower()

    with open(config_path, "r") as f:
        if file_ext == ".json":
            config = json.load(f)
        elif file_ext in [".yaml", ".yml"]:
            config = yaml.safe_load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {file_ext}. Supported formats: .json, .yaml, .yml"
            )

    return config


def setup_partial_training(
    model: nn.Module, partial_config: Dict[str, Any]
) -> nn.Module:
    """Setup partial parameter training (alternative to LoRA)"""
    method = partial_config.get("method", "layer_wise")
    ratio = partial_config.get("ratio", 0.6)  # 60% of parameters

    if method == "layer_wise":
        # Freeze/unfreeze entire layers
        total_layers = len(model.model.layers)
        layers_to_train = int(total_layers * ratio)

        # Freeze all parameters first
        freeze_model(model)

        # Unfreeze the last N layers
        for i in range(total_layers - layers_to_train, total_layers):
            unfreeze_model(model.model.layers[i])

        # Also unfreeze the final layer norm and lm_head
        if hasattr(model.model, "norm"):
            unfreeze_model(model.model.norm)
        if hasattr(model, "lm_head"):
            unfreeze_model(model.lm_head)

        print(
            f"Training last {layers_to_train} layers out of {total_layers} total layers"
        )

    elif method == "parameter_wise":
        # Freeze/unfreeze based on parameter importance or random selection
        all_params = list(model.named_parameters())
        total_params = len(all_params)
        params_to_train = int(total_params * ratio)

        # Freeze all first
        freeze_model(model)

        # Unfreeze last N parameters (you can implement more sophisticated selection)
        for name, param in all_params[-params_to_train:]:
            param.requires_grad = True

        print(
            f"Training {params_to_train} parameters out of {total_params} total parameters"
        )

    return model


def build_layer_mapping(n_target=28, n_source=36):

    source_positions = [i / (n_source - 1) for i in range(n_source)]
    target_positions = [j / (n_target - 1) for j in range(n_target)]

    mapping = []
    for i, sp in enumerate(target_positions):
        closest_j = min(range(n_source), key=lambda j: abs(source_positions[j] - sp))
        mapping.append((i, closest_j))

    return mapping


def build_shared_mlp(
    source_dim: int,
    hidden_dim: int,
    target_dim: int,
    num_layers: int,
    use_layer_norm: bool,
    dropout: float,
    dtype: torch.dtype,
) -> nn.Sequential:
    """Build a single MLP projection module"""
    layers = []

    # Input projection
    layers.append(nn.Linear(source_dim, hidden_dim, dtype=dtype))
    if use_layer_norm:
        layers.append(nn.LayerNorm(hidden_dim, dtype=dtype))
    layers.append(nn.GELU())
    layers.append(nn.Dropout(dropout))

    # Hidden layers
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim, dtype=dtype))
        if use_layer_norm:
            layers.append(nn.LayerNorm(hidden_dim, dtype=dtype))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))

    # Output projection
    if num_layers > 1:
        layers.append(nn.Linear(hidden_dim, target_dim, dtype=dtype))
    else:
        # Single layer case
        layers = [nn.Linear(source_dim, target_dim, dtype=dtype)]

    return nn.Sequential(*layers)


def _load_projector_checkpoint_dir(
    projector_list: List[nn.Module],
    checkpoint_dir: str,
    strict: bool = True,
) -> None:
    """Load per-layer projector state dicts from an existing checkpoint dir."""
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(
            f"projector_checkpoint_dir does not exist: {checkpoint_dir}"
        )

    for projector_idx, projector in enumerate(projector_list):
        state_path = os.path.join(checkpoint_dir, f"projector_{projector_idx}.pt")
        if not os.path.isfile(state_path):
            raise FileNotFoundError(
                f"Missing projector checkpoint for index {projector_idx}: {state_path}"
            )
        state_dict = torch.load(state_path, map_location="cpu")
        incompatible = projector.load_state_dict(state_dict, strict=strict)
        if not strict and (incompatible.missing_keys or incompatible.unexpected_keys):
            print(
                "Loaded projector "
                f"{projector_idx} with missing={incompatible.missing_keys} "
                f"unexpected={incompatible.unexpected_keys}"
            )

    print(f"Loaded {len(projector_list)} projectors from {checkpoint_dir}")


def setup_models(
    model_config: Dict[str, Any],
    training_mode: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Setup models based on training mode (baseline or rosetta)"""

    if training_mode == "baseline":
        # Baseline mode: single model training
        model_name = model_config["baseline_model"]
        model_path = resolve_model_path(model_name)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        set_default_chat_template(tokenizer, model_name)

        # Load baseline model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation=model_config.get("attn_implementation", None),
        )

        return model, tokenizer, None, None

    else:  # rosetta mode
        base_model_name = model_config["base_model"]
        teacher_model_name = model_config["teacher_model"]
        base_model_path = resolve_model_path(base_model_name)
        teacher_model_path = resolve_model_path(teacher_model_name)

        # Load tokenizer (use base model tokenizer)
        slm_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

        if slm_tokenizer.pad_token is None:
            slm_tokenizer.pad_token = slm_tokenizer.eos_token
            slm_tokenizer.pad_token_id = slm_tokenizer.eos_token_id
        set_default_chat_template(slm_tokenizer, base_model_name)

        # Load LLM tokenizer if alignment is enabled
        llm_tokenizer = None
        if model_config.get("is_do_alignment", False):
            llm_tokenizer = AutoTokenizer.from_pretrained(teacher_model_path)
            if llm_tokenizer.pad_token is None:
                llm_tokenizer.pad_token = llm_tokenizer.eos_token
                llm_tokenizer.pad_token_id = llm_tokenizer.eos_token_id
            set_default_chat_template(llm_tokenizer, teacher_model_name)

        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            attn_implementation=model_config.get("attn_implementation", None),
        )

        # Load teacher model
        if model_matches(teacher_model_name, "gemma-3-1b-it"):
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_path,
                torch_dtype=dtype,
                attn_implementation=model_config.get("attn_implementation", None),
                sliding_window=4096,
            )
        else:
            teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model_path,
                torch_dtype=dtype,
                attn_implementation=model_config.get("attn_implementation", None),
            )

        # Get model dimensions and layer counts
        base_dim = int(
            base_model.model.layers[0].self_attn.k_proj.out_features
            / base_model.config.num_key_value_heads
        )
        teacher_dim = int(
            teacher_model.model.layers[0].self_attn.k_proj.out_features
            / teacher_model.config.num_key_value_heads
        )
        base_num_heads = base_model.config.num_key_value_heads
        teacher_num_heads = teacher_model.config.num_key_value_heads
        slm_num_layers = base_model.config.num_hidden_layers
        llm_num_layers = teacher_model.config.num_hidden_layers

        # Create projector from config
        projector_config = model_config["projector"]
        projector_params = projector_config["params"].copy()
        projector_params["dtype"] = dtype
        projector_list = []
        # Only M projectors (share projector across sources): one per target layer
        num_projectors = slm_num_layers

        # shared_key_projection=build_shared_mlp(
        #     source_dim=teacher_dim,
        #     hidden_dim=projector_params["hidden_dim"],
        #     target_dim=base_dim,
        #     num_layers=projector_params["num_layers"],
        #     use_layer_norm=projector_params["use_layer_norm"],
        #     dropout=projector_params["dropout"],
        #     dtype=dtype
        # )
        # shared_value_projection=build_shared_mlp(
        #     source_dim=teacher_dim,
        #     hidden_dim=projector_params["hidden_dim"],
        #     target_dim=base_dim,
        #     num_layers=projector_params["num_layers"],
        #     use_layer_norm=projector_params["use_layer_norm"],
        #     dropout=projector_params["dropout"],
        #     dtype=dtype
        # )
        for target_layer_idx in range(num_projectors):
            layer_projector_params = projector_params.copy()
            if (
                layer_projector_params.get(
                    "alignment_confidence_layer_scale_mode", "none"
                )
                != "none"
            ):
                layer_projector_params.setdefault(
                    "alignment_confidence_num_layers", num_projectors
                )
                layer_projector_params["alignment_confidence_layer_idx"] = (
                    target_layer_idx
                )
            projector = create_projector(
                projector_config["type"],
                source_dim=teacher_dim,
                target_dim=base_dim,
                source_num_heads=teacher_num_heads,
                target_num_heads=base_num_heads,
                # shared_key_projection=shared_key_projection,
                # shared_value_projection=shared_value_projection,
                **layer_projector_params,
            )
            projector_list.append(projector.to(device))

        projector_checkpoint_dir = model_config.get("projector_checkpoint_dir")
        if projector_checkpoint_dir:
            _load_projector_checkpoint_dir(
                projector_list,
                projector_checkpoint_dir,
                strict=bool(model_config.get("projector_checkpoint_strict", True)),
            )

        # Init RosettaModel
        K = 1

        rosetta_model = (
            RosettaModel(
                model_list=[base_model, teacher_model],
                base_model_idx=0,
                projector_list=projector_list,
                include_response=model_config.get("include_response", False),
                multi_source_fusion_mode=model_config.get(
                    "multi_source_fusion_mode", "sequential"
                ),
                fpct_operator=model_config.get("fpct_operator"),
                fpct_replicated_collapse=model_config.get(
                    "fpct_replicated_collapse", False
                ),
                fpct_instrumentation=model_config.get(
                    "fpct_instrumentation", False
                ),
            )
            .to(device)
            .eval()
        )

        # mapping stretegy
        if model_config["mapping"] == "last_aligned":
            source_target_mapping = last_aligned_sources(
                slm_num_layers, llm_num_layers, K
            )
        elif model_config["mapping"] == "k_nearest":
            source_target_mapping = k_nearest_sources(slm_num_layers, llm_num_layers, K)
        else:
            raise ValueError(f"Invalid mapping strategy: {model_config['mapping']}")
        print(f"Using {model_config['mapping']} mapping strategy (target: [sources])")

        # set projector
        for target_layer_idx, src_list in source_target_mapping.items():
            for source_layer_idx in src_list:
                rosetta_model.set_projector_config(
                    source_model_idx=1,  # Teacher model
                    source_model_layer_idx=source_layer_idx,
                    target_model_idx=0,  # Base model
                    target_model_layer_idx=target_layer_idx,
                    projector_idx=target_layer_idx,  # share projector per target layer
                )

        # Optional aligner construction (used by collator)
        aligner = None
        if model_config.get("is_do_alignment", False):
            # Build tokenizers for both models
            strategy = model_config.get("alignment_strategy", "first")
            aligner = TokenAligner(
                slm_tokenizer=slm_tokenizer,
                llm_tokenizer=llm_tokenizer,
                strategy=AlignmentStrategy(strategy),
                soft_alignment_score_mode=model_config.get(
                    "soft_alignment_score_mode", "overlap"
                ),
                soft_alignment_boundary_bonus=model_config.get(
                    "soft_alignment_boundary_bonus", 0.0
                ),
                soft_alignment_boundary_tolerance=model_config.get(
                    "soft_alignment_boundary_tolerance", 1
                ),
                soft_alignment_min_weight=model_config.get(
                    "soft_alignment_min_weight", 0.0
                ),
                soft_alignment_confidence_mode=model_config.get(
                    "soft_alignment_confidence_mode", "none"
                ),
                soft_alignment_confidence_alpha=model_config.get(
                    "soft_alignment_confidence_alpha", 0.5
                ),
                soft_alignment_confidence_floor=model_config.get(
                    "soft_alignment_confidence_floor", 0.0
                ),
                soft_alignment_fallback_confidence=model_config.get(
                    "soft_alignment_fallback_confidence", 1.0
                ),
                soft_alignment_confidence_control_mode=model_config.get(
                    "soft_alignment_confidence_control_mode", "native"
                ),
                soft_alignment_confidence_constant_value=model_config.get(
                    "soft_alignment_confidence_constant_value"
                ),
                soft_alignment_confidence_shuffle_seed=model_config.get(
                    "soft_alignment_confidence_shuffle_seed", 0
                ),
                soft_alignment_reweight_mode=model_config.get(
                    "soft_alignment_reweight_mode", "none"
                ),
                soft_alignment_reweight_strength=model_config.get(
                    "soft_alignment_reweight_strength", 1.0
                ),
                soft_alignment_reweight_power=model_config.get(
                    "soft_alignment_reweight_power", 2.0
                ),
                soft_alignment_candidate_window=model_config.get(
                    "soft_alignment_candidate_window", 0
                ),
                learned_alignment_prior_mode=model_config.get(
                    "learned_alignment_prior_mode", "anchor"
                ),
            )

        return rosetta_model, slm_tokenizer, aligner, llm_tokenizer


def train_step(
    model: nn.Module,
    batch: Dict[str, Any],
    tokenizer: AutoTokenizer,
    max_length: int,
    device: str,
    training_mode: str,
    include_auxiliary_loss: bool = True,
    answer_prior_config: Optional[Dict[str, Any]] = None,
    answer_margin_config: Optional[Dict[str, Any]] = None,
    candidate_replay_config: Optional[Dict[str, Any]] = None,
):
    """Single training step for both baseline and Rosetta models"""
    train_step.last_answer_prior_metrics = {}
    train_step.last_answer_margin_metrics = {}
    train_step.last_candidate_replay_metrics = {}

    if training_mode == "baseline":
        # Baseline model training
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass for baseline model
        outputs = model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        loss = outputs.loss
        labels_for_prior = labels

    else:  # rosetta mode
        # Rosetta model training
        if isinstance(batch["input_ids"], list):
            input_ids = [sample_ids.to(device) for sample_ids in batch["input_ids"]]
            attention_mask = [
                sample_attention_mask.to(device)
                for sample_attention_mask in batch["attention_mask"]
            ]
        else:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

        position_ids = batch["position_ids"].to(device)
        labels = batch["labels"].to(device)
        kv_cache_index = [x.to(device) for x in batch["kv_cache_index"]]
        soft_alignment = None
        if "soft_alignment" in batch:
            soft_alignment = [
                {
                    "source_indices": section["source_indices"].to(device),
                    "source_weights": section["source_weights"].to(device),
                    "source_confidence": section.get(
                        "source_confidence",
                        torch.ones_like(
                            section["source_indices"][..., 0],
                            dtype=torch.float,
                        ),
                    ).to(device),
                    "source_entropy": section.get(
                        "source_entropy",
                        torch.zeros_like(
                            section["source_indices"][..., 0],
                            dtype=torch.float,
                        ),
                    ).to(device),
                    "source_entropy_override": section.get(
                        "source_entropy_override",
                        torch.zeros_like(
                            section["source_indices"][..., 0],
                            dtype=torch.bool,
                        ),
                    ).to(device),
                }
                for section in batch["soft_alignment"]
            ]

        _set_learned_alignment_replay_target(model, None)
        _set_learned_alignment_replay_utility(model, None)
        _set_learned_alignment_replay_utility_valid(model, None)
        if include_auxiliary_loss and candidate_replay_config:
            replay_utility_valid = None
            if candidate_replay_config.get("cache_mode", "online") == "cached":
                (
                    replay_target,
                    replay_utility,
                    replay_utility_valid,
                    replay_metrics,
                ) = _cached_candidate_replay_target_from_batch(
                    batch=batch,
                    device=device,
                    config=candidate_replay_config,
                )
            else:
                replay_target, replay_utility, replay_metrics = (
                    _candidate_replay_target_from_forced_ranks(
                        model=model,
                        kv_cache_index=kv_cache_index,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        labels=labels,
                        soft_alignment=soft_alignment,
                        config=candidate_replay_config,
                    )
                )
            train_step.last_candidate_replay_metrics = replay_metrics
            if replay_target is not None:
                _set_learned_alignment_replay_target(model, replay_target)
            if replay_utility is not None:
                _set_learned_alignment_replay_utility(model, replay_utility)
            if replay_utility_valid is not None:
                _set_learned_alignment_replay_utility_valid(
                    model,
                    replay_utility_valid,
                )

        try:
            # Forward pass for Rosetta model
            outputs = model.forward(
                kv_cache_index=kv_cache_index,
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels,
                soft_alignment=soft_alignment,
                use_cache=True,
            )

            loss = outputs.loss
            task_loss_for_likelihood = outputs.loss
            labels_for_prior = _final_section_labels(
                labels=labels,
                kv_cache_index=kv_cache_index,
                output_seq_len=outputs.logits.size(1),
            )

            if include_auxiliary_loss:
                auxiliary_loss = _projector_auxiliary_loss(model)
                if auxiliary_loss is not None:
                    loss = loss + auxiliary_loss
                likelihood_auxiliary_loss = _projector_likelihood_auxiliary_loss(
                    model,
                    task_loss_for_likelihood,
                )
                if likelihood_auxiliary_loss is not None:
                    loss = loss + likelihood_auxiliary_loss
        finally:
            _set_learned_alignment_replay_target(model, None)
            _set_learned_alignment_replay_utility(model, None)
            _set_learned_alignment_replay_utility_valid(model, None)

    if answer_prior_config and answer_prior_config.get("enabled", False):
        prior_output = compute_answer_prior_regularization(
            logits=outputs.logits,
            labels=labels_for_prior,
            option_token_ids=answer_prior_config["option_token_ids"],
            weight=answer_prior_config.get("weight", 0.0),
            target=answer_prior_config.get("target", "smoothed_gold"),
            label_smoothing=answer_prior_config.get("label_smoothing", 0.7),
            temperature=answer_prior_config.get("temperature", 1.0),
            min_positions=answer_prior_config.get("min_positions", 1),
        )
        train_step.last_answer_prior_metrics = prior_output.metrics
        if prior_output.loss is not None:
            loss = loss + prior_output.loss

    if answer_margin_config and answer_margin_config.get("enabled", False):
        margin_output = compute_answer_margin_routing_loss(
            logits=outputs.logits,
            labels=labels_for_prior,
            option_token_ids=answer_margin_config["option_token_ids"],
            weight=answer_margin_config.get("weight", 0.0),
            objective=answer_margin_config.get("objective", "hinge"),
            margin=answer_margin_config.get("margin", 0.5),
            temperature=answer_margin_config.get("temperature", 1.0),
            ce_weight=answer_margin_config.get("ce_weight", 1.0),
            hinge_weight=answer_margin_config.get("hinge_weight", 1.0),
            min_positions=answer_margin_config.get("min_positions", 1),
        )
        train_step.last_answer_margin_metrics = margin_output.metrics
        if margin_output.loss is not None:
            loss = loss + margin_output.loss

    return loss


def evaluate_model(
    model: nn.Module,
    eval_loader: DataLoader,
    tokenizer: AutoTokenizer,
    max_length: int,
    device: str,
    training_mode: str,
) -> float:
    """Evaluate the model and return average loss"""
    model.eval()
    eval_loss_total = 0.0
    num_batches = 0

    with torch.no_grad():
        for eval_batch in eval_loader:
            eval_loss = train_step(
                model,
                eval_batch,
                tokenizer,
                max_length,
                device,
                training_mode,
                include_auxiliary_loss=False,
            )
            eval_loss_total += eval_loss.item()
            num_batches += 1

    avg_eval_loss = eval_loss_total / num_batches if num_batches > 0 else 0.0
    model.train()  # Set back to train mode
    return avg_eval_loss


def main():
    """
    Train a model (Rosetta or baseline) using hyper-parameters defined in a JSON
    or YAML configuration file. The mode is automatically detected from the config:
    - If 'baseline_model' is provided: baseline training
    - If 'base_model' and 'teacher_model' are provided: Rosetta training
    Training progress is tracked with Weights & Biases and the original config
    is copied alongside checkpoints for full reproducibility.
    """

    # ------------------------------------------------------------------
    # Configuration loading
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="Train RosettaModel from a JSON or YAML config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="recipe/all_in_one.yaml",
        help="Path to JSON or YAML config file",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="Local rank for distributed training"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save outputs and checkpoints",
    )
    parser.add_argument(
        "--eval_only", action="store_true", help="Run evaluation only (no training)"
    )
    args = parser.parse_args()

    cfg: Dict[str, Any] = load_config(args.config)

    # Extract configuration sections
    model_config = cfg["model"]
    training_config = cfg["training"]
    output_config = cfg["output"]
    data_config = cfg["data"]

    # Set seed for reproducibility and enable stricter determinism
    set_seed(seed=training_config["seed"])
    enable_full_determinism()

    # Create datetime subfolder under output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # timestamped_output_dir = os.path.join(output_config["output_dir"], "3")
    timestamped_output_dir = output_config["output_dir"]
    # timestamped_output_dir = args.output_dir

    # Ensure output directory exists and copy config for reproducibility
    os.makedirs(timestamped_output_dir, exist_ok=True)
    shutil.copy(args.config, os.path.join(timestamped_output_dir, "config.json"))

    # ------------------------------------------------------------------
    # Distributed training setup
    # ------------------------------------------------------------------
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    distributed = world_size > 1
    local_rank = args.local_rank

    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
    else:
        rank = 0
        local_rank = 0
        device = training_config.get("device", "cuda")

    is_main_process = rank == 0

    # ------------------------------------------------------------------
    # Weights & Biases initialisation
    # ------------------------------------------------------------------
    run_name = f"{output_config['wandb_config']['run_name']}_{timestamp}"
    if is_main_process:
        wandb.init(
            project=output_config["wandb_config"]["project"],
            name=run_name,
            config=cfg,
            mode=output_config["wandb_config"]["mode"],
            entity=output_config["wandb_config"]["entity"],
        )

    print(f"Outputs will be saved to: {timestamped_output_dir}")

    # ------------------------------------------------------------------
    # Detect training mode and setup models
    # ------------------------------------------------------------------
    training_mode = detect_training_mode(model_config)
    if is_main_process:
        print(f"Training mode: {training_mode}")
        print("Setting up models…")

    model, main_tokenizer, aligner, llm_tokenizer = setup_models(
        model_config, training_mode, device, torch.bfloat16
    )
    model = model.to(device)

    answer_prior_config = training_config.get("answer_prior_regularization")
    if answer_prior_config and answer_prior_config.get("enabled", False):
        answer_prior_config = dict(answer_prior_config)
        if "option_token_ids" not in answer_prior_config:
            answer_prior_config["option_token_ids"] = resolve_option_token_ids(
                main_tokenizer,
                option_labels=answer_prior_config.get(
                    "option_labels", ["A", "B", "C", "D"]
                ),
                option_token_texts=answer_prior_config.get("option_token_texts"),
            )
        if is_main_process:
            print(
                "Answer-prior regularization enabled: "
                f"target={answer_prior_config.get('target', 'smoothed_gold')}, "
                f"weight={answer_prior_config.get('weight', 0.0)}, "
                f"token_ids={answer_prior_config['option_token_ids']}"
            )
    else:
        answer_prior_config = None

    answer_margin_config = training_config.get("answer_margin_routing")
    if answer_margin_config and answer_margin_config.get("enabled", False):
        answer_margin_config = dict(answer_margin_config)
        if "option_token_ids" not in answer_margin_config:
            answer_margin_config["option_token_ids"] = resolve_option_token_ids(
                main_tokenizer,
                option_labels=answer_margin_config.get(
                    "option_labels", ["A", "B", "C", "D"]
                ),
                option_token_texts=answer_margin_config.get("option_token_texts"),
            )
        if is_main_process:
            print(
                "Answer-margin routing enabled: "
                f"objective={answer_margin_config.get('objective', 'hinge')}, "
                f"weight={answer_margin_config.get('weight', 0.0)}, "
                f"margin={answer_margin_config.get('margin', 0.5)}, "
                f"token_ids={answer_margin_config['option_token_ids']}"
            )
    else:
        answer_margin_config = None

    candidate_replay_config = training_config.get("candidate_replay_alignment")
    if candidate_replay_config and candidate_replay_config.get("enabled", False):
        candidate_replay_config = dict(candidate_replay_config)
        score_mode = str(candidate_replay_config.get("score_mode", "task_loss"))
        if (
            score_mode in {"answer_token_ce", "answer_suffix_ce", "answer_margin"}
            and "option_token_ids" not in candidate_replay_config
        ):
            candidate_replay_config["option_token_ids"] = resolve_option_token_ids(
                main_tokenizer,
                option_labels=candidate_replay_config.get(
                    "option_labels", ["A", "B", "C", "D"]
                ),
                option_token_texts=candidate_replay_config.get("option_token_texts"),
            )
        if is_main_process:
            print(
                "Candidate replay alignment enabled: "
                f"num_candidates={candidate_replay_config.get('num_candidates', 0)}, "
                f"temperature={candidate_replay_config.get('target_temperature', 1.0)}, "
                f"score_mode={score_mode}"
            )
    else:
        candidate_replay_config = None

    # Apply freezing/training configuration based on mode
    if training_mode == "baseline":
        # Check for LoRA or partial training configuration
        lora_config = training_config.get("lora", None)
        partial_config = training_config.get("partial_training", None)

        if lora_config is not None:
            if is_main_process:
                print("Setting up LoRA training...")
            model = setup_lora_model(model, lora_config)
            if is_main_process:
                print("LoRA setup completed")
        elif partial_config is not None:
            if is_main_process:
                print("Setting up partial parameter training...")
            model = setup_partial_training(model, partial_config)
            if is_main_process:
                print("Partial training setup completed")
        else:
            # Apply freezing based on configuration
            freeze_config = training_config.get("freeze", [])
            if is_main_process:
                print(f"Applying freeze configuration: {freeze_config}")

            if "baseline" in freeze_config or "base" in freeze_config:
                freeze_model(model)
            else:
                unfreeze_model(model)
    else:  # rosetta mode
        freeze_config = training_config["freeze"]  # including ["base", "teacher"]

        if is_main_process:
            print(f"Applying freeze configuration: {freeze_config}")

        if "base" in freeze_config:
            freeze_model(model.model_list[0])  # Base model
        else:
            unfreeze_model(model.model_list[0])

        if "teacher" in freeze_config:
            freeze_model(model.model_list[1])  # Teacher model
        else:
            unfreeze_model(model.model_list[1])

        if "projector" in freeze_config:
            # Freeze projectors
            for projector in model.projector_list:
                freeze_model(projector)
        else:
            unfreeze_projectors(model)

    # Wrap with DDP if needed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(
        f"Percentage of trainable parameters: {trainable_params / total_params * 100:.4f}%"
    )

    # ------------------------------------------------------------------
    # Dataset & dataloaders
    # ------------------------------------------------------------------
    print("Loading dataset…")
    # Create dataset using the auto-registration system
    instruct_ds = create_dataset(
        dataset_type=data_config["type"], **data_config["kwargs"]
    )

    # Create dataset based on training mode
    if training_mode == "baseline":
        full_dataset = BaselineChatDataset(
            instruct_ds,
            main_tokenizer,
            max_length=training_config.get("max_length", 2048),
        )
    elif training_mode == "rosetta":  # rosetta mode
        if model_config.get("is_do_alignment", False) and aligner is not None:
            full_dataset = AlignedChatDataset(
                instruct_ds,
                aligner,
                max_length=training_config.get("max_length", 2048),
                soft_alignment_top_k=model_config.get("soft_alignment_top_k", 4),
                candidate_replay_cache_path=training_config.get(
                    "candidate_replay_cache_path"
                ),
                fpct_alignment_sanitizer=model_config.get(
                    "fpct_alignment_sanitizer", "none"
                ),
            )
        else:
            full_dataset = ChatDataset(instruct_ds, main_tokenizer)
    else:
        raise ValueError(f"Invalid training mode: {training_mode}")

    train_size = int(data_config["train_ratio"] * len(full_dataset))
    eval_size = len(full_dataset) - train_size
    split_indices_path = data_config.get("split_indices_path")
    resolved_split_path = (
        Path(split_indices_path).expanduser().resolve()
        if split_indices_path
        else None
    )
    # The April 2026 v2.2 checkpoint used the process-global torch RNG for its
    # split, after model/projector initialization. ``legacy_global_rng`` exists
    # only to recover that historical split and freeze its captured indices.
    train_dataset, eval_dataset, split_mode = _create_dataset_split(
        full_dataset,
        train_size,
        eval_size,
        seed=int(training_config["seed"]),
        split_mode=data_config.get("split_mode", "seeded"),
        split_indices_path=resolved_split_path,
    )
    if resolved_split_path is not None and rank == 0:
        print(f"Loaded frozen dataset split: {resolved_split_path}")

    split_indices_output = data_config.get("split_indices_output")
    if split_indices_output and rank == 0:
        resolved_output_path = Path(split_indices_output).expanduser().resolve()
        _write_dataset_split_indices(
            resolved_output_path,
            list(train_dataset.indices),
            list(eval_dataset.indices),
            dataset_size=len(full_dataset),
            seed=int(training_config["seed"]),
            split_mode=split_mode,
        )
        print(f"Saved frozen dataset split: {resolved_output_path}")

    per_device_batch_size = training_config["per_device_train_batch_size"]
    grad_accum_steps = training_config.get("gradient_accumulation_steps", 1)

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, shuffle=True, seed=training_config["seed"]
        )
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset, shuffle=False, seed=training_config["seed"]
        )
    else:
        train_sampler = None
        eval_sampler = None

    # Create collator based on training mode
    if training_mode == "baseline":
        collator = BaselineDataCollator(
            tokenizer=main_tokenizer,
            pad_to_multiple_of=training_config.get("pad_to_multiple_of", None),
        )
    elif training_mode == "rosetta":  # rosetta mode
        collator = RosettaDataCollator(
            slm_tokenizer=main_tokenizer,
            llm_tokenizer=llm_tokenizer,
            pad_to_multiple_of=training_config.get("pad_to_multiple_of", None),
            max_length=training_config.get("max_length", 2048),
            aligner=aligner,
            do_alignment=model_config.get("is_do_alignment", False),
        )
    else:
        raise ValueError(f"Invalid training mode: {training_mode}")

    # Ensure per-worker seeding if num_workers > 0
    def _worker_init_fn(worker_id):
        worker_seed = training_config["seed"] + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=per_device_batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        collate_fn=collator,
        worker_init_fn=_worker_init_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=per_device_batch_size,
        shuffle=False,
        sampler=eval_sampler,
        collate_fn=collator,
        worker_init_fn=_worker_init_fn,
    )

    # ------------------------------------------------------------------
    # Evaluation-only short-circuit
    # ------------------------------------------------------------------
    if args.eval_only:
        if distributed:
            local_eval_loss = evaluate_model(
                model,
                eval_loader,
                main_tokenizer,
                training_config["max_length"],
                device,
                training_mode,
            )
            loss_tensor = torch.tensor(
                [local_eval_loss], device=device, dtype=torch.float32
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_eval_loss = loss_tensor.item()
            if is_main_process:
                print(f"Evaluation (eval_only) loss: {avg_eval_loss:.4f}")
                wandb.log(
                    {
                        "eval/loss": avg_eval_loss,
                        "mode": "eval_only",
                    },
                    step=0,
                )
        else:
            eval_loss = evaluate_model(
                model,
                eval_loader,
                main_tokenizer,
                training_config["max_length"],
                device,
                training_mode,
            )
            print(f"Evaluation (eval_only) loss: {eval_loss:.4f}")
            if is_main_process:
                wandb.log(
                    {
                        "eval/loss": eval_loss,
                        "mode": "eval_only",
                    },
                    step=0,
                )

        if is_main_process:
            print("Evaluation-only run completed!")
            wandb.finish()
        if distributed:
            dist.destroy_process_group()
        return

    updates_per_epoch = math.ceil(len(train_loader) / grad_accum_steps)
    total_steps = updates_per_epoch * training_config["num_epochs"]

    # ------------------------------------------------------------------
    # Optimiser & scheduler
    # ------------------------------------------------------------------
    lr = training_config["learning_rate"]

    if training_mode == "baseline":
        # Simple optimizer for baseline mode
        optimizer = AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            weight_decay=training_config["weight_decay"],
        )
    else:  # rosetta mode
        # Separate parameter groups for Rosetta mode
        gate_params = []
        weight_params = []
        other_params = []

        for name, param in model.named_parameters():
            if param.requires_grad:
                if "gate" in name:
                    gate_params.append(param)
                elif "key_weight" in name or "value_weight" in name:
                    weight_params.append(param)
                else:
                    other_params.append(param)

        optimizer = AdamW(
            [
                {"params": gate_params, "lr": lr},
                {"params": weight_params, "lr": lr},
                {"params": other_params, "lr": lr},
            ],
            weight_decay=training_config["weight_decay"],
        )

    scheduler = get_scheduler(
        training_config["scheduler_type"],
        optimizer=optimizer,
        num_warmup_steps=int(training_config["warmup_ratio"] * total_steps),
        num_training_steps=total_steps,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print("Starting training…")
    global_step = 0
    optimizer.zero_grad()
    for epoch in range(training_config["num_epochs"]):
        if distributed and train_sampler is not None:
            # Ensure different shuffles across epochs in distributed setup
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        progress_bar = tqdm(
            total=updates_per_epoch,
            desc=f"Epoch {epoch + 1}/{training_config['num_epochs']}",
            disable=not is_main_process,
        )

        macro_step_in_epoch = 0
        accum_true_loss = 0.0
        micro_in_window = 0
        accum_metric_sums: Dict[str, float] = {}
        accum_metric_counts: Dict[str, int] = {}

        for batch_idx, batch in enumerate(train_loader):
            # Forward/backward with gradient accumulation and DDP no_sync for micro-steps
            is_accum_step = ((batch_idx + 1) % grad_accum_steps) != 0
            sync_ctx = (
                model.no_sync()
                if distributed and hasattr(model, "no_sync") and is_accum_step
                else contextlib.nullcontext()
            )

            with sync_ctx:
                loss = train_step(
                    model,
                    batch,
                    main_tokenizer,
                    training_config["max_length"],
                    device,
                    training_mode,
                    answer_prior_config=answer_prior_config,
                    answer_margin_config=answer_margin_config,
                    candidate_replay_config=candidate_replay_config,
                )
                _accumulate_metric_window(
                    accum_metric_sums,
                    accum_metric_counts,
                    getattr(train_step, "last_answer_prior_metrics", {}),
                )
                _accumulate_metric_window(
                    accum_metric_sums,
                    accum_metric_counts,
                    getattr(train_step, "last_answer_margin_metrics", {}),
                )
                _accumulate_metric_window(
                    accum_metric_sums,
                    accum_metric_counts,
                    getattr(train_step, "last_candidate_replay_metrics", {}),
                )
                true_loss_value = loss.detach().item()
                scaled_loss = loss / grad_accum_steps  # Gradient accumulation
                scaled_loss.backward()

            # accumulate true (unscaled) loss for averaging/printing
            epoch_loss += true_loss_value
            accum_true_loss += true_loss_value
            micro_in_window += 1

            # Optimizer step on boundaries or at last batch of the epoch
            did_step = (not is_accum_step) or (batch_idx + 1 == len(train_loader))
            grad_norm_value = None
            if did_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    max_norm=training_config["max_grad_norm"],
                )
                grad_norm_value = (
                    grad_norm.item()
                    if isinstance(grad_norm, torch.Tensor)
                    else float(grad_norm)
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                macro_step_in_epoch += 1

                # Update temperatures for Rosetta models
                if training_mode == "rosetta":
                    model_to_use = model.module if hasattr(model, "module") else model
                    for proj in model_to_use.projector_list:
                        if hasattr(proj, "update_temperature") and callable(
                            proj.update_temperature
                        ):
                            proj.update_temperature(global_step)

            # Progress bar and logging
            if is_main_process and did_step:
                # Calculate fractional epoch based on macro steps
                fractional_epoch = epoch + (macro_step_in_epoch / updates_per_epoch)

                avg_window_loss = accum_true_loss / max(1, micro_in_window)
                avg_metrics = _average_metric_window(
                    accum_metric_sums, accum_metric_counts
                )
                postfix = {
                    "loss": f"{avg_window_loss:.4f}",
                    "avg_loss": f"{epoch_loss / (batch_idx + 1):.4f}",
                    "lr": f"{scheduler.get_last_lr()[0]:.2e}",
                }
                if "answer_prior/weighted_loss" in avg_metrics:
                    postfix["ap"] = f"{avg_metrics['answer_prior/weighted_loss']:.4f}"
                if "answer_margin/weighted_loss" in avg_metrics:
                    postfix["am"] = f"{avg_metrics['answer_margin/weighted_loss']:.4f}"
                progress_bar.set_postfix(postfix)
                progress_bar.update(1)

                train_log = {
                    "train/loss": avg_window_loss,
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/grad_norm": grad_norm_value,
                    "train/epoch": fractional_epoch,
                }
                train_log.update(avg_metrics)
                train_log.update(_collect_projector_diagnostics(model))
                wandb.log(train_log, step=global_step)

                # reset window accumulators
                accum_true_loss = 0.0
                micro_in_window = 0
                accum_metric_sums = {}
                accum_metric_counts = {}

            # Evaluation and checkpointing only on real optimizer steps
            if did_step:
                # Calculate fractional epoch based on macro steps
                fractional_epoch = epoch + (macro_step_in_epoch / updates_per_epoch)
                # Evaluation at regular intervals under DDP using broadcasted decision
                want_eval = global_step % output_config["eval_steps"] == 0
                want_eval = broadcast_decision_from_rank0(
                    want_eval, distributed, device, rank
                )
                if want_eval:
                    if distributed:
                        # All ranks evaluate their shard and average
                        local_eval_loss = evaluate_model(
                            model,
                            eval_loader,
                            main_tokenizer,
                            training_config["max_length"],
                            device,
                            training_mode,
                        )
                        loss_tensor = torch.tensor(
                            [local_eval_loss], device=device, dtype=torch.float32
                        )
                        dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
                        avg_eval_loss = loss_tensor.item()
                        if is_main_process:
                            print(
                                f"\nEvaluation (mid-epoch) at step {global_step}: {avg_eval_loss:.4f}"
                            )
                            wandb.log(
                                {
                                    "eval/loss": avg_eval_loss,
                                    "eval/step": global_step,
                                    "eval/epoch": fractional_epoch,
                                },
                                step=global_step,
                            )
                    else:
                        eval_loss = evaluate_model(
                            model,
                            eval_loader,
                            main_tokenizer,
                            training_config["max_length"],
                            device,
                            training_mode,
                        )
                        print(
                            f"\nEvaluation loss at step {global_step}: {eval_loss:.4f}"
                        )
                        wandb.log(
                            {
                                "eval/loss": eval_loss,
                                "eval/step": global_step,
                                "eval/epoch": fractional_epoch,
                            },
                            step=global_step,
                        )

                # Checkpointing under DDP using broadcasted decision
                want_save = global_step % output_config["save_steps"] == 0
                want_save = broadcast_decision_from_rank0(
                    want_save, distributed, device, rank
                )
                if want_save:
                    if is_main_process:
                        checkpoint_dir = os.path.join(
                            timestamped_output_dir, f"checkpoint-{global_step}"
                        )
                        os.makedirs(checkpoint_dir, exist_ok=True)

                        # Unwrap DDP to access underlying model
                        base_model_ref = (
                            model.module
                            if isinstance(model, DistributedDataParallel)
                            else model
                        )

                        if training_mode == "baseline":
                            # Save baseline model
                            if hasattr(base_model_ref, "save_pretrained"):
                                # LoRA model - save LoRA weights
                                base_model_ref.save_pretrained(checkpoint_dir)
                                if hasattr(base_model_ref, "config"):
                                    base_model_ref.config.save_pretrained(
                                        checkpoint_dir
                                    )
                            else:
                                # Regular model - save full state dict
                                torch.save(
                                    base_model_ref.state_dict(),
                                    os.path.join(checkpoint_dir, "model.pt"),
                                )
                            main_tokenizer.save_pretrained(checkpoint_dir)
                        else:  # rosetta mode
                            # Save Rosetta components
                            for i, proj in enumerate(base_model_ref.projector_list):
                                # We save both the trainable weights and the constructor config
                                torch.save(
                                    proj.state_dict(),
                                    os.path.join(checkpoint_dir, f"projector_{i}.pt"),
                                )
                                save_projector(
                                    proj,
                                    os.path.join(checkpoint_dir, f"projector_{i}.json"),
                                )
                            base_model_ref.save_projector_config(
                                os.path.join(checkpoint_dir, "projector_config.json")
                            )

                        torch.save(
                            {
                                "step": global_step,
                                "epoch": epoch,
                                "optimizer_state_dict": optimizer.state_dict(),
                                "scheduler_state_dict": scheduler.state_dict(),
                                "loss": true_loss_value,  # true loss for this batch window
                            },
                            os.path.join(checkpoint_dir, "training_state.pt"),
                        )
                        print(f"\nCheckpoint saved at step {global_step}")

        avg_epoch_loss = epoch_loss / len(train_loader)

        # ------------------------------------------------------------------
        # Evaluation phase
        # ------------------------------------------------------------------
        if distributed:
            # Run eval on all ranks and average for deterministic sync
            local_eval_loss = evaluate_model(
                model,
                eval_loader,
                main_tokenizer,
                training_config["max_length"],
                device,
                training_mode,
            )
            loss_tensor = torch.tensor(
                [local_eval_loss], device=device, dtype=torch.float32
            )
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_eval_loss = loss_tensor.item()
            if is_main_process:
                print(
                    f"Epoch {epoch + 1} completed. Train loss: {avg_epoch_loss:.4f} | Eval loss: {avg_eval_loss:.4f}"
                )
                wandb.log(
                    {
                        "eval/epoch_loss": avg_eval_loss,
                        "epoch": epoch + 1,
                        "train/epoch_avg_loss": avg_epoch_loss,
                    },
                    step=global_step,
                )
        else:
            print(f"Running end-of-epoch evaluation for epoch {epoch + 1}...")
            avg_eval_loss = evaluate_model(
                model,
                eval_loader,
                main_tokenizer,
                training_config["max_length"],
                device,
                training_mode,
            )
            print(
                f"Epoch {epoch + 1} completed. Train loss: {avg_epoch_loss:.4f} | Eval loss: {avg_eval_loss:.4f}"
            )
            wandb.log(
                {
                    "eval/epoch_loss": avg_eval_loss,
                    "epoch": epoch + 1,
                    "train/epoch_avg_loss": avg_epoch_loss,
                },
                step=global_step,
            )

    # ------------------------------------------------------------------
    # Save final artefacts
    # ------------------------------------------------------------------
    if is_main_process:
        final_dir = os.path.join(timestamped_output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)

        base_model_ref = (
            model.module if isinstance(model, DistributedDataParallel) else model
        )

        if training_mode == "baseline":
            # Save final baseline model
            if hasattr(base_model_ref, "save_pretrained"):
                # LoRA model - save LoRA weights
                base_model_ref.save_pretrained(final_dir)
                if hasattr(base_model_ref, "config"):
                    base_model_ref.config.save_pretrained(final_dir)
            else:
                # Regular model - save full state dict
                torch.save(
                    base_model_ref.state_dict(), os.path.join(final_dir, "model.pt")
                )
            main_tokenizer.save_pretrained(final_dir)
        else:  # rosetta mode
            # Save final Rosetta components
            for i, proj in enumerate(base_model_ref.projector_list):
                torch.save(
                    proj.state_dict(), os.path.join(final_dir, f"projector_{i}.pt")
                )
                save_projector(proj, os.path.join(final_dir, f"projector_{i}.json"))
            base_model_ref.save_projector_config(
                os.path.join(final_dir, "projector_config.json")
            )

    if is_main_process:
        print("Training completed!")
        wandb.finish()

    # Clean up distributed training
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    # import debugpy
    # debugpy.listen(("0.0.0.0", 5678))
    # print("Waiting for debugger attach...")
    # debugpy.wait_for_client()
    # print("Debugger attached, running...")
    main()
