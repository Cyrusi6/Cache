"""Evaluation-only interventions for Route-1 causal diagnostics.

The helpers in this module deliberately operate after a checkpoint is selected.
They never alter checkpoint tensors, training recipes, or optimizer state.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence


EVAL_INTERVENTION_SCHEMA_VERSION = 1
VALID_TOP_K = {1, 4}
VALID_ENTROPY_MODES = {"native", "constant", "shuffled"}
VALID_GATE_MODES = {
    "learned",
    "static",
    "forced_on",
    "alignment_forced_on",
    "legacy_forced_on",
}


def _optional_int(value: Any, field: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be an integer, got {value!r}") from exc


def normalize_eval_intervention(
    intervention: Optional[Mapping[str, Any]],
) -> Optional[dict[str, Any]]:
    """Validate and canonicalize one evaluation-only intervention."""
    if intervention is None:
        return None
    if not isinstance(intervention, Mapping):
        raise ValueError("eval.intervention must be a mapping")

    top_k = _optional_int(intervention.get("top_k"), "top_k")
    if top_k is not None and top_k not in VALID_TOP_K:
        raise ValueError(f"top_k must be one of {sorted(VALID_TOP_K)}, got {top_k}")

    entropy_mode = intervention.get("entropy_mode")
    if entropy_mode == "shuffle":
        entropy_mode = "shuffled"
    if entropy_mode is not None:
        entropy_mode = str(entropy_mode)
        if entropy_mode not in VALID_ENTROPY_MODES:
            raise ValueError(
                "entropy_mode must be one of "
                f"{sorted(VALID_ENTROPY_MODES)}, got {entropy_mode!r}"
            )

    gate_mode = intervention.get("gate_mode")
    if gate_mode == "forced-on":
        gate_mode = "forced_on"
    if gate_mode is not None:
        gate_mode = str(gate_mode)
        if gate_mode not in VALID_GATE_MODES:
            raise ValueError(
                f"gate_mode must be one of {sorted(VALID_GATE_MODES)}, "
                f"got {gate_mode!r}"
            )

    constant_value = intervention.get("entropy_constant_value")
    if entropy_mode == "constant":
        if constant_value is None:
            raise ValueError(
                "entropy_constant_value is required for entropy_mode='constant'"
            )
        constant_value = float(constant_value)
        if not 0.0 <= constant_value <= 1.0:
            raise ValueError("entropy_constant_value must be in [0, 1]")
    elif constant_value is not None:
        raise ValueError(
            "entropy_constant_value is only valid for entropy_mode='constant'"
        )

    shuffle_seed = _optional_int(
        intervention.get("entropy_shuffle_seed"), "entropy_shuffle_seed"
    )
    if entropy_mode == "shuffled" and shuffle_seed is None:
        raise ValueError(
            "entropy_shuffle_seed is required for entropy_mode='shuffled'"
        )
    if entropy_mode != "shuffled" and shuffle_seed is not None:
        raise ValueError(
            "entropy_shuffle_seed is only valid for entropy_mode='shuffled'"
        )

    intervention_id = intervention.get("id")
    if intervention_id is None:
        fields = [
            f"k{top_k}" if top_k is not None else None,
            f"entropy_{entropy_mode}" if entropy_mode is not None else None,
            f"gate_{gate_mode}" if gate_mode is not None else None,
        ]
        intervention_id = "__".join(field for field in fields if field) or "native"
    intervention_id = str(intervention_id).strip()
    if not intervention_id:
        raise ValueError("intervention id must not be empty")

    normalized: dict[str, Any] = {
        "schema_version": EVAL_INTERVENTION_SCHEMA_VERSION,
        "id": intervention_id,
        "scope": "evaluation_only",
        "top_k": top_k,
        "entropy_mode": entropy_mode,
        "gate_mode": gate_mode,
    }
    if gate_mode is not None:
        alignment_component = {
            "learned": "learned",
            "static": "static",
            "forced_on": "forced_on",
            "alignment_forced_on": "forced_on",
            "legacy_forced_on": "learned",
        }[gate_mode]
        legacy_component = {
            "learned": "checkpoint_native",
            "static": "checkpoint_native",
            "forced_on": "forced_on",
            "alignment_forced_on": "checkpoint_native",
            "legacy_forced_on": "forced_on",
        }[gate_mode]
        normalized["gate_components"] = {
            "alignment_confidence": alignment_component,
            "legacy_scalar_kv": legacy_component,
        }
    if constant_value is not None:
        normalized["entropy_constant_value"] = constant_value
    if shuffle_seed is not None:
        normalized["entropy_shuffle_seed"] = shuffle_seed
    return normalized


def apply_eval_intervention_to_config(
    config: MutableMapping[str, Any],
    cli_override: Optional[Mapping[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Apply an intervention to an evaluation config in memory.

    ``top_k`` and entropy controls are materialized in ``rosetta_config`` because
    alignment is recomputed for every dev example. The gate mode remains under
    ``eval.intervention`` and is applied only after loading projector weights.
    """
    eval_config = config.setdefault("eval", {})
    raw = copy.deepcopy(eval_config.get("intervention"))
    if raw is None:
        raw = {}
    if cli_override:
        raw.update(
            {key: value for key, value in cli_override.items() if value is not None}
        )
    if not raw:
        return None

    normalized = normalize_eval_intervention(raw)
    assert normalized is not None
    model_config = config.get("model")
    if not isinstance(model_config, MutableMapping):
        raise ValueError("evaluation intervention requires a model mapping")
    if "rosetta" not in str(model_config.get("model_name", "")).lower():
        raise ValueError("evaluation intervention requires model_name=Rosetta")
    rosetta_config = model_config.get("rosetta_config")
    if not isinstance(rosetta_config, MutableMapping):
        raise ValueError("evaluation intervention requires model.rosetta_config")

    if normalized["top_k"] is not None:
        rosetta_config["soft_alignment_top_k"] = normalized["top_k"]

    entropy_mode = normalized["entropy_mode"]
    if entropy_mode is not None:
        rosetta_config.pop("soft_alignment_confidence_constant_value", None)
        rosetta_config.pop("soft_alignment_confidence_shuffle_seed", None)
        if entropy_mode == "native":
            rosetta_config["soft_alignment_confidence_control_mode"] = "native"
        elif entropy_mode == "constant":
            rosetta_config["soft_alignment_confidence_control_mode"] = "constant"
            rosetta_config["soft_alignment_confidence_constant_value"] = normalized[
                "entropy_constant_value"
            ]
        else:
            rosetta_config["soft_alignment_confidence_control_mode"] = "shuffle"
            rosetta_config["soft_alignment_confidence_shuffle_seed"] = normalized[
                "entropy_shuffle_seed"
            ]

    eval_config["intervention"] = normalized
    return normalized


def apply_projector_eval_intervention(
    model: Any,
    intervention: Optional[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply a normalized gate intervention after checkpoint loading."""
    normalized = normalize_eval_intervention(intervention)
    gate_mode = None if normalized is None else normalized.get("gate_mode")
    if gate_mode is None:
        return {"gate_mode": None, "projector_count": 0}

    projectors: Sequence[Any] = list(getattr(model, "projector_list", []) or [])
    if not projectors:
        raise ValueError("gate intervention requested but model has no projectors")

    applied = 0
    for projector in projectors:
        setter = getattr(projector, "set_alignment_confidence_eval_mode", None)
        if not callable(setter):
            raise ValueError(
                "gate intervention requires projectors supporting "
                "set_alignment_confidence_eval_mode"
            )
        setter(gate_mode)
        applied += 1
    return {
        "gate_mode": gate_mode,
        "gate_components": normalized.get("gate_components"),
        "projector_count": applied,
    }


def config_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_eval_intervention_provenance(
    *,
    config: Mapping[str, Any],
    config_path: Optional[Path] = None,
    argv: Optional[Sequence[str]] = None,
) -> Optional[dict[str, Any]]:
    intervention = normalize_eval_intervention(
        config.get("eval", {}).get("intervention")
    )
    if intervention is None:
        return None
    rosetta_config = config.get("model", {}).get("rosetta_config", {})
    provenance: dict[str, Any] = {
        "schema_version": EVAL_INTERVENTION_SCHEMA_VERSION,
        "intervention": intervention,
        "checkpoint_dir": rosetta_config.get("checkpoints_dir"),
        "base_model": rosetta_config.get("base_model"),
        "teacher_model": rosetta_config.get("teacher_model"),
        "effective_soft_alignment_top_k": rosetta_config.get(
            "soft_alignment_top_k"
        ),
        "effective_confidence_control_mode": rosetta_config.get(
            "soft_alignment_confidence_control_mode", "native"
        ),
        "training_state_mutated": False,
    }
    if config_path is not None:
        resolved = config_path.resolve()
        provenance["config_path"] = str(resolved)
        provenance["config_sha256"] = config_sha256(resolved)
    if argv is not None:
        provenance["argv"] = list(argv)
    canonical = json.dumps(provenance, sort_keys=True, separators=(",", ":"))
    provenance["provenance_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return provenance
