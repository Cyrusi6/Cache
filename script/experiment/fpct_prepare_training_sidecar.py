from __future__ import annotations

"""Materialize one immutable certified training alignment cache before GPU output."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import torch
from transformers import AutoTokenizer

try:
    from fpct_bootstrap import require_active
except ModuleNotFoundError:
    from script.runtime.fpct_bootstrap import require_active

from rosetta.model.aligner import AlignmentStrategy, TokenAligner
from rosetta.train.dataset_adapters import AlignedChatDataset, create_dataset
from rosetta.utils.evaluate import set_default_chat_template
from rosetta.utils.model_loading import resolve_model_path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    require_active(target=Path(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--examples", type=int, default=2048)
    args = parser.parse_args()
    if args.output.exists() or args.manifest.exists():
        raise RuntimeError("refusing to overwrite frozen training sidecar")
    receiver_name = "Qwen/Qwen3-0.6B"
    sender_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    receiver_path = resolve_model_path(receiver_name)
    sender_path = resolve_model_path(sender_name)
    receiver = AutoTokenizer.from_pretrained(receiver_path, local_files_only=True)
    sender = AutoTokenizer.from_pretrained(sender_path, local_files_only=True)
    set_default_chat_template(receiver, receiver_name)
    set_default_chat_template(sender, sender_name)
    aligner = TokenAligner(
        slm_tokenizer=receiver, llm_tokenizer=sender,
        strategy=AlignmentStrategy.SOFT_SPAN_OVERLAP_V2,
        soft_alignment_score_mode="uniform", soft_alignment_boundary_bonus=.5,
        soft_alignment_boundary_tolerance=1, soft_alignment_min_weight=0.0,
        soft_alignment_confidence_mode="entropy", soft_alignment_confidence_alpha=.5,
        soft_alignment_confidence_floor=.5, soft_alignment_fallback_confidence=.25,
    )
    source = create_dataset(
        "MMLUChatDataset", split="auxiliary_train", num_samples=args.examples,
        max_word_count=1024,
    )
    dataset = AlignedChatDataset(
        source, aligner, max_length=1024, soft_alignment_top_k=4,
        fpct_alignment_sanitizer="certified_slot0_v1",
    )
    items = [dataset[index] for index in range(len(dataset))]
    payload = {
        "schema_version": 1, "alignment_sanitizer": "certified_slot0_v1",
        "top_k": 4, "examples": len(items), "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, args.output)
    manifest = {
        "schema_version": 1,
        "scientific_code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "sidecar": {"path": str(args.output.resolve()), "sha256": sha256(args.output), "bytes": args.output.stat().st_size, "examples": len(items)},
        "receiver": receiver_path, "sender": sender_path,
        "alignment": {"strategy": "soft_span_overlap_v2", "top_k": 4, "score_mode": "uniform", "sanitizer": "certified_slot0_v1", "max_length": 1024},
    }
    atomic_json(args.manifest, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
