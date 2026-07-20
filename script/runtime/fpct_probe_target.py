from __future__ import annotations

"""Synthetic target used to verify the sealed FPCT runtime before natural data."""

import argparse
from dataclasses import dataclass
import importlib
import inspect
import json
from pathlib import Path
from typing import Any

from fpct_bootstrap import loaded_module, require_active


require_active(target=Path(__file__))


@dataclass
class _Encoding:
    input_ids: list[int]
    offset_mapping: list[tuple[int, int]]

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def get(self, key: str) -> Any:
        return getattr(self, key)


class _Backend:
    def to_str(self) -> str:
        return json.dumps(
            {
                "normalizer": None,
                "pre_tokenizer": {"type": "synthetic"},
                "decoder": None,
                "post_processor": None,
                "model": {"type": "synthetic"},
                "added_tokens": [],
            },
            sort_keys=True,
        )


class _Tokenizer:
    def __init__(self, directory: Path) -> None:
        self.name_or_path = str(directory)
        self.backend_tokenizer = _Backend()
        self.is_fast = True
        self.chat_template = "synthetic"
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.unk_token = "<unk>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.special_tokens_map_extended = {}
        self.all_special_tokens = ["<pad>", "<eos>", "<bos>", "<unk>"]
        self.all_special_ids = [0, 1, 2, 3]

    def get_vocab(self) -> dict[str, int]:
        return {
            "<pad>": 0,
            "<eos>": 1,
            "<bos>": 2,
            "<unk>": 3,
            "x": 4,
        }

    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def apply_chat_template(
        self,
        messages,
        tokenize: bool = False,
        add_generation_prompt: bool = False,
        enable_thinking: bool = False,
    ):
        text = "".join(message["content"] for message in messages)
        if add_generation_prompt:
            text += "|assistant|"
        if tokenize:
            return self(text)["input_ids"]
        return text

    def __call__(
        self,
        text: str,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
        **_kwargs,
    ) -> _Encoding:
        return _Encoding(
            input_ids=[100 + ord(char) for char in text],
            offset_mapping=[(index, index + 1) for index in range(len(text))],
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--delayed-import", action="store_true")
    parser.add_argument("--monkeypatch-signature", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        raise RuntimeError("probe output must be absolute")
    aligner_module = loaded_module("aligner")
    tokenizer_dir = output.parent / "synthetic-tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    if not tokenizer_file.exists():
        tokenizer_file.write_text('{"synthetic":true}\n', encoding="utf-8")
    receiver = _Tokenizer(tokenizer_dir)
    sender = _Tokenizer(tokenizer_dir)
    aligner = aligner_module.TokenAligner(
        receiver,
        sender,
        strategy=aligner_module.AlignmentStrategy.EXACT_IDENTITY,
        verbose=False,
    )
    details = aligner.align_chat_messages_soft(
        [{"role": "user", "content": "sealed probe"}],
        add_generation_prompt=True,
        top_k=4,
        return_details=True,
        apply_confidence_control=False,
    )
    eligible = [
        index for index, value in enumerate(details["message_mask"]) if value
    ]
    if not eligible:
        raise RuntimeError("synthetic probe produced no eligible parent")
    for index in eligible:
        if details["soft_alignment"]["source_indices"][index] != [
            index,
            -1,
            -1,
            -1,
        ]:
            raise RuntimeError("synthetic identity index mismatch")
    fpct1b = loaded_module("fpct_1b_audit")
    delayed_origin = None
    if args.delayed_import:
        sample = fpct1b.Sample(
            task="mmlu-redux",
            subject="synthetic",
            question_id="synthetic",
            question="Synthetic question?",
            choices=("A", "B"),
            content_group_sha256="0" * 64,
            sample_key_sha256="1" * 64,
            split="fit",
        )
        fpct1b.prompt_for_sample(sample)
        delayed = {
            "aligner": importlib.import_module("rosetta.model.aligner"),
            "evaluate": importlib.import_module("rosetta.utils.evaluate"),
        }
        delayed_origin = {
            key: str(Path(value.__file__).resolve())
            for key, value in delayed.items()
            if getattr(value, "__file__", None)
        }
    if args.monkeypatch_signature:
        def _wrong_signature(self):
            return None

        aligner_module.TokenAligner.align_chat_messages_soft = _wrong_signature
    payload = {
        "status": "SEALED_SYNTHETIC_PROBE_OK",
        "rank": args.rank,
        "aligner_origin": str(Path(aligner_module.__file__).resolve()),
        "fpct1b_origin": str(Path(fpct1b.__file__).resolve()),
        "eligible_parents": len(eligible),
        "extra_slots": details["soft_alignment"]["fpct_extra_slots"],
        "delayed_import_origins": delayed_origin,
        "align_chat_messages_soft_signature": str(
            inspect.signature(aligner_module.TokenAligner.align_chat_messages_soft)
        ),
    }
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
