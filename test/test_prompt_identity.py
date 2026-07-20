from __future__ import annotations

from datetime import datetime as real_datetime
import json
import locale
import os
from pathlib import Path
import time

import pytest
from transformers import AutoTokenizer
import transformers.utils.chat_template_utils as chat_template_utils

from rosetta.model.aligner import TokenAligner
from rosetta.utils.prompt_identity import (
    PromptIdentityContract,
    PromptIdentityError,
    build_prompt_identity_record,
    chat_template_sha256,
    load_prompt_identity_manifest,
    prompt_identity_record_sha256,
    scan_chat_template,
    tokenizer_identity,
    validate_formal_template_contract,
    verify_prompt_identity_row,
)


MODEL_ROOT = Path("/netdisk/lijunsi/c2c-route1-identifiability/models")
MODEL_NAMES = (
    "Llama-3.2-1B-Instruct",
    "Qwen2.5-0.5B-Instruct",
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "TinyLlama-1.1B-Chat-v1.0",
)


def _tokenizer(name: str):
    return AutoTokenizer.from_pretrained(MODEL_ROOT / name, local_files_only=True)


def _record(
    tokenizer, model_path: Path, *, date_string: str, include_material: bool = False
):
    identity = tokenizer_identity(tokenizer, model_path)
    return build_prompt_identity_record(
        messages=[{"role": "user", "content": "Determinism probe."}],
        tokenizers={"sender": tokenizer},
        model_paths={"sender": model_path},
        tokenizer_records={"sender": identity},
        add_generation_prompt=True,
        enable_thinking=False,
        remove_last_suffix=False,
        template_kwargs={"date_string": date_string},
        include_material=include_material,
    )


def test_all_formal_tokenizers_have_stable_asset_and_template_fingerprints() -> None:
    records = []
    for name in MODEL_NAMES:
        path = MODEL_ROOT / name
        tokenizer = _tokenizer(name)
        record = tokenizer_identity(tokenizer, path)
        assert record["revision"].startswith("local-assets-sha256:")
        assert len(record["chat_template_sha256"]) == 64
        assert len(record["tokenizer_assets"]["asset_sha256"]) == 64
        records.append((name, record))
    dynamic = [
        name
        for name, record in records
        if record["template_environment_scan"]["has_dynamic_dependency"]
    ]
    assert dynamic == ["Llama-3.2-1B-Instruct"]


def test_llama_template_requires_fixed_date_but_qwen_does_not() -> None:
    llama = _tokenizer("Llama-3.2-1B-Instruct")
    llama_record = tokenizer_identity(llama, MODEL_ROOT / "Llama-3.2-1B-Instruct")
    scan = scan_chat_template(llama.chat_template)
    assert scan["requires_fixed_date_string"] is True
    assert scan["supports_explicit_date_string"] is True
    with pytest.raises(
        PromptIdentityError, match="requires explicit fixed date_string"
    ):
        validate_formal_template_contract(
            llama_record,
            date_string=None,
            timezone="UTC",
            locale="C",
            template_kwargs={},
        )

    qwen = _tokenizer("Qwen3-0.6B")
    qwen_record = tokenizer_identity(qwen, MODEL_ROOT / "Qwen3-0.6B")
    validate_formal_template_contract(
        qwen_record,
        date_string=None,
        timezone="UTC",
        locale="C",
        template_kwargs={},
    )


def test_fixed_date_is_invariant_to_ambient_date_timezone_and_locale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _tokenizer("Llama-3.2-1B-Instruct")
    model_path = MODEL_ROOT / "Llama-3.2-1B-Instruct"
    original_locale = locale.setlocale(locale.LC_ALL)
    original_tz = os.environ.get("TZ")

    class July17(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 17, 1, 2, 3, tzinfo=tz)

    class December31(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2031, 12, 31, 23, 59, 59, tzinfo=tz)

    try:
        locale.setlocale(locale.LC_ALL, "C")
        monkeypatch.setenv("TZ", "UTC")
        time.tzset()
        monkeypatch.setattr(chat_template_utils, "datetime", July17)
        first = _record(tokenizer, model_path, date_string="17 Jul 2026")

        locale.setlocale(locale.LC_ALL, "zh_CN.utf8")
        monkeypatch.setenv("TZ", "Asia/Shanghai")
        time.tzset()
        monkeypatch.setattr(chat_template_utils, "datetime", December31)
        second = _record(tokenizer, model_path, date_string="17 Jul 2026")
    finally:
        locale.setlocale(locale.LC_ALL, original_locale)
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()

    assert prompt_identity_record_sha256(first) == prompt_identity_record_sha256(second)
    assert first == second


def test_unfixed_llama_template_changes_with_ambient_date(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _tokenizer("Llama-3.2-1B-Instruct")
    messages = [{"role": "user", "content": "Determinism probe."}]

    class July17(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 17, tzinfo=tz)

    class July19(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 7, 19, tzinfo=tz)

    monkeypatch.setattr(chat_template_utils, "datetime", July17)
    first = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    monkeypatch.setattr(chat_template_utils, "datetime", July19)
    second = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    assert first != second


def test_token_aligner_passes_same_fixed_date_to_both_tokenizers() -> None:
    receiver = _tokenizer("Qwen3-0.6B")
    sender = _tokenizer("Llama-3.2-1B-Instruct")
    aligner = TokenAligner(
        receiver,
        sender,
        strategy="soft_span_overlap_v2",
        chat_template_kwargs={"date_string": "17 Jul 2026"},
    )
    details = aligner.align_chat_messages_soft(
        [{"role": "user", "content": "Determinism probe."}],
        return_details=True,
    )
    assert "Today Date: 17 Jul 2026" in details["llm_text"]
    assert "Today Date:" not in details["slm_text"]
    identity = build_prompt_identity_record(
        messages=[{"role": "user", "content": "Determinism probe."}],
        tokenizers={"receiver": receiver, "sender": sender},
        model_paths={
            "receiver": MODEL_ROOT / "Qwen3-0.6B",
            "sender": MODEL_ROOT / "Llama-3.2-1B-Instruct",
        },
        add_generation_prompt=True,
        enable_thinking=False,
        remove_last_suffix=False,
        template_kwargs={"date_string": "17 Jul 2026"},
        tokenize_add_special_tokens=False,
        include_material=True,
    )
    assert identity["roles"]["receiver"]["input_ids"] == details["slm_ids"]
    assert identity["roles"]["sender"]["input_ids"] == details["llm_ids"]


def test_manifest_and_row_verification_fail_closed(tmp_path: Path) -> None:
    tokenizer = _tokenizer("Llama-3.2-1B-Instruct")
    model_path = MODEL_ROOT / "Llama-3.2-1B-Instruct"
    record = _record(tokenizer, model_path, date_string="17 Jul 2026")
    row = {**record, "sample_key": "sample-1"}
    manifest = {
        "schema_version": 1,
        "protocol": "c2c_prompt_identity_v1",
        "rows": [row],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_prompt_identity_manifest(path)
    assert loaded["rows"][0]["sample_key"] == "sample-1"

    changed = json.loads(json.dumps(record))
    changed["roles"]["sender"]["input_ids_sha256"] = "0" * 64
    with pytest.raises(PromptIdentityError, match="roles"):
        verify_prompt_identity_row(
            expected=record, actual=changed, sample_key="sample-1"
        )


def test_frozen_material_is_checked_during_cpu_preflight() -> None:
    tokenizer = _tokenizer("Llama-3.2-1B-Instruct")
    model_path = MODEL_ROOT / "Llama-3.2-1B-Instruct"
    expected = _record(
        tokenizer,
        model_path,
        date_string="17 Jul 2026",
        include_material=True,
    )
    actual = json.loads(json.dumps(expected))
    actual["roles"]["sender"]["rendered_prompt"] += " "

    # Runtime verification intentionally uses compact hashes only.
    verify_prompt_identity_row(
        expected=expected,
        actual=actual,
        sample_key="sample-1",
        require_material=False,
    )
    with pytest.raises(PromptIdentityError, match="rendered_prompt"):
        verify_prompt_identity_row(
            expected=expected,
            actual=actual,
            sample_key="sample-1",
            require_material=True,
        )


def test_formal_contract_requires_verify_manifest() -> None:
    with pytest.raises(PromptIdentityError, match="manifest"):
        PromptIdentityContract.from_config(
            {
                "enabled": True,
                "formal_experiment": True,
                "mode": "verify",
                "date_string": "17 Jul 2026",
                "timezone": "UTC",
                "locale": "C",
            }
        )


def test_chat_template_sha_uses_exact_template_bytes() -> None:
    tokenizer = _tokenizer("Llama-3.2-1B-Instruct")
    assert chat_template_sha256(tokenizer) == (
        "5816fce10444e03c2e9ee1ef8a4a1ea61ae7e69e438613f3b17b69d0426223a4"
    )
