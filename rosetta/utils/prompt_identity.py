"""Fail-closed prompt identity primitives for formal evaluations.

The functions in this module are CPU-only.  They canonicalize chat messages,
render tokenizer templates with explicit variables, fingerprint tokenizer
assets, and compare per-sample prompt identities without using labels or model
outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional, Sequence


PROMPT_IDENTITY_SCHEMA_VERSION = 1
PROMPT_IDENTITY_PROTOCOL = "c2c_prompt_identity_v1"
PROMPT_IDENTITY_ROLE_FIELDS = (
    "rendered_prompt_sha256",
    "input_ids_sha256",
    "token_count",
    "rendered_utf8_bytes",
    "chat_template_sha256",
    "tokenizer_revision",
    "tokenize_add_special_tokens",
)
_FIXED_DATE = re.compile(r"^[0-9]{2} [A-Z][a-z]{2} [0-9]{4}$")
_TOKENIZER_ASSET_NAMES = {
    "added_tokens.json",
    "chat_template.jinja",
    "merges.txt",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "vocab.json",
    "vocab.txt",
}


class PromptIdentityError(RuntimeError):
    """A formal prompt identity contract or fingerprint did not match."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_messages(messages: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for index, message in enumerate(messages):
        if not isinstance(message, Mapping):
            raise PromptIdentityError(f"message {index} is not a mapping")
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise PromptIdentityError(f"message {index} must have string role/content")
        # Preserve every JSON-compatible message field because some templates
        # consume tool/name metadata in addition to role/content.
        output.append(json.loads(canonical_json_bytes(dict(message)).decode("utf-8")))
    if not output:
        raise PromptIdentityError("canonical messages cannot be empty")
    return output


def canonical_messages_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return sha256_bytes(canonical_json_bytes(canonical_messages(messages)))


def _template_bytes(template: Any) -> bytes:
    if isinstance(template, str):
        return template.encode("utf-8")
    return canonical_json_bytes(template)


def chat_template_sha256(tokenizer: Any) -> str:
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        raise PromptIdentityError("tokenizer has no chat_template")
    return sha256_bytes(_template_bytes(template))


def _template_text(template: Any) -> str:
    if isinstance(template, str):
        return template
    return json.dumps(template, ensure_ascii=False, sort_keys=True)


def scan_chat_template(template: Any) -> dict[str, Any]:
    """Identify clock, timezone, locale, random, and process-env dependencies."""

    text = _template_text(template)
    patterns = {
        "system_clock": (
            r"\bstrftime_now\b",
            r"\b(?:utc)?now\s*\(",
            r"\bcurrent_(?:date|time|datetime)\b",
            r"\btimestamp\s*\(",
        ),
        "timezone": (
            r"\btimezone\b",
            r"\btzinfo\b",
            r"\b(?:local|utc)_?time\b",
        ),
        "locale": (
            r"\blocale\b",
            r"\bgetlocale\b",
            r"\bsetlocale\b",
        ),
        "process_environment": (
            r"\bos\.environ\b",
            r"\benviron\s*\[",
            r"\bgetenv\s*\(",
        ),
        "randomness": (
            r"\brandom\s*\(",
            r"\buuid(?:1|4)?\s*\(",
            r"\bsecrets\.",
        ),
    }
    matches = {
        category: sorted(
            {
                match.group(0)
                for pattern in category_patterns
                for match in re.finditer(pattern, text, flags=re.IGNORECASE)
            }
        )
        for category, category_patterns in patterns.items()
    }
    has_date_override = bool(re.search(r"\bdate_string\b", text))
    active_categories = [name for name, values in matches.items() if values]
    return {
        "chat_template_sha256": sha256_bytes(_template_bytes(template)),
        "has_dynamic_dependency": bool(active_categories),
        "active_categories": active_categories,
        "matches": matches,
        "supports_explicit_date_string": has_date_override,
        "requires_fixed_date_string": bool(matches["system_clock"]),
    }


def _tokenizer_asset_paths(model_path: Optional[Path]) -> list[Path]:
    if model_path is None or not model_path.is_dir():
        return []
    return sorted(
        (
            path
            for path in model_path.iterdir()
            if path.is_file()
            and (
                path.name in _TOKENIZER_ASSET_NAMES
                or path.name.startswith("tokenizer.")
            )
        ),
        key=lambda path: path.name,
    )


def tokenizer_asset_fingerprint(model_path: Optional[Path]) -> dict[str, Any]:
    paths = _tokenizer_asset_paths(model_path)
    digest = hashlib.sha256()
    files = []
    for path in paths:
        file_sha = sha256_file(path)
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_sha.encode("ascii"))
        digest.update(b"\0")
        files.append(
            {"name": path.name, "sha256": file_sha, "bytes": path.stat().st_size}
        )
    return {
        "asset_sha256": digest.hexdigest() if files else None,
        "files": files,
    }


def tokenizer_identity(
    tokenizer: Any, model_path: Optional[Path] = None
) -> dict[str, Any]:
    model_path = None if model_path is None else Path(model_path).resolve()
    assets = tokenizer_asset_fingerprint(model_path)
    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    revision = init_kwargs.get("_commit_hash") or init_kwargs.get("revision")
    if revision is None and assets["asset_sha256"] is not None:
        revision = f"local-assets-sha256:{assets['asset_sha256']}"
    template = getattr(tokenizer, "chat_template", None)
    if template is None:
        raise PromptIdentityError("tokenizer has no chat_template")
    return {
        "model_path": None if model_path is None else str(model_path),
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": int(len(tokenizer)),
        "revision": revision,
        "chat_template_sha256": chat_template_sha256(tokenizer),
        "template_environment_scan": scan_chat_template(template),
        "tokenizer_assets": assets,
    }


def validate_formal_template_contract(
    tokenizer_record: Mapping[str, Any],
    *,
    date_string: Optional[str],
    timezone: Optional[str],
    locale: Optional[str],
    template_kwargs: Mapping[str, Any],
) -> None:
    if timezone != "UTC":
        raise PromptIdentityError("formal prompt identity requires timezone='UTC'")
    if locale != "C":
        raise PromptIdentityError("formal prompt identity requires locale='C'")
    scan = tokenizer_record["template_environment_scan"]
    if scan.get("requires_fixed_date_string"):
        if (
            not isinstance(date_string, str)
            or _FIXED_DATE.fullmatch(date_string) is None
        ):
            raise PromptIdentityError(
                "dynamic chat template requires explicit fixed date_string like "
                "'17 Jul 2026'"
            )
        if not scan.get("supports_explicit_date_string"):
            raise PromptIdentityError(
                "dynamic chat template uses the system clock without a date_string override"
            )
    unsupported = set(scan.get("active_categories", [])) - {"system_clock"}
    if unsupported:
        raise PromptIdentityError(
            "formal chat template has unsupported runtime dependencies: "
            f"{sorted(unsupported)}"
        )
    if (
        "date_string" in template_kwargs
        and template_kwargs["date_string"] != date_string
    ):
        raise PromptIdentityError(
            "template_kwargs.date_string disagrees with date_string"
        )


def normalized_template_kwargs(
    *, date_string: Optional[str], template_kwargs: Optional[Mapping[str, Any]] = None
) -> dict[str, Any]:
    output = dict(template_kwargs or {})
    if date_string is not None:
        prior = output.get("date_string")
        if prior is not None and prior != date_string:
            raise PromptIdentityError("conflicting date_string template variables")
        output["date_string"] = date_string
    canonical_json_bytes(output)
    return output


def render_chat_template_text(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    remove_last_suffix: bool = False,
    template_kwargs: Optional[Mapping[str, Any]] = None,
) -> tuple[list[dict[str, Any]], str]:
    canonical = canonical_messages(messages)
    kwargs = dict(template_kwargs or {})
    if remove_last_suffix:
        if canonical[-1]["role"] != "assistant":
            raise PromptIdentityError(
                "remove_last_suffix requires a final assistant message"
            )
        rendered = tokenizer.apply_chat_template(
            canonical[:-1],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            **kwargs,
        )
        rendered += canonical[-1]["content"]
    else:
        rendered = tokenizer.apply_chat_template(
            canonical,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            **kwargs,
        )
    return canonical, rendered


def render_chat_identity(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
    remove_last_suffix: bool = False,
    template_kwargs: Optional[Mapping[str, Any]] = None,
    model_path: Optional[Path] = None,
    tokenizer_record: Optional[Mapping[str, Any]] = None,
    tokenize_add_special_tokens: bool = False,
    include_material: bool = False,
) -> dict[str, Any]:
    canonical, rendered = render_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        remove_last_suffix=remove_last_suffix,
        template_kwargs=template_kwargs,
    )
    encoded = tokenizer(rendered, add_special_tokens=tokenize_add_special_tokens)
    input_ids = [int(value) for value in encoded["input_ids"]]
    identity = (
        dict(tokenizer_record)
        if tokenizer_record is not None
        else tokenizer_identity(tokenizer, model_path)
    )
    record = {
        "rendered_prompt_sha256": sha256_bytes(rendered.encode("utf-8")),
        "input_ids_sha256": sha256_bytes(canonical_json_bytes(input_ids)),
        "token_count": len(input_ids),
        "rendered_utf8_bytes": len(rendered.encode("utf-8")),
        "chat_template_sha256": identity["chat_template_sha256"],
        "tokenizer_revision": identity["revision"],
        "tokenize_add_special_tokens": bool(tokenize_add_special_tokens),
    }
    if include_material:
        record["rendered_prompt"] = rendered
        record["input_ids"] = input_ids
    return record


def build_prompt_identity_record(
    *,
    messages: Sequence[Mapping[str, Any]],
    tokenizers: Mapping[str, Any],
    model_paths: Mapping[str, Optional[Path]],
    tokenizer_records: Optional[Mapping[str, Mapping[str, Any]]] = None,
    tokenize_add_special_tokens: bool = False,
    add_generation_prompt: bool,
    enable_thinking: bool,
    remove_last_suffix: bool,
    template_kwargs: Mapping[str, Any],
    sample_identity: Optional[Mapping[str, Any]] = None,
    include_material: bool = False,
) -> dict[str, Any]:
    canonical = canonical_messages(messages)
    roles = {}
    for role, tokenizer in tokenizers.items():
        roles[str(role)] = render_chat_identity(
            tokenizer,
            canonical,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            remove_last_suffix=remove_last_suffix,
            template_kwargs=template_kwargs,
            model_path=model_paths.get(role),
            tokenizer_record=(tokenizer_records or {}).get(role),
            tokenize_add_special_tokens=tokenize_add_special_tokens,
            include_material=include_material,
        )
    output = {
        "schema_version": PROMPT_IDENTITY_SCHEMA_VERSION,
        "protocol": PROMPT_IDENTITY_PROTOCOL,
        "canonical_messages_sha256": sha256_bytes(canonical_json_bytes(canonical)),
        "message_count": len(canonical),
        "message_roles": [message["role"] for message in canonical],
        "add_generation_prompt": bool(add_generation_prompt),
        "enable_thinking": bool(enable_thinking),
        "remove_last_suffix": bool(remove_last_suffix),
        "template_kwargs_sha256": sha256_bytes(canonical_json_bytes(template_kwargs)),
        "roles": roles,
    }
    if sample_identity is not None:
        output["sample_identity"] = dict(sample_identity)
    if include_material:
        output["canonical_messages"] = canonical
    return output


def prompt_identity_comparison_view(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "canonical_messages_sha256",
        "message_count",
        "message_roles",
        "add_generation_prompt",
        "enable_thinking",
        "remove_last_suffix",
        "template_kwargs_sha256",
    )
    output = {field: record[field] for field in fields}
    output["roles"] = {
        str(role): {field: values.get(field) for field in PROMPT_IDENTITY_ROLE_FIELDS}
        for role, values in record["roles"].items()
    }
    return output


def prompt_identity_record_sha256(record: Mapping[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(prompt_identity_comparison_view(record)))


def prompt_identity_sample_key(identity: Mapping[str, Any]) -> str:
    required = ("task", "subject", "question_id")
    if any(identity.get(field) is None for field in required):
        raise PromptIdentityError(
            f"sample identity requires fields {required}: {identity}"
        )
    value = {field: str(identity[field]) for field in required}
    if identity.get("content_hash") is not None:
        value["content_hash"] = str(identity["content_hash"])
    return sha256_bytes(canonical_json_bytes(value))


@dataclass(frozen=True)
class PromptIdentityContract:
    enabled: bool
    formal_experiment: bool
    mode: str
    date_string: Optional[str]
    timezone: Optional[str]
    locale: Optional[str]
    template_kwargs: Mapping[str, Any]
    manifest: Optional[Path]
    manifest_sha256: Optional[str]
    expected_rows: Optional[int]

    @classmethod
    def from_config(
        cls, config: Optional[Mapping[str, Any]], *, repo_root: Optional[Path] = None
    ) -> "PromptIdentityContract":
        value = dict(config or {})
        enabled = bool(value.get("enabled", False))
        formal = bool(value.get("formal_experiment", False))
        mode = str(value.get("mode", "verify" if enabled else "off"))
        if mode not in {"off", "freeze", "verify"}:
            raise PromptIdentityError("prompt_identity.mode must be off/freeze/verify")
        if formal and not enabled:
            raise PromptIdentityError(
                "formal_experiment requires prompt_identity.enabled=true"
            )
        if enabled and mode == "off":
            raise PromptIdentityError("enabled prompt identity cannot use mode=off")
        date_string = value.get("date_string")
        timezone = value.get("timezone")
        locale = value.get("locale")
        template_kwargs = normalized_template_kwargs(
            date_string=date_string,
            template_kwargs=value.get("template_kwargs"),
        )
        manifest_value = value.get("manifest")
        manifest = None
        if manifest_value:
            manifest = Path(str(manifest_value)).expanduser()
            if not manifest.is_absolute() and repo_root is not None:
                manifest = (repo_root / manifest).resolve()
        manifest_sha = value.get("manifest_sha256")
        expected_rows = value.get("expected_rows")
        if mode == "verify" and enabled:
            if manifest is None or not manifest_sha:
                raise PromptIdentityError(
                    "verify mode requires manifest and manifest_sha256"
                )
        if formal:
            if timezone != "UTC" or locale != "C":
                raise PromptIdentityError(
                    "formal prompt identity requires timezone=UTC and locale=C"
                )
        return cls(
            enabled=enabled,
            formal_experiment=formal,
            mode=mode,
            date_string=None if date_string is None else str(date_string),
            timezone=None if timezone is None else str(timezone),
            locale=None if locale is None else str(locale),
            template_kwargs=template_kwargs,
            manifest=manifest,
            manifest_sha256=None if manifest_sha is None else str(manifest_sha),
            expected_rows=None if expected_rows is None else int(expected_rows),
        )


def load_prompt_identity_manifest(
    path: Path, *, expected_sha256: Optional[str] = None
) -> dict[str, Any]:
    path = Path(path)
    payload = path.read_bytes()
    actual_sha = sha256_bytes(payload)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise PromptIdentityError(
            f"prompt identity manifest SHA mismatch: {actual_sha} != {expected_sha256}"
        )
    value = json.loads(payload.decode("utf-8"))
    if value.get("schema_version") != PROMPT_IDENTITY_SCHEMA_VERSION:
        raise PromptIdentityError("unexpected prompt identity manifest schema")
    if value.get("protocol") != PROMPT_IDENTITY_PROTOCOL:
        raise PromptIdentityError("unexpected prompt identity manifest protocol")
    rows = value.get("rows")
    if not isinstance(rows, list):
        raise PromptIdentityError("prompt identity manifest rows must be a list")
    keys = [str(row.get("sample_key", "")) for row in rows]
    if any(not key for key in keys) or len(keys) != len(set(keys)):
        raise PromptIdentityError("prompt identity manifest has invalid sample keys")
    return value


def manifest_row_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["sample_key"]): row for row in manifest.get("rows", [])}


def verify_prompt_identity_row(
    *,
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    sample_key: str,
    require_material: bool = False,
) -> None:
    expected_view = prompt_identity_comparison_view(expected)
    actual_view = prompt_identity_comparison_view(actual)
    if expected_view != actual_view:
        differing = [
            key
            for key in expected_view
            if expected_view.get(key) != actual_view.get(key)
        ]
        raise PromptIdentityError(
            f"prompt identity mismatch for {sample_key}: fields={differing}"
        )
    if not require_material:
        return
    if expected.get("canonical_messages") != actual.get("canonical_messages"):
        raise PromptIdentityError(
            f"prompt identity material mismatch for {sample_key}: canonical_messages"
        )
    for role in sorted(expected_view["roles"]):
        expected_role = expected.get("roles", {}).get(role, {})
        actual_role = actual.get("roles", {}).get(role, {})
        for field in ("rendered_prompt", "input_ids"):
            if field not in expected_role or field not in actual_role:
                raise PromptIdentityError(
                    f"prompt identity frozen material missing for {sample_key}: "
                    f"{role}.{field}"
                )
            if expected_role[field] != actual_role[field]:
                raise PromptIdentityError(
                    f"prompt identity material mismatch for {sample_key}: "
                    f"{role}.{field}"
                )


def flatten_prompt_identity_record(record: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "prompt_identity_record_sha256": prompt_identity_record_sha256(record),
        "canonical_messages_sha256": record["canonical_messages_sha256"],
    }
    for role, values in record["roles"].items():
        prefix = f"{role}_prompt_"
        for key in (
            "rendered_prompt_sha256",
            "input_ids_sha256",
            "token_count",
            "chat_template_sha256",
            "tokenizer_revision",
            "tokenize_add_special_tokens",
        ):
            output[prefix + key] = values.get(key)
    return output


def audit_tokenizer_paths(
    model_paths: Iterable[Path], loader: Any
) -> list[dict[str, Any]]:
    output = []
    for model_path in sorted((Path(path).resolve() for path in model_paths), key=str):
        tokenizer = loader(model_path)
        identity = tokenizer_identity(tokenizer, model_path)
        output.append({"model": model_path.name, **identity})
    return output
