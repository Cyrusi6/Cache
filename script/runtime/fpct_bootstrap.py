from __future__ import annotations

"""Isolated, attestable launcher for formal FPCT Python entrypoints."""

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import sysconfig
from types import ModuleType
from typing import Any, Iterable


class BootstrapError(RuntimeError):
    pass


_SENTINEL = object()
_ACTIVE_SENTINEL: object | None = None
_ACTIVE_REPO: Path | None = None
_ACTIVE_TARGET: Path | None = None
_PRE_ATTESTATION: dict[str, Any] | None = None
_LOADED_BY_KEY: dict[str, ModuleType] = {}
_PROTECTED_OPENS: list[str] = []

MANDATORY_MODULES = {
    "rosetta": "rosetta",
    "aligner": "rosetta.model.aligner",
    "dataset_adapters": "rosetta.train.dataset_adapters",
    "evaluate": "rosetta.utils.evaluate",
}

SCRIPT_MODULES = {
    "fpct_1b_audit": "script/analysis/fpct_1b_structural_support_audit.py",
    "fpct_3_5_audit": "script/analysis/fpct_3_5_alignment_correctness.py",
    "fpct_3_7_audit": "script/analysis/fpct_3_7_certified_support_audit.py",
}

GPU_CLOSURE_MODULES = {
    "projector": "rosetta.model.projector",
    "wrapper": "rosetta.model.wrapper",
    "fpct_attention": "rosetta.model.fpct_attention",
}

GPU_SCRIPT_MODULES = {
    "sft_train": "script/train/SFT_train.py",
    "unified_evaluator": "script/evaluation/unified_evaluator.py",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_sha(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _real_absolute(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise BootstrapError(f"{label} must be absolute")
    resolved = path.resolve(strict=True)
    if str(path) != str(resolved):
        raise BootstrapError(f"{label} must be an absolute realpath")
    return resolved


def _real_absolute_parent(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise BootstrapError("attestation output must be absolute")
    parent = path.parent.resolve(strict=True)
    if str(path.parent) != str(parent):
        raise BootstrapError("attestation output parent must be a realpath")
    return parent / path.name


def _git(repo: Path, *args: str, allow_failure: bool = False) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        if allow_failure:
            return None
        raise BootstrapError(
            f"git {' '.join(args)} failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if event != "open" or not args:
        return
    raw = args[0]
    if not isinstance(raw, (str, bytes, os.PathLike)):
        return
    try:
        value = os.fsdecode(raw)
    except Exception:
        return
    if value.startswith("/netdisk/"):
        _PROTECTED_OPENS.append(value)


def _module_origin(module: ModuleType) -> Path | None:
    origin = getattr(getattr(module, "__spec__", None), "origin", None)
    file_value = getattr(module, "__file__", None)
    candidate = file_value or origin
    if not candidate or candidate in {"built-in", "frozen", "namespace"}:
        return None
    return Path(candidate).resolve(strict=True)


def _assert_under(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BootstrapError(f"{label} outside expected source root: {path}") from exc


def _load_script_module(key: str, path: Path) -> ModuleType:
    name = f"_fpct_sealed_{key}"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise BootstrapError(f"cannot create module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _method_contract() -> dict[str, Any]:
    aligner_module = _LOADED_BY_KEY["aligner"]
    token_aligner = getattr(aligner_module, "TokenAligner", None)
    if token_aligner is None:
        raise BootstrapError("TokenAligner missing")
    if token_aligner.__module__ != "rosetta.model.aligner":
        raise BootstrapError("TokenAligner.__module__ mismatch")
    source = Path(inspect.getsourcefile(token_aligner) or "").resolve(strict=True)
    expected = (_ACTIVE_REPO / "rosetta/model/aligner.py").resolve(strict=True)
    if source != expected:
        raise BootstrapError(f"TokenAligner source mismatch: {source}")
    required = (
        "align_chat_messages_soft",
        "sanitize_fpct_soft_alignment",
    )
    methods: dict[str, Any] = {}
    for name in required:
        value = getattr(token_aligner, name, None)
        if value is None:
            raise BootstrapError(f"TokenAligner.{name} missing")
        signature = str(inspect.signature(value))
        methods[name] = {
            "signature": signature,
            "signature_sha256": _sha256_bytes(signature.encode("utf-8")),
        }
    if "apply_confidence_control" not in methods["align_chat_messages_soft"][
        "signature"
    ]:
        raise BootstrapError("align_chat_messages_soft signature mismatch")
    strategy = getattr(aligner_module, "AlignmentStrategy", None)
    if strategy is None or not hasattr(strategy, "EXACT_IDENTITY"):
        raise BootstrapError("AlignmentStrategy.EXACT_IDENTITY missing")
    return {
        "class_module": token_aligner.__module__,
        "source": str(source),
        "methods": methods,
    }


def _distribution_metadata() -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution("rosetta")
    except importlib.metadata.PackageNotFoundError:
        return {"installed": False}
    direct_url_text = distribution.read_text("direct_url.json")
    try:
        direct_url = json.loads(direct_url_text) if direct_url_text else None
    except json.JSONDecodeError:
        direct_url = {"invalid_json": direct_url_text}
    return {
        "installed": True,
        "name": distribution.metadata.get("Name"),
        "version": distribution.version,
        "location": str(Path(distribution.locate_file("")).resolve()),
        "direct_url": direct_url,
    }


def _module_record(key: str, module: ModuleType, repo: Path) -> dict[str, Any]:
    origin = _module_origin(module)
    if origin is None:
        raise BootstrapError(f"mandatory module has no file origin: {key}")
    _assert_under(origin, repo, key)
    spec_origin = getattr(getattr(module, "__spec__", None), "origin", None)
    return {
        "name": module.__name__,
        "spec_origin": spec_origin,
        "file": str(origin),
        "sha256": sha256_file(origin),
    }


def _loaded_rosetta_modules(repo: Path) -> list[dict[str, Any]]:
    expected_root = (repo / "rosetta").resolve(strict=True)
    records: list[dict[str, Any]] = []
    for name, module in sorted(sys.modules.items()):
        if name != "rosetta" and not name.startswith("rosetta."):
            continue
        if module is None:
            continue
        origin = _module_origin(module)
        paths = [
            str(Path(item).resolve(strict=True))
            for item in list(getattr(module, "__path__", []))
        ]
        if origin is not None:
            _assert_under(origin, expected_root, name)
        for item in paths:
            _assert_under(Path(item), expected_root, f"{name}.__path__")
        records.append(
            {
                "name": name,
                "origin": str(origin) if origin is not None else None,
                "path": paths,
                "sha256": sha256_file(origin) if origin is not None else None,
            }
        )
    rosetta = sys.modules.get("rosetta")
    if rosetta is None:
        raise BootstrapError("rosetta not imported")
    package_paths = [
        str(Path(item).resolve(strict=True))
        for item in list(getattr(rosetta, "__path__", []))
    ]
    if package_paths != [str(expected_root)]:
        raise BootstrapError(f"rosetta.__path__ is not unique: {package_paths}")
    return records


def _stable_sys_path(repo: Path) -> list[str]:
    """Normalize only verified-empty interpreter-created temporary entries.

    Torch distributed imports append a fresh ``/tmp/tmpXXXX`` directory that
    contains a generated ``_remote_module_non_scriptable.py``.  Its directory
    name is random; the generated source bytes are the relevant identity.
    Any other temporary payload remains forbidden rather than normalized.
    """

    stable: list[str] = []
    expected_rosetta = (repo / "rosetta").resolve(strict=True)
    for raw in sys.path:
        if not raw or not os.path.isabs(raw):
            stable.append(raw)
            continue
        path = Path(raw).resolve(strict=False)
        foreign_rosetta = path / "rosetta"
        if foreign_rosetta.exists() and foreign_rosetta.resolve() != expected_rosetta:
            raise BootstrapError(
                f"sys.path contains a foreign rosetta candidate: {foreign_rosetta}"
            )
        if path.parent == Path("/tmp") and path.name.startswith("tmp"):
            if not path.is_dir():
                raise BootstrapError(
                    f"ephemeral sys.path entry is not a directory: {path}"
                )
            entries = sorted(item.name for item in path.iterdir())
            allowed = {"_remote_module_non_scriptable.py", "__pycache__"}
            if set(entries) != allowed:
                raise BootstrapError(
                    f"unexpected ephemeral sys.path payload: {path}: {entries}"
                )
            source = path / "_remote_module_non_scriptable.py"
            if not source.is_file():
                raise BootstrapError("generated torch remote-module source missing")
            cached = path / "__pycache__"
            if not cached.is_dir() or any(
                item.is_dir()
                or not item.name.startswith("_remote_module_non_scriptable.")
                or item.suffix != ".pyc"
                for item in cached.iterdir()
            ):
                raise BootstrapError("unexpected generated remote-module cache")
            stable.append(
                "/tmp/<torch-remote-module-sha256="
                f"{sha256_file(source)}>"
            )
            continue
        stable.append(str(path) if path.exists() else raw)
    return stable


def _closure(repo: Path, include_gpu: bool) -> dict[str, ModuleType]:
    loaded: dict[str, ModuleType] = {}
    for key, module_name in MANDATORY_MODULES.items():
        loaded[key] = importlib.import_module(module_name)
    for key, relative in SCRIPT_MODULES.items():
        loaded[key] = _load_script_module(key, (repo / relative).resolve(strict=True))
    if include_gpu:
        for key, module_name in GPU_CLOSURE_MODULES.items():
            loaded[key] = importlib.import_module(module_name)
        for key, relative in GPU_SCRIPT_MODULES.items():
            loaded[key] = _load_script_module(
                key, (repo / relative).resolve(strict=True)
            )
    return loaded


def _attest(repo: Path, target: Path, include_gpu: bool) -> dict[str, Any]:
    executable = Path(sys.executable)
    if not executable.is_absolute():
        raise BootstrapError("sys.executable is not absolute")
    executable_real = executable.resolve(strict=True)
    if str(executable) != str(executable_real):
        raise BootstrapError("interpreter must be invoked through its realpath")
    original_interpreter = Path(sys.orig_argv[0])
    if not original_interpreter.is_absolute():
        raise BootstrapError("interpreter invocation must be absolute")
    original_interpreter_real = original_interpreter.resolve(strict=True)
    if str(original_interpreter) != str(original_interpreter_real):
        raise BootstrapError("interpreter invocation must use its realpath")
    if original_interpreter_real != executable_real:
        raise BootstrapError("interpreter invocation and sys.executable differ")
    if sys.flags.isolated != 1:
        raise BootstrapError("Python isolated mode is required")
    if sys.flags.no_user_site != 1:
        raise BootstrapError("user site must be disabled")
    if Path.cwd().resolve(strict=True) != repo:
        raise BootstrapError("cwd must equal the sealed repo root")
    init_path = repo / "rosetta/__init__.py"
    if not init_path.is_file():
        raise BootstrapError("rosetta/__init__.py is required")

    mandatory_records = {
        key: _module_record(key, module, repo)
        for key, module in sorted(_LOADED_BY_KEY.items())
    }
    method_contract = _method_contract()
    loaded_rosetta = _loaded_rosetta_modules(repo)
    git_head = _git(repo, "rev-parse", "HEAD")
    git_branch = _git(repo, "branch", "--show-current")
    git_upstream = _git(repo, "rev-parse", "@{upstream}", allow_failure=True)
    git_status = _git(repo, "status", "--short")
    full = {
        "schema_version": 1,
        "repo_root": str(repo),
        "target": str(target),
        "executable": {
            "path": str(executable_real),
            "sha256": sha256_file(executable_real),
            "invocation": str(original_interpreter_real),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "abi": sysconfig.get_config_var("SOABI"),
            "cache_tag": sys.implementation.cache_tag,
            "flags": {
                "isolated": sys.flags.isolated,
                "no_user_site": sys.flags.no_user_site,
                "ignore_environment": sys.flags.ignore_environment,
            },
        },
        "process": {
            "cwd": str(Path.cwd().resolve()),
            "argv": list(sys.argv),
            "sys_path": list(sys.path),
            "stable_sys_path": _stable_sys_path(repo),
            "meta_path_types": [
                f"{type(item).__module__}.{type(item).__qualname__}"
                for item in sys.meta_path
            ],
        },
        "git": {
            "head": git_head,
            "branch": git_branch,
            "upstream": git_upstream,
            "clean": not bool(git_status),
            "status": git_status,
        },
        "mandatory_modules": mandatory_records,
        "loaded_rosetta_modules": loaded_rosetta,
        "method_contract": method_contract,
        "distribution": _distribution_metadata(),
        "protected_data_opens_before_target": list(_PROTECTED_OPENS),
        "include_gpu_closure": include_gpu,
    }
    stable = {
        "repo_root": full["repo_root"],
        "executable": full["executable"],
        "python": full["python"],
        "sys_path": full["process"]["stable_sys_path"],
        "meta_path_types": full["process"]["meta_path_types"],
        "git": {
            "head": git_head,
            "branch": git_branch,
            "upstream": git_upstream,
            "clean": full["git"]["clean"],
        },
        "mandatory_modules": mandatory_records,
        "method_contract": method_contract,
        "distribution": full["distribution"],
        "include_gpu_closure": include_gpu,
    }
    full["stable_projection"] = stable
    full["stable_fingerprint_sha256"] = _stable_sha(stable)
    return full


def require_active(*, target: str | Path | None = None) -> dict[str, Any]:
    if _ACTIVE_SENTINEL is not _SENTINEL or _PRE_ATTESTATION is None:
        raise BootstrapError("formal FPCT entrypoint requires canonical bootstrap")
    if target is not None:
        expected = Path(target).resolve(strict=True)
        if expected != _ACTIVE_TARGET:
            raise BootstrapError("bootstrap target mismatch")
    return _PRE_ATTESTATION


def loaded_module(key: str) -> ModuleType:
    require_active()
    try:
        return _LOADED_BY_KEY[key]
    except KeyError as exc:
        raise BootstrapError(f"sealed module not loaded: {key}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp.replace(path)


def _parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--attestation-out")
    parser.add_argument("--expected-attestation")
    parser.add_argument(
        "--expected-module-sha",
        action="append",
        default=[],
        metavar="KEY=SHA256",
    )
    parser.add_argument("--include-gpu-closure", action="store_true")
    parser.add_argument("target_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.target_args and args.target_args[0] == "--":
        args.target_args = args.target_args[1:]
    return args


def main(argv: Iterable[str] | None = None) -> int:
    global _ACTIVE_REPO, _ACTIVE_SENTINEL, _ACTIVE_TARGET
    global _LOADED_BY_KEY, _PRE_ATTESTATION

    args = _parse_args(argv)
    repo = _real_absolute(args.repo_root, "repo root")
    target = _real_absolute(args.target, "target")
    bootstrap = Path(sys.argv[0]).resolve(strict=True)
    expected_bootstrap = (repo / "script/runtime/fpct_bootstrap.py").resolve(
        strict=True
    )
    if bootstrap != expected_bootstrap:
        raise BootstrapError("bootstrap path does not match sealed repo")
    _assert_under(target, repo, "target")
    if Path.cwd().resolve(strict=True) != repo:
        raise BootstrapError("cwd must equal the sealed repo root")
    if sys.flags.isolated != 1 or sys.flags.ignore_environment != 1:
        raise BootstrapError("canonical invocation requires python -I")
    init_path = repo / "rosetta/__init__.py"
    if not init_path.is_file():
        raise BootstrapError("rosetta/__init__.py is required before import sealing")
    if any(name == "rosetta" or name.startswith("rosetta.") for name in sys.modules):
        raise BootstrapError("rosetta was imported before bootstrap sealing")
    if "fpct_bootstrap" in sys.modules:
        raise BootstrapError("bootstrap alias already present")

    os.environ.pop("PYTHONPATH", None)
    os.environ.pop("PYTHONHOME", None)
    sys.addaudithook(_audit_hook)
    sys.path.insert(0, str(repo))
    importlib.invalidate_caches()

    _ACTIVE_REPO = repo
    _ACTIVE_TARGET = target
    _ACTIVE_SENTINEL = _SENTINEL
    sys.modules["fpct_bootstrap"] = sys.modules[__name__]
    _LOADED_BY_KEY = _closure(repo, args.include_gpu_closure)
    # Establish a temporary sealed state so the formal target can import its
    # bootstrap guard without executing its main function.  Then add that exact
    # target file to the mandatory same-process closure and recompute the lock.
    _PRE_ATTESTATION = _attest(repo, target, args.include_gpu_closure)
    _LOADED_BY_KEY["formal_target"] = _load_script_module(
        "formal_target", target
    )
    _PRE_ATTESTATION = _attest(repo, target, args.include_gpu_closure)
    if _PRE_ATTESTATION["protected_data_opens_before_target"]:
        raise BootstrapError("bootstrap probe touched protected natural-data paths")

    if args.expected_attestation:
        expected_path = _real_absolute(
            args.expected_attestation, "expected attestation"
        )
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        expected_fingerprint = expected.get("stable_fingerprint_sha256")
        if expected_fingerprint != _PRE_ATTESTATION[
            "stable_fingerprint_sha256"
        ]:
            raise BootstrapError("freeze/shard attestation fingerprint mismatch")
    for item in args.expected_module_sha:
        if "=" not in item:
            raise BootstrapError("expected module SHA must be KEY=SHA256")
        key, expected_sha = item.split("=", 1)
        record = _PRE_ATTESTATION["mandatory_modules"].get(key)
        if record is None:
            raise BootstrapError(f"unknown sealed module key: {key}")
        if record["sha256"] != expected_sha:
            raise BootstrapError(f"sealed module SHA mismatch: {key}")

    target_argv = [str(target), *args.target_args]
    prior_argv = sys.argv
    exit_code = 0
    target_error: BaseException | None = None
    try:
        sys.argv = target_argv
        runpy.run_path(str(target), run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            target_error = exc
            exit_code = int(exc.code) if isinstance(exc.code, int) else 1
    except BaseException as exc:
        target_error = exc
        exit_code = 1
    finally:
        sys.argv = prior_argv

    post_attestation_error: BaseException | None = None
    try:
        post_attestation = _attest(repo, target, args.include_gpu_closure)
        if post_attestation["stable_fingerprint_sha256"] != _PRE_ATTESTATION[
            "stable_fingerprint_sha256"
        ]:
            raise BootstrapError(
                "sealed module closure changed while the target was running"
            )
    except BaseException as exc:
        post_attestation_error = exc
        post_attestation = None
        exit_code = 1

    post = dict(_PRE_ATTESTATION)
    post["post_target_attestation"] = post_attestation
    post["loaded_rosetta_modules_after_target"] = (
        post_attestation["loaded_rosetta_modules"]
        if post_attestation is not None
        else _loaded_rosetta_modules(repo)
    )
    post["protected_data_opens_after_target"] = list(_PROTECTED_OPENS)
    post["target_exit_code"] = exit_code
    if args.attestation_out:
        output_path = _real_absolute_parent(args.attestation_out)
        _write_json(output_path, post)
    print(
        json.dumps(
            {
                "fpct_bootstrap": "SEALED",
                "stable_fingerprint_sha256": post[
                    "stable_fingerprint_sha256"
                ],
                "target_exit_code": exit_code,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if post_attestation_error is not None:
        raise post_attestation_error
    if target_error is not None:
        raise target_error
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
