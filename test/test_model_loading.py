from pathlib import Path

from rosetta.utils import model_loading


def test_resolve_model_path_prefers_existing_explicit_directory(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit-model"
    explicit.mkdir()

    assert model_loading.resolve_model_path(explicit) == str(explicit)


def test_resolve_model_path_maps_huggingface_id_to_unified_root(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "models"
    local = root / "Qwen3-8B"
    local.mkdir(parents=True)
    monkeypatch.setenv("C2C_MODEL_ROOT", str(root))

    assert model_loading.resolve_model_path("Qwen/Qwen3-8B") == str(local)
    assert model_loading.resolve_model_path(
        "/share/public/public_models/Qwen3-8B"
    ) == str(local)


def test_resolve_model_path_supports_known_directory_alias(tmp_path: Path) -> None:
    local = tmp_path / "Qwen2.5-Math-1.5B-Instruct"
    local.mkdir()

    assert model_loading.resolve_model_path(
        "Qwen/Qwen2.5-Math-1.5B", tmp_path
    ) == str(local)


def test_model_matches_huggingface_ids_and_legacy_paths() -> None:
    assert model_loading.model_matches(
        "google/gemma-3-1b-it", "gemma-3-1b-it"
    )
    assert model_loading.model_matches(
        "/share/public/public_models/gemma-3-1b-it", "gemma-3-1b-it"
    )


def test_resolve_model_path_falls_back_to_original_id(tmp_path: Path) -> None:
    model_id = "other/model-not-downloaded"

    assert model_loading.resolve_model_path(model_id, tmp_path) == model_id
