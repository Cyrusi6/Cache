# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment and commands

### Setup
- Python requirement: `>=3.10`
- Create env from README:
  - `conda create -n rosetta python=3.10`
  - `conda activate rosetta`
- Install package:
  - `pip install -e .`
- Install training/eval extras:
  - `pip install -e ".[training,evaluation]"`
- Install dev tools (black, isort, flake8, mypy) needed for formatting/lint commands:
  - `pip install -e ".[dev]"`
- There is also a pinned Conda environment in `environment.yml` (note: contains a hardcoded user `prefix` path that needs customization).

### Training
- Main training entrypoint:
  - `python script/train/SFT_train.py --config recipe/train_recipe/C2C_0.6+0.5.json`
- Single-GPU launcher used in the repo:
  - `bash bash/train/sft_train.sh`
- Include-response variant:
  - `bash bash/train/include_response.sh`
- Multi-GPU example from README:
  - `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 script/train/SFT_train.py --config recipe/train_recipe/C2C_0.6+0.5.json`
- Eval-only mode on a training config:
  - `python script/train/SFT_train.py --config <train-config> --eval_only`

### Evaluation
- Main evaluation entrypoint:
  - `python script/evaluation/unified_evaluator.py --config recipe/eval_recipe/unified_eval.yaml`
- Repo launcher:
  - `bash bash/eval/run_eval.sh`
- For a targeted run, make a copy of an eval recipe and set `eval.subjects` and/or `eval.limit` in the YAML; the evaluator CLI only accepts `--config`.

### Playground / manual validation
- Interactive C2C chat with one or more checkpoints:
  - `python script/playground/live_chat_example.py --checkpoint_dir path/to/checkpoint`
  - `python script/playground/live_chat_example.py --checkpoint_dir path/to/ckpt1 path/to/ckpt2`
- Gradio demo:
  - `python script/playground/gradio_demo.py`

### Formatting / static checks
- Format Python:
  - `black rosetta script`
  - `isort rosetta script`
- Lint:
  - `flake8 rosetta script`
- Type check:
  - `mypy rosetta`

### Tests
- `pyproject.toml` configures `pytest`, but the repository currently does not contain a `test/` directory.
- Use targeted evaluation runs through `script/evaluation/unified_evaluator.py` as the main validation path unless a test suite is added.

## High-level architecture

### Big picture
This repo implements Cache-to-Cache (C2C) / Rosetta: a wrapper around a base causal LM plus one or more sharer models. Instead of exchanging text between models, it projects sharer KV caches into the base model’s cache space and then generates from the base model.

The codebase is config-driven:
- training configs live in `recipe/train_recipe/*.json`
- evaluation configs live in `recipe/eval_recipe/*.yaml`
- scripts in `script/` are the operational entrypoints
- `rosetta/` contains the reusable model, training, and evaluation logic

### Core runtime model
- `rosetta/model/wrapper.py` defines `RosettaModel`, the central abstraction.
- `RosettaModel` wraps:
  - `model_list=[base_model, sharer_model1, sharer_model2, ...]`
  - `projector_list`, which maps sharer-layer KV representations into base-model layers.
- Projection routing is controlled by `kv_cache_index`, a `torch.LongTensor` of shape `(seq_len, 2)`:
  - **Column 0** (routing bitmask): `-1` = no projection (label tokens), `1` = project from sharer (instruction tokens), positive bitmask for multi-source.
  - **Column 1** (confidence/source ID): currently always 0 in standard training; reserved for confidence-gated routing.
  - Constructed by `generate_kv_cache_index()` in `rosetta/train/dataset_adapters.py`.
- `RosettaModel.__init__` accepts two important config-driven parameters:
  - `include_response` (bool): whether sharer response tokens are also projected (not just instruction tokens). Controlled by `model.include_response` in training configs.
  - `multi_source_fusion_mode` (str): `"parallel"` = all sources project from a clean base-cache clone and sum; `"sequential"` = each source updates the base cache in order.
- `RosettaModel.generate()` is custom generation logic, not just a thin pass-through to Hugging Face generation.

### Projectors
- Main projector classes in `rosetta/model/projector.py`:
  - `AllInOneProjector` — full-featured projector with SwiGLU, layer norm, configurable granularity. The primary choice.
  - `C2CProjector` — standard projector with gumbel-softmax gates.
  - `RegularMLP` — building-block MLP used internally by the projectors above.
- `rosetta/model/ablation_projector.py` contains `AblationProjector` for ablation studies (levels 0–4).
- `rosetta/model/oracle.py` defines `OracleRosettaModel`, an older variant used by some evaluation paths.
- Projectors are registry-based via `rosetta/utils/registry.py`. New projector types should be registered, then referenced from config by `model.projector.type`.
- Projector base class exposes two extension hooks: `uses_internal_source_confidence()` and `calibrate_source_weights()`.
- A trained checkpoint is expected to contain projector weights/config, not a monolithic model save.

### Training flow
- Main entrypoint: `script/train/SFT_train.py`.
- The trainer auto-detects the mode from config:
  - baseline mode: `baseline_model` (supports LoRA via `peft` when configured).
  - Rosetta mode: `base_model` + `teacher_model`
- In Rosetta mode, the script:
  1. loads base + teacher model(s)
  2. builds projector modules from config
  3. creates layer mappings between source and target models (controlled by the required `model.mapping` config key)
  4. freezes backbone models and trains projector parameters
- **`model.mapping`** is a required config key. Supported values:
  - `"last_aligned"` — aligns the last target layer to the last source layer, walking backward.
  - `"k_nearest"` — maps each target layer to the K nearest source layers by uniform position.
  - Implemented in `rosetta/train/model_utils.py` via `last_aligned_sources()` and `k_nearest_sources()`.
- Auxiliary training losses in new modules:
  - `rosetta/train/answer_margin.py` — `compute_answer_margin_routing_loss()` for option-level hinge/CE loss at MMLU answer positions.
  - `rosetta/train/answer_prior.py` — `compute_answer_prior_regularization()` for KL-divergence based answer prior.
- Alignment helpers live in `rosetta/model/aligner.py`.

### Datasets and batching
- Dataset loading is registry-based in `rosetta/train/dataset_adapters.py`.
- `create_dataset(...)` resolves dataset classes from the registry.
- This file is also where chat formatting, filtering, and `kv_cache_index` construction are centralized.
- `ChatDataset`, `AlignedChatDataset`, and the collators are important if you change prompt structure, alignment logic, or training labels.
- Many training/eval paths render prompts through chat templates with `enable_thinking=False`; preserve that behavior unless there is a clear reason to change prompt format.

### Evaluation flow
- Main entrypoint: `script/evaluation/unified_evaluator.py`.
- It evaluates multiple benchmark families from one config format.
- The evaluator dispatches on `model_name` (case-insensitive):
  - Any Hugging Face model ID — loaded via `load_hf_model()` as a plain causal LM.
  - `"Rosetta"` — requires `model.rosetta_config` sub-section with `base_model`, `teacher_model`, `checkpoints_dir`.
  - `"two_stage"` — two-stage text baseline via `TwoStageInference` in `rosetta/baseline/multi_stage.py`.
  - `"two_stage_rosetta"` — hybrid two-stage + Rosetta via `TwoStageRosetta`.
- `rosetta/utils/evaluate.py` contains the main model-loading helpers and shared evaluation utilities. `load_rosetta_model()` supports both single-teacher (string path) and multi-teacher dict formats (`{"model_name": "path", ...}` with a matching `checkpoints_dir` list).

### Checkpoint layout
- Single-teacher checkpoints:
  - checkpoint root `config.json`
  - a subfolder such as `final/`
  - `projector_<n>.pt`, `projector_<n>.json`
  - `projector_config.json`
- Multi-teacher eval uses a dict format: `teacher_model: {"name": "path", ...}` with `checkpoints_dir: ["ckpt1/", "ckpt2/"]`.
- `load_rosetta_model()` and the playground scripts rely on this layout.

### Recipes are the real interface
Most repo work should start by reading the relevant recipe file before editing code.
- Training recipes define model pairing, projector type/params, optimization settings, output paths, and dataset selection.
- Eval recipes define model loading, generation behavior, benchmark choice, output paths, and GPU assignment.

When a user asks to “run training” or “run evaluation,” the right action is usually to edit or choose the right recipe rather than changing script code.

## Non-obvious conventions

### Multi-sharer selection uses bitmasks
For multi-source inference, sharer selection is encoded as a bitmask in `kv_cache_index[:, 0]` rather than a list of model IDs. Check `rosetta/utils/core.py` and `script/playground/live_chat_example.py` before changing this behavior.

### Source execution is the normal workflow
`pyproject.toml` excludes `script*` from the packaged distribution. Most practical workflows in this repo assume running scripts from the source checkout, not from an installed console entrypoint.

### Some scripts are experimental or stale
Prefer the actively used paths below unless the user explicitly asks otherwise:
- training: `script/train/SFT_train.py`
- evaluation: `script/evaluation/unified_evaluator.py`
- playground: `script/playground/live_chat_example.py`, `script/playground/gradio_demo.py`

## Important repo locations
- `rosetta/model/` — wrapper, projectors, aligner, sampling, oracle wrapper
- `rosetta/train/` — dataset adapters, layer mapping helpers, auxiliary losses (answer_margin, answer_prior)
- `rosetta/utils/` — registries, model loading, shared eval helpers
- `rosetta/baseline/` — text/two-stage baseline pipelines
- `script/train/` — training entrypoints
- `script/evaluation/` — benchmark runners
- `script/playground/` — demos and manual validation flows
- `script/analysis/` — diagnostic scripts (alignment, confidence gates, eval flips)
- `recipe/train_recipe/` — training configs (JSON)
- `recipe/eval_recipe/` — evaluation configs (YAML)
