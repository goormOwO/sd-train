# sd-train

Textual TUI launcher for SDXL/LoRA training workflows on remote machines (SSH or Vast.ai), with automatic artifact sync via `rclone`, built-in dataset tagging tools, and external model materialization from Hugging Face or CivitAI.

## What this project does

- Manages train sessions through a terminal UI.
- Supports two runtime environments:
  - SSH host
  - Vast.ai instance provisioning + connection
- Runs selected `kohya-ss/sd-scripts` training entrypoint on remote.
- Syncs training outputs from remote to local `outputs/<run-id>` continuously.
- Includes dataset tagger workspace (auto-tag, tag edit utilities).
- Resolves model/dataset references from:
  - Local paths
  - Hugging Face refs
  - CivitAI refs
- Validates external references before training starts.

## Requirements

- Python `>= 3.12`
- [uv](https://docs.astral.sh/uv/)
- Local `rclone` installed (used for sync)
- Remote machine with:
  - `python3`
  - `rclone`
  - `uv`
  - `~/.sd-train/sd-scripts` prepared

## Install

```bash
uv sync
```

## Run

```bash
uv run -m sd_train.cli
```

or via entrypoint:

```bash
uv run launch
```

This opens the Textual launcher.

## Launcher flow

1. Select or create an environment.
2. Select `train.toml` path.
3. Select training script.
4. Open `Other Options` to set global API credentials.
5. Start train.

## Configuration

Configuration is stored in `config.toml`.

Top-level sections:

- `[[environments]]`: SSH or Vast.ai environments
- `[last]`: last selected environment/config/script
- `[tagger]`: tagger defaults
- `[other_options]`: global auth options used by all environments

Example `other_options`:

```toml
[other_options]
hf_token = ""
civitai_api_key = ""
```

## Training config behavior (`train.toml`)

Path-like keys are interpreted as:

- Existing local path -> uploaded to remote artifacts cache
- Non-existing path that matches HF/CivitAI reference -> downloaded on remote

Output-related keys are rewritten to remote run directories:

- `output_dir`
- `logging_dir`

## External reference formats

### Hugging Face

Accepted:

- `model:org/repo`
- `dataset:org/repo`
- `org/repo`
- `model:org/repo@revision`
- `model:org/repo::file.safetensors`
- `model:org/repo@revision::file.safetensors`

Notes:

- File mode requires `::subpath`.
- If `huggingface_hub` is missing on remote, it is auto-installed at runtime.

### CivitAI

Accepted:

- `civitai:46846`
- `civitai:https://civitai.com/models/1234567?modelVersionId=46846`
- `civitai:https://civitai.com/api/download/models/46846`
- Optional filename override: `civitai:46846::my-model.safetensors`

Notes:

- CivitAI download uses `ashleykleynhans/civitai-downloader` script fetched on remote:
  - `~/.sd-train/bin/civitai-downloader.py`
- `civitai_api_key` in `Other Options` is required for CivitAI refs.
- Numeric ids are auto-resolved:
  - If value is a valid `modelVersionId`, it is used directly.
  - If value is a `modelId`, the latest version id is resolved automatically.

## Preflight checks (before start)

Before training starts, the app validates:

- `train.toml` structure and required keys
- local dataset/file path consistency
- external reference format validity
- external resource accessibility:
  - Hugging Face repo/file access
  - CivitAI model version download endpoint access

If a check fails, training does not start and the error is shown in preflight screen.

## Output sync

- Periodic sync every `SYNC_INTERVAL_SECONDS` (default 15s)
- Stability window `SYNC_STABILITY_SECONDS` (default 20s) to avoid copying temp/incomplete files
- Final full output sync after training ends
- Transfer progress is enabled for the whole training session

## Tagger workspace

From launcher, choose `Tagger Workspace` to:

- scan dataset tag stats
- auto-generate tags using `SmilingWolf/wd-vit-tagger-v3` (default)
- batch edit captions (add/remove/front/shuffle/delete)

## Security notes

- API keys are stored in plain text in `config.toml`.
- Prefer using short-lived/limited-scope tokens.
- Do not commit `config.toml` with real credentials.

## Project files

- `sd_train/cli.py`: application entrypoint
- `sd_train/app/`: launcher orchestration and training session execution
- `sd_train/domain/`: preflight, reference parsing/validation, path rules
- `sd_train/infra/environment/`: SSH/Vast.ai environment adapters and types
- `sd_train/ui/`: Textual apps/screens
- `sd_train/tagger/`: tagging and caption editing logic
- `scripts/graph.py`: import analysis -> DOT -> PNG renderer
- `docs/dependency-graph.dot`: graph source
- `docs/dependency-graph.png`: dependency graph image

## Breaking changes

- Root compatibility modules were removed (`train.py`, `tui.py`, `ssh.py`, `vastai_client.py`, `tagger.py`, `environment.py`).
- Official run command is now `uv run -m sd_train.cli`.
- Internal Python package path changed from `sdxl_train.*` to `sd_train.*`.
