# sd-train Developer Notes

This document contains development, implementation, and operator-facing details that were moved out of the user-facing [README.md](README.md).

## Project Summary

`sd-train` is a Textual TUI launcher for SDXL/LoRA training workflows on remote machines, with:

- Local, SSH, and Vast.ai environment support
- remote `kohya-ss/sd-scripts` execution
- automatic artifact sync via `rclone`
- built-in dataset tagging tools
- external model materialization from Hugging Face and CivitAI

## Install

```bash
uv sync
```

## Run

```bash
uv run launch
```

Alternative entrypoint:

```bash
uv run -m sd_train.cli
```

## Development

Run tests:

```bash
uv run pytest
```

Type check:

```bash
uv run pyright
```

## Runtime Requirements

Local:

- Python `>= 3.12`
- `uv`
- `rclone`

Remote:

- `python3`
- `rclone`
- `git`

Notes:

- `sd_train.core.execution` clones and pins `~/.sd-train/sd-scripts` automatically.
- If remote `uv` is missing, the launcher attempts to install it automatically.
- Remote setup can still fail if required tools such as `git` or `curl` are unavailable, or if the remote cannot access the network.

## Configuration

Configuration is stored in `config.toml`.

Top-level sections:

- `[[environments]]`: built-in local environment plus SSH or Vast.ai environments
- `[last]`: last selected environment/config/script
- `[tagger]`: tagger defaults
- `[other_options]`: global auth options used by all environments

Example:

```toml
[other_options]
hf_token = ""
civitai_api_key = ""
```

## Launcher Flow

1. Select the built-in local environment or create a remote environment.
2. Select `train.toml`.
3. Select a training script.
4. Configure auth in `Other Options` if needed.
5. Start training.

## Training Config Behavior

Path-like keys in `train.toml` are interpreted as:

- existing local path: uploaded to the remote artifacts cache
- non-existing path matching an external reference: downloaded on the remote

Output-related keys are rewritten to remote run directories:

- `output_dir`
- `logging_dir`

## External Reference Formats

### Hugging Face

Accepted forms:

- `model:org/repo`
- `dataset:org/repo`
- `org/repo`
- `model:org/repo@revision`
- `model:org/repo::file.safetensors`
- `model:org/repo@revision::file.safetensors`

Notes:

- File mode requires `::subpath`.
- If `huggingface_hub` is missing on the remote, it is installed at runtime.

### CivitAI

Accepted forms:

- `civitai:46846`
- `civitai:https://civitai.com/models/1234567?modelVersionId=46846`
- `civitai:https://civitai.com/api/download/models/46846`
- `civitai:46846::my-model.safetensors`

Notes:

- Downloads use `ashleykleynhans/civitai-downloader` on the remote.
- `civitai_api_key` is required.
- Numeric ids are auto-resolved:
  - `modelVersionId` is used directly when valid.
  - `modelId` resolves to the latest version id.

## Preflight Checks

Before training starts, the app validates:

- `train.toml` structure and required keys
- local dataset/file path consistency
- external reference format validity
- remote accessibility of referenced Hugging Face/CivitAI assets

If validation fails, training does not start.

## Output Sync

- Periodic sync every `SYNC_INTERVAL_SECONDS` (default `15`)
- Stability window `SYNC_STABILITY_SECONDS` (default `20`)
- Final full output sync after training ends
- Transfer progress remains enabled during the session

## Tagger Workspace

Capabilities:

- scan dataset tag stats
- auto-generate tags using `SmilingWolf/wd-vit-tagger-v3`
- batch edit captions: add, remove, front, shuffle, delete

## Security Notes

- API keys are stored in plain text in `config.toml`.
- Prefer short-lived or limited-scope credentials.
- Do not commit `config.toml` with real secrets.

## Project Layout

- `sd_train/cli.py`: CLI entrypoint
- `sd_train/app/`: launcher orchestration and training session startup
- `sd_train/core/`: script selection, preflight, environment setup, execution helpers
- `sd_train/domain/`: refs, path rules, preflight domain models
- `sd_train/infra/environment/`: SSH and Vast.ai adapters
- `sd_train/ui/`: Textual apps and screens
- `sd_train/tagger/`: tagging and caption editing logic
- `sd_train/config/`: config models and persistence
- `scripts/graph.py`: dependency graph helper

## Notable Changes

- Root compatibility modules were removed.
- Official run command is `uv run -m sd_train.cli` or `uv run launch`.
- Internal package path uses `sd_train.*`.
