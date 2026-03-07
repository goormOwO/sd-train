# sd-train

[English](README.md) | [한국어](README.ko.md)

`sd-train` is a terminal UI for running SDXL/LoRA training jobs on remote machines. It helps you pick a training environment, validate your config before launch, start a `kohya-ss/sd-scripts` job remotely, and sync outputs back to your local machine.

## What You Can Do

- Run training on an SSH server or a Vast.ai instance
- Select a `train.toml` file from the launcher
- Choose a training script from `kohya-ss/sd-scripts`
- Download model or dataset assets from Hugging Face or CivitAI
- Sync training outputs to your local `outputs/<run-id>` directory
- Open the built-in tagger workspace for dataset caption work

## Requirements

- Python 3.12+
- `uv`
- Local `rclone`
- A remote environment with:
  - `python3`
  - `rclone`
  - `git`

`sd-train` prepares `~/.sd-train/sd-scripts` on the remote automatically when training starts. If `uv` is missing on the remote, it also tries to install it automatically.

## Install

```bash
uv sync
```

## Run

```bash
uv run launch
```

You can also run:

```bash
uv run -m sd_train.cli
```

On first launch, `config.toml` is created automatically in the project root.

## Quick Start

1. Run `uv run launch`.
2. Create or select an environment.
3. Choose your `train.toml`.
4. Choose the training script.
5. Open `Other Options` if you need a Hugging Face token or CivitAI API key.
6. Start training after the preflight check passes.

## Using Hugging Face and CivitAI Models

You can reference remote assets directly in `train.toml` instead of downloading them manually first. During preflight, `sd-train` validates the reference format and checks whether the remote asset is accessible.

### Hugging Face

Accepted forms:

- `org/repo`
- `model:org/repo`
- `dataset:org/repo`
- `model:org/repo@revision`
- `model:org/repo::file.safetensors`
- `model:org/repo@revision::file.safetensors`

Examples:

```toml
pretrained_model_name_or_path = "model:stabilityai/stable-diffusion-xl-base-1.0"
network_weights = "model:some-user/my-lora::my-lora.safetensors"
dataset_config = "dataset:some-user/my-dataset"
```

Notes:

- Use `::subpath` when a key expects a file such as `.safetensors`.
- If the repo is private or gated, set your Hugging Face token in `Other Options`.
- Bare `org/repo` is treated like a Hugging Face model reference.

### CivitAI

Accepted forms:

- `civitai:46846`
- `civitai:https://civitai.com/models/1234567?modelVersionId=46846`
- `civitai:https://civitai.com/api/download/models/46846`
- `civitai:46846::my-model.safetensors`

Examples:

```toml
pretrained_model_name_or_path = "civitai:46846"
network_weights = "civitai:46846::my-lora.safetensors"
```

Notes:

- Set `civitai_api_key` in `Other Options` before using CivitAI downloads.
- If you provide a model page URL, `sd-train` tries to resolve it to a downloadable version id automatically.
- `::filename` is optional, but helps keep the downloaded file name predictable.

## Tagger Workspace

From the launcher, you can open `Tagger Workspace` to:

- inspect dataset tag statistics
- auto-tag images
- batch edit captions

Default tagger model: `SmilingWolf/wd-vit-tagger-v3`

## Notes

- Credentials are stored in plain text in `config.toml`.
- Avoid committing real tokens or API keys.
- CivitAI downloads require `civitai_api_key`.
