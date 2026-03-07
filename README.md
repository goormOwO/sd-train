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
