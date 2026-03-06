from pathlib import Path

import toml

from sd_train.config.models import AppConfig


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        config = AppConfig()
        save_config(path, config)
        return config
    return AppConfig(**toml.load(path))


def save_config(path: Path, config: AppConfig) -> None:
    path.write_text(toml.dumps(config.model_dump()), encoding="utf-8")
