from pathlib import Path

import toml

from sd_train.config.models import AppConfig, normalize_app_config


def load_config(path: Path) -> AppConfig:
    if not path.exists():
        config = normalize_app_config(AppConfig())
        save_config(path, config)
        return config
    return normalize_app_config(AppConfig(**toml.load(path)))


def save_config(path: Path, config: AppConfig) -> None:
    normalized = normalize_app_config(config)
    path.write_text(toml.dumps(normalized.model_dump()), encoding="utf-8")
