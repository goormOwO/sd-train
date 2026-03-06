from pathlib import Path

import pytest

from sd_train.config.models import AppConfig
from sd_train.config.store import load_config, save_config


def test_load_config_creates_default_when_missing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    config = load_config(cfg_path)
    assert isinstance(config, AppConfig)
    assert cfg_path.exists()


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    config = AppConfig()
    config.other_options = {"hf_token": "abc"}
    save_config(cfg_path, config)
    loaded = load_config(cfg_path)
    assert loaded.other_options["hf_token"] == "abc"


def test_load_config_raises_for_invalid_toml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("not = [valid", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(cfg_path)
