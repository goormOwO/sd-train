from pathlib import Path

import pytest

from sd_train.config.models import AppConfig, DEFAULT_LOCAL_ENVIRONMENT_NAME
from sd_train.config.store import load_config, save_config


def test_load_config_creates_default_when_missing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    config = load_config(cfg_path)
    assert isinstance(config, AppConfig)
    assert cfg_path.exists()
    assert config.environments[0]["name"] == DEFAULT_LOCAL_ENVIRONMENT_NAME
    assert config.last.environment_name == DEFAULT_LOCAL_ENVIRONMENT_NAME


def test_save_and_load_roundtrip(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    config = AppConfig()
    config.environments = [{"name": "dev", "type": "ssh", "host": "127.0.0.1"}]
    config.other_options = {"hf_token": "abc"}
    save_config(cfg_path, config)
    loaded = load_config(cfg_path)
    assert loaded.other_options["hf_token"] == "abc"
    assert [env["name"] for env in loaded.environments] == [DEFAULT_LOCAL_ENVIRONMENT_NAME, "dev"]


def test_load_config_restores_builtin_local_when_missing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        """
[[environments]]
name = "dev"
type = "ssh"
host = "127.0.0.1"

[last]
environment_name = "missing"
""".strip(),
        encoding="utf-8",
    )

    loaded = load_config(cfg_path)
    assert [env["name"] for env in loaded.environments] == [DEFAULT_LOCAL_ENVIRONMENT_NAME, "dev"]
    assert loaded.last.environment_name == DEFAULT_LOCAL_ENVIRONMENT_NAME


def test_load_config_raises_for_invalid_toml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("not = [valid", encoding="utf-8")
    with pytest.raises(Exception):
        load_config(cfg_path)
