from pathlib import Path

from sd_train.domain.path_rules import (
    expected_hf_mode_for_key,
    expected_local_mode_for_key,
    is_path_value_key,
    resolve_path_from_config,
    sanitize_component,
    to_abs_remote,
    to_sync_remote,
)


def test_remote_path_roundtrip() -> None:
    home = "/home/ubuntu"
    sync = "~/.sd-train/runs/x"
    abs_path = to_abs_remote(sync, home)
    assert abs_path == "/home/ubuntu/.sd-train/runs/x"
    assert to_sync_remote(abs_path, home) == sync


def test_path_rule_helpers(tmp_path: Path) -> None:
    assert is_path_value_key("network_weights")
    assert is_path_value_key("custom_path")
    assert expected_hf_mode_for_key("network_weights", has_subpath=True) == "file"
    assert expected_hf_mode_for_key("resume", has_subpath=False) == "any"
    assert expected_local_mode_for_key("dataset_config") == "file"
    assert sanitize_component("a/b c?d") == "a_b_c_d"
    assert to_abs_remote("~", "/home/ubuntu") == "/home/ubuntu"
    assert to_sync_remote("/other/place", "/home/ubuntu") == "/other/place"
    base = tmp_path / "x" / "train.toml"
    base.parent.mkdir(parents=True)
    assert resolve_path_from_config("./data", base) == (base.parent / "data").resolve()
