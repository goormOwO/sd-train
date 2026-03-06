import re
from pathlib import Path
from typing import Literal

PATH_VALUE_KEYS = {
    "pretrained_model_name_or_path",
    "train_data_dir",
    "reg_data_dir",
    "sample_prompts",
    "dataset_config",
    "in_json",
    "network_weights",
    "vae",
    "qwen3",
    "resume",
    "tokenizer_cache_dir",
    "output_dir",
    "logging_dir",
}
PATH_KEY_SUFFIXES = ("_dir", "_path", "_file", "_json", "_weights", "_prompts")
FORCE_REMOTE_OUTPUT_KEYS = {"output_dir", "logging_dir"}


def is_path_value_key(key: str) -> bool:
    lowered = key.lower()
    if lowered in PATH_VALUE_KEYS:
        return True
    return lowered.endswith(PATH_KEY_SUFFIXES)


def sanitize_component(value: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9._-]", "_", value)
    return safe or "item"


def to_abs_remote(sync_path: str, remote_home: str) -> str:
    if sync_path == "~":
        return remote_home
    if sync_path.startswith("~/"):
        return remote_home + sync_path[1:]
    return sync_path


def to_sync_remote(abs_path: str, remote_home: str) -> str:
    if abs_path == remote_home:
        return "~"
    if abs_path.startswith(remote_home + "/"):
        return "~" + abs_path[len(remote_home) :]
    return abs_path


def resolve_path_from_config(value: str, base_path: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_path.parent / path).resolve()
    return path


def expected_hf_mode_for_key(key: str, *, has_subpath: bool) -> Literal["file", "dir", "any"]:
    lowered = key.lower()
    if lowered in {"train_data_dir", "reg_data_dir", "tokenizer_cache_dir"}:
        return "dir"
    if lowered in {"sample_prompts", "dataset_config", "in_json", "network_weights", "qwen3"}:
        return "file"
    if lowered in {"output_dir", "logging_dir"}:
        return "dir"
    if lowered == "pretrained_model_name_or_path":
        return "file" if has_subpath else "dir"
    if lowered in {"vae", "resume"}:
        return "file" if has_subpath else "any"
    if lowered.endswith("_dir"):
        return "dir"
    if lowered.endswith(("_file", "_json", "_prompts", "_weights")):
        return "file"
    return "any"


def expected_local_mode_for_key(key: str) -> Literal["file", "dir", "any"]:
    lowered = key.lower()
    if lowered in {"train_data_dir", "reg_data_dir", "tokenizer_cache_dir"}:
        return "dir"
    if lowered in {"sample_prompts", "dataset_config", "in_json", "network_weights", "qwen3"}:
        return "file"
    if lowered in {"output_dir", "logging_dir"}:
        return "dir"
    if lowered.endswith("_dir"):
        return "dir"
    if lowered.endswith(("_file", "_json", "_prompts", "_weights")):
        return "file"
    return "any"
