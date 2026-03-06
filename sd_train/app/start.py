import asyncio
from pathlib import Path
from typing import Any

from colorama import Fore, Style

from sd_train.app.preflight import run_preflight_gate
from sd_train.config.models import AppConfig
from sd_train.core.environment_setup import build_environment, env_to_model
from sd_train.core.execution import run_training_session
from sd_train.core.script_selection import normalize_script_path
from sd_train.core.source_auth import build_download_auth


def _find_environment(config: AppConfig, name: str) -> dict[str, object]:
    for environment in config.environments:
        if str(environment.get("name", "")) == name:
            return environment
    raise ValueError(f"Selected environment not found: {name}")


def start_training(
    config: AppConfig,
    result: Any,
    script_options: list[str],
) -> bool:
    if result.selection is None:
        return False

    script = normalize_script_path(result.selection.train_script)
    if script_options and script not in script_options:
        raise ValueError(f"Selected script is not in GitHub script list: {script}")

    local_train_config_path = Path(result.selection.train_config_path).expanduser().resolve()
    if not local_train_config_path.is_file():
        raise ValueError(f"Train config file not found: {local_train_config_path}")

    auth = build_download_auth(config.other_options)
    gate = run_preflight_gate(local_train_config_path, script, auth)
    if gate is None:
        return False

    env_raw = _find_environment(config, result.selection.environment_name)
    env_model = env_to_model(env_raw)

    print(f"{Fore.CYAN}Running on environment: {result.selection.environment_name} ({env_model.type}){Style.RESET_ALL}")
    asyncio.run(run_training_session(build_environment(env_model), result.selection, auth))
    return True
