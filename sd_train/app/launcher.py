from pathlib import Path
from types import SimpleNamespace

from colorama import Fore, Style
from colorama import init as colorama_init
from pydantic import ValidationError

from sd_train.core.dataset_detection import guess_dataset_dir_from_train_config
from sd_train.core.script_selection import scan_train_scripts
from sd_train.app.start import start_training
from sd_train.config.models import (
    AppConfig,
    DEFAULT_VASTAI_OFFER_QUERY,
    LastSelection,
    normalize_app_config,
)
from sd_train.config.store import load_config, save_config
from sd_train.ui.apps.launcher import TaggerWorkspaceApp, TrainLauncherApp

CONFIG_PATH = Path("config.toml")


def run_last_training() -> bool:
    colorama_init()
    config: AppConfig = normalize_app_config(load_config(CONFIG_PATH))

    script_options, scan_error = scan_train_scripts()
    if scan_error is not None:
        print(f"{Fore.YELLOW}{scan_error}{Style.RESET_ALL}")

    if not config.last.train_script and script_options:
        preferred = "train_network.py"
        config.last.train_script = preferred if preferred in script_options else script_options[0]

    normalize_app_config(config)
    save_config(CONFIG_PATH, config)

    result = SimpleNamespace(
        action="start",
        selection=SimpleNamespace(
            environment_name=config.last.environment_name,
            train_config_path=config.last.train_config_path or "train.toml",
            train_script=config.last.train_script,
        ),
    )

    started = start_training(config, result, script_options, require_confirmation=False)
    normalize_app_config(config)
    save_config(CONFIG_PATH, config)
    return started


def main() -> None:
    colorama_init()
    config: AppConfig = normalize_app_config(load_config(CONFIG_PATH))

    script_options, scan_error = scan_train_scripts()
    if scan_error is not None:
        print(f"{Fore.YELLOW}{scan_error}{Style.RESET_ALL}")

    try:
        while True:
            app = TrainLauncherApp(
                environments=config.environments,
                last=config.last.model_dump(),
                other_options=config.other_options,
                default_offer_query=DEFAULT_VASTAI_OFFER_QUERY,
                train_script_options=script_options,
            )
            result = app.run()
            if result is None:
                return

            config.environments = result.environments
            config.last = LastSelection(**result.last)
            config.other_options = {
                "hf_token": str(result.other_options.get("hf_token", "")).strip(),
                "civitai_api_key": str(result.other_options.get("civitai_api_key", "")).strip(),
            }
            if not config.last.train_script and script_options:
                preferred = "train_network.py"
                config.last.train_script = preferred if preferred in script_options else script_options[0]
            normalize_app_config(config)
            save_config(CONFIG_PATH, config)

            if result.action == "start":
                started = start_training(config, result, script_options)
                if started:
                    return
                continue

            if result.action == "tagger":
                dataset_dir = config.tagger.dataset_dir
                if not dataset_dir:
                    dataset_dir = guess_dataset_dir_from_train_config(config.last.train_config_path or "train.toml")
                workspace = TaggerWorkspaceApp(
                    dataset_dir=dataset_dir,
                    model=config.tagger.model,
                    threshold=config.tagger.threshold,
                    batch=config.tagger.batch,
                )
                workspace_result = workspace.run()
                if workspace_result is not None:
                    config.tagger.dataset_dir = workspace_result.dataset_dir
                    config.tagger.model = workspace_result.model
                    config.tagger.threshold = workspace_result.threshold
                    config.tagger.batch = workspace_result.batch
                    normalize_app_config(config)
                    save_config(CONFIG_PATH, config)
                    if workspace_result.action == "quit":
                        return
                continue

            if result.action == "quit":
                return
    except KeyboardInterrupt:
        print(f"{Fore.YELLOW}Interrupted by user. Cleaning up...{Style.RESET_ALL}")
    except ValidationError as exc:
        print(f"{Fore.RED}Validation error: {exc}{Style.RESET_ALL}")
    except Exception as exc:
        print(f"{Fore.RED}Error: {exc}{Style.RESET_ALL}")
    finally:
        normalize_app_config(config)
        save_config(CONFIG_PATH, config)
