from sd_train.ui.apps.launcher import PreflightReviewApp, TrainLauncherApp


def test_ui_apps_construct() -> None:
    preflight = PreflightReviewApp(summary="ok")
    launcher = TrainLauncherApp(
        environments=[],
        last={"environment_name": "", "train_config_path": "", "train_script": ""},
        other_options={"hf_token": "", "civitai_api_key": ""},
        default_offer_query="x",
        train_script_options=["train_network.py"],
    )
    assert preflight is not None
    assert launcher is not None
