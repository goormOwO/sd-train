from types import SimpleNamespace

from pydantic import ValidationError

from sd_train.app import launcher
from sd_train.config.models import AppConfig, DEFAULT_LOCAL_ENVIRONMENT_NAME


def test_launcher_main_exits_when_dialog_returns_none(monkeypatch) -> None:
    saved: list[AppConfig] = []

    class FakeLauncherApp:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN003
            return

        def run(self):
            return None

    monkeypatch.setattr(launcher, "colorama_init", lambda: None)
    monkeypatch.setattr(launcher, "scan_train_scripts", lambda: (["train_network.py"], None))
    monkeypatch.setattr(launcher, "load_config", lambda _p: AppConfig())
    monkeypatch.setattr(launcher, "save_config", lambda _p, cfg: saved.append(cfg))
    monkeypatch.setattr(launcher, "TrainLauncherApp", FakeLauncherApp)
    launcher.main()
    assert len(saved) == 1
    assert saved[0].environments[0]["name"] == DEFAULT_LOCAL_ENVIRONMENT_NAME


def test_launcher_main_start_action_calls_start_training(monkeypatch) -> None:
    calls = {"start": 0, "save": 0}
    result = SimpleNamespace(
        action="start",
        environments=[{"name": "dev", "type": "ssh", "host": "127.0.0.1", "user": "me", "port": 22}],
        last={"environment_name": "dev", "train_config_path": "train.toml", "train_script": "train_network.py"},
        other_options={"hf_token": "a", "civitai_api_key": "b"},
        selection=SimpleNamespace(),
    )

    class FakeLauncherApp:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN003
            return

        def run(self):
            return result

    monkeypatch.setattr(launcher, "colorama_init", lambda: None)
    monkeypatch.setattr(launcher, "scan_train_scripts", lambda: (["train_network.py"], None))
    monkeypatch.setattr(launcher, "load_config", lambda _p: AppConfig())
    monkeypatch.setattr(launcher, "save_config", lambda _p, _cfg: calls.__setitem__("save", calls["save"] + 1))
    monkeypatch.setattr(launcher, "TrainLauncherApp", FakeLauncherApp)
    monkeypatch.setattr(launcher, "start_training", lambda *_args: calls.__setitem__("start", calls["start"] + 1) or True)
    launcher.main()
    assert calls["start"] == 1
    assert calls["save"] >= 1


def test_launcher_main_tagger_flow_updates_config(monkeypatch) -> None:
    saves: list[AppConfig] = []
    seq = iter(
        [
            SimpleNamespace(
                action="tagger",
                environments=[],
                last={"environment_name": "", "train_config_path": "train.toml", "train_script": ""},
                other_options={"hf_token": "", "civitai_api_key": ""},
                selection=None,
            ),
            SimpleNamespace(
                action="quit",
                environments=[],
                last={"environment_name": "", "train_config_path": "train.toml", "train_script": ""},
                other_options={"hf_token": "", "civitai_api_key": ""},
                selection=None,
            ),
        ]
    )

    class FakeLauncherApp:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN003
            return

        def run(self):
            return next(seq)

    class FakeTaggerApp:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN003
            return

        def run(self):
            return SimpleNamespace(
                action="back",
                dataset_dir="/tmp/d",
                model="m",
                threshold=0.5,
                batch=2,
            )

    monkeypatch.setattr(launcher, "colorama_init", lambda: None)
    monkeypatch.setattr(launcher, "scan_train_scripts", lambda: (["train_network.py"], None))
    monkeypatch.setattr(launcher, "load_config", lambda _p: AppConfig())
    monkeypatch.setattr(launcher, "save_config", lambda _p, cfg: saves.append(cfg.model_copy(deep=True)))
    monkeypatch.setattr(launcher, "TrainLauncherApp", FakeLauncherApp)
    monkeypatch.setattr(launcher, "TaggerWorkspaceApp", FakeTaggerApp)
    monkeypatch.setattr(launcher, "guess_dataset_dir_from_train_config", lambda _p: "/tmp/guess")
    launcher.main()
    assert any(cfg.tagger.dataset_dir == "/tmp/d" for cfg in saves)


def test_launcher_main_handles_keyboard_interrupt(monkeypatch) -> None:
    class FakeLauncherApp:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN003
            return

        def run(self):
            raise KeyboardInterrupt

    saves = {"n": 0}
    monkeypatch.setattr(launcher, "colorama_init", lambda: None)
    monkeypatch.setattr(launcher, "scan_train_scripts", lambda: (["train_network.py"], None))
    monkeypatch.setattr(launcher, "load_config", lambda _p: AppConfig())
    monkeypatch.setattr(launcher, "save_config", lambda _p, _cfg: saves.__setitem__("n", saves["n"] + 1))
    monkeypatch.setattr(launcher, "TrainLauncherApp", FakeLauncherApp)
    launcher.main()
    assert saves["n"] == 1


def test_launcher_main_handles_validation_error(monkeypatch) -> None:
    class FakeLauncherApp:
        def __init__(self, **_kwargs) -> None:  # noqa: ANN003
            return

        def run(self):
            return SimpleNamespace(
                action="start",
                environments=[],
                last={"environment_name": "", "train_config_path": "", "train_script": ""},
                other_options={"hf_token": "", "civitai_api_key": ""},
                selection=None,
            )

    monkeypatch.setattr(launcher, "colorama_init", lambda: None)
    monkeypatch.setattr(launcher, "scan_train_scripts", lambda: (["train_network.py"], None))
    monkeypatch.setattr(launcher, "load_config", lambda _p: AppConfig())
    monkeypatch.setattr(launcher, "save_config", lambda _p, _cfg: None)
    monkeypatch.setattr(launcher, "TrainLauncherApp", FakeLauncherApp)

    def _raise_validation(*_args):  # noqa: ANN002
        raise ValidationError.from_exception_data("LastSelection", [])

    monkeypatch.setattr(launcher, "start_training", _raise_validation)
    launcher.main()


def test_run_last_training_calls_start_training_with_saved_selection(monkeypatch, tmp_path) -> None:
    calls = {"start": 0}
    saved: list[AppConfig] = []
    config = AppConfig()
    config.last.environment_name = "local"
    config.last.train_config_path = str(tmp_path / "train.toml")
    config.last.train_script = "train_network.py"

    monkeypatch.setattr(launcher, "colorama_init", lambda: None)
    monkeypatch.setattr(launcher, "load_config", lambda _p: config)
    monkeypatch.setattr(launcher, "save_config", lambda _p, cfg: saved.append(cfg.model_copy(deep=True)))
    monkeypatch.setattr(launcher, "scan_train_scripts", lambda: (["train_network.py"], None))

    def _fake_start(cfg, result, script_options, require_confirmation=True):  # noqa: ANN001
        calls["start"] += 1
        assert cfg.last.environment_name == "local"
        assert result.selection.environment_name == "local"
        assert result.selection.train_config_path == str(tmp_path / "train.toml")
        assert result.selection.train_script == "train_network.py"
        assert script_options == ["train_network.py"]
        assert require_confirmation is False
        return True

    monkeypatch.setattr(launcher, "start_training", _fake_start)
    assert launcher.run_last_training() is True
    assert calls["start"] == 1
    assert len(saved) == 2


def test_run_last_training_fills_default_script_when_missing(monkeypatch, tmp_path) -> None:
    config = AppConfig()
    config.last.environment_name = "local"
    config.last.train_config_path = str(tmp_path / "train.toml")
    saved: list[AppConfig] = []

    monkeypatch.setattr(launcher, "colorama_init", lambda: None)
    monkeypatch.setattr(launcher, "load_config", lambda _p: config)
    monkeypatch.setattr(launcher, "save_config", lambda _p, cfg: saved.append(cfg.model_copy(deep=True)))
    monkeypatch.setattr(launcher, "scan_train_scripts", lambda: (["foo.py", "train_network.py"], None))

    def _fake_start(cfg, result, _script_options, require_confirmation=True):  # noqa: ANN001
        assert cfg.last.train_script == "train_network.py"
        assert result.selection.train_script == "train_network.py"
        assert require_confirmation is False
        return True

    monkeypatch.setattr(launcher, "start_training", _fake_start)
    assert launcher.run_last_training() is True
    assert saved[0].last.train_script == "train_network.py"
