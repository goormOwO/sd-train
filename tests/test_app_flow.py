from types import SimpleNamespace
from pathlib import Path

import pytest

from sd_train.app import preflight as app_preflight
from sd_train.app import start as app_start
from sd_train.config.models import AppConfig
from sd_train.domain.refs import DownloadAuth


def test_run_preflight_gate_returns_none_on_check_exception(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class FakeReview:
        def __init__(self, summary: str, error_message: str | None = None) -> None:
            self.summary = summary
            self.error_message = error_message

        def run(self) -> bool:
            return False

    def _boom(_path: Path, _script: str, _auth):  # noqa: ANN001
        raise RuntimeError("External reference timeout")

    monkeypatch.setattr(app_preflight, "PreflightReviewApp", FakeReview)
    monkeypatch.setattr(app_preflight, "run_preflight_checks", _boom)
    result = app_preflight.run_preflight_gate(tmp_path / "train.toml", "train_network.py", DownloadAuth())
    assert result is None


def test_run_preflight_gate_returns_result_when_user_proceeds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeReview:
        def __init__(self, summary: str, error_message: str | None = None) -> None:
            self.summary = summary
            self.error_message = error_message

        def run(self) -> bool:
            return True

    report = SimpleNamespace()
    checks: list = []
    monkeypatch.setattr(app_preflight, "PreflightReviewApp", FakeReview)
    monkeypatch.setattr(app_preflight, "run_preflight_checks", lambda *_args: (report, checks))
    monkeypatch.setattr(app_preflight, "build_external_ref_failure_message", lambda _checks: "")
    monkeypatch.setattr(app_preflight, "build_preflight_summary", lambda *_args: "ok")
    result = app_preflight.run_preflight_gate(tmp_path / "train.toml", "train_network.py", DownloadAuth())
    assert result is not None
    assert result.report is report


def test_run_preflight_or_raise_returns_result_without_ui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = SimpleNamespace()
    checks: list = []
    monkeypatch.setattr(app_preflight, "run_preflight_checks", lambda *_args: (report, checks))
    monkeypatch.setattr(app_preflight, "build_external_ref_failure_message", lambda _checks: "")
    result = app_preflight.run_preflight_or_raise(tmp_path / "train.toml", "train_network.py", DownloadAuth())
    assert result.report is report
    assert result.checks is checks


def test_run_preflight_or_raise_raises_on_external_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    report = SimpleNamespace()
    checks = [SimpleNamespace(ok=False, provider="hf", key="x", ref="y", detail="denied")]
    monkeypatch.setattr(app_preflight, "run_preflight_checks", lambda *_args: (report, checks))
    monkeypatch.setattr(
        app_preflight,
        "build_external_ref_failure_message",
        lambda _checks: "External reference check failed:\n- [hf] x: y\n  denied",
    )

    with pytest.raises(RuntimeError, match="External reference check failed"):
        app_preflight.run_preflight_or_raise(tmp_path / "train.toml", "train_network.py", DownloadAuth())


def test_start_training_returns_false_without_selection() -> None:
    config = AppConfig()
    result = SimpleNamespace(selection=None)
    assert app_start.start_training(config, result, ["train_network.py"]) is False


def test_start_training_rejects_unknown_script(tmp_path: Path) -> None:
    cfg = AppConfig()
    selection = SimpleNamespace(
        train_script="unknown.py",
        train_config_path=str(tmp_path / "train.toml"),
        environment_name="dev",
    )
    result = SimpleNamespace(selection=selection)
    with pytest.raises(ValueError, match="not in GitHub script list"):
        app_start.start_training(cfg, result, ["train_network.py"])


def test_start_training_returns_false_when_preflight_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train_cfg = tmp_path / "train.toml"
    train_cfg.write_text("x=1", encoding="utf-8")
    cfg = AppConfig(
        other_options={"hf_token": "", "civitai_api_key": ""},
        environments=[{"name": "dev", "type": "ssh", "host": "127.0.0.1", "user": "me", "port": 22}],
    )
    selection = SimpleNamespace(
        train_script="train_network.py",
        train_config_path=str(train_cfg),
        environment_name="dev",
    )
    result = SimpleNamespace(selection=selection)
    monkeypatch.setattr(app_start, "run_preflight_gate", lambda *_args: None)
    assert app_start.start_training(cfg, result, ["train_network.py"]) is False


def test_start_training_raises_when_config_missing(tmp_path: Path) -> None:
    cfg = AppConfig()
    selection = SimpleNamespace(
        train_script="train_network.py",
        train_config_path=str(tmp_path / "missing.toml"),
        environment_name="dev",
    )
    result = SimpleNamespace(selection=selection)
    with pytest.raises(ValueError, match="Train config file not found"):
        app_start.start_training(cfg, result, ["train_network.py"])


def test_start_training_raises_when_environment_is_missing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train_cfg = tmp_path / "train.toml"
    train_cfg.write_text("x=1", encoding="utf-8")
    cfg = AppConfig(
        other_options={"hf_token": "", "civitai_api_key": ""},
        environments=[{"name": "dev", "type": "ssh", "host": "127.0.0.1", "user": "me", "port": 22}],
    )
    selection = SimpleNamespace(
        train_script="train_network.py",
        train_config_path=str(train_cfg),
        environment_name="missing-env",
    )
    result = SimpleNamespace(selection=selection)
    monkeypatch.setattr(app_start, "run_preflight_gate", lambda *_args: object())

    with pytest.raises(ValueError, match="Selected environment not found: missing-env"):
        app_start.start_training(cfg, result, ["train_network.py"])


def test_start_training_success_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train_cfg = tmp_path / "train.toml"
    train_cfg.write_text("x=1", encoding="utf-8")
    cfg = AppConfig(
        other_options={"hf_token": "hf", "civitai_api_key": "ck"},
        environments=[{"name": "dev", "type": "ssh", "host": "127.0.0.1", "user": "me", "port": 22}],
    )
    selection = SimpleNamespace(
        train_script="train_network.py",
        train_config_path=str(train_cfg),
        environment_name="dev",
    )
    result = SimpleNamespace(selection=selection)

    monkeypatch.setattr(app_start, "run_preflight_gate", lambda *_args: object())
    monkeypatch.setattr(app_start, "build_environment", lambda _cfg: object())
    monkeypatch.setattr(app_start, "run_training_session", lambda *_args: None)
    called = {"n": 0}

    def _fake_asyncio_run(_coroutine):  # noqa: ANN001
        called["n"] += 1
        return None

    monkeypatch.setattr(app_start.asyncio, "run", _fake_asyncio_run)
    assert app_start.start_training(cfg, result, ["train_network.py"]) is True
    assert called["n"] == 1


def test_start_training_success_path_for_builtin_local(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    train_cfg = tmp_path / "train.toml"
    train_cfg.write_text("x=1", encoding="utf-8")
    cfg = AppConfig(other_options={"hf_token": "", "civitai_api_key": ""})
    selection = SimpleNamespace(
        train_script="train_network.py",
        train_config_path=str(train_cfg),
        environment_name="local",
    )
    result = SimpleNamespace(selection=selection)

    monkeypatch.setattr(app_start, "run_preflight_gate", lambda *_args: object())
    monkeypatch.setattr(app_start, "build_environment", lambda _cfg: object())
    monkeypatch.setattr(app_start, "run_training_session", lambda *_args: None)
    called = {"n": 0}

    def _fake_asyncio_run(_coroutine):  # noqa: ANN001
        called["n"] += 1
        return None

    monkeypatch.setattr(app_start.asyncio, "run", _fake_asyncio_run)
    assert app_start.start_training(cfg, result, ["train_network.py"]) is True
    assert called["n"] == 1


def test_start_training_non_interactive_bypasses_confirmation_ui(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    train_cfg = tmp_path / "train.toml"
    train_cfg.write_text("x=1", encoding="utf-8")
    cfg = AppConfig(other_options={"hf_token": "", "civitai_api_key": ""})
    selection = SimpleNamespace(
        train_script="train_network.py",
        train_config_path=str(train_cfg),
        environment_name="local",
    )
    result = SimpleNamespace(selection=selection)
    calls = {"gate": 0, "raise": 0}

    monkeypatch.setattr(app_start, "run_preflight_gate", lambda *_args: calls.__setitem__("gate", calls["gate"] + 1))
    monkeypatch.setattr(
        app_start,
        "run_preflight_or_raise",
        lambda *_args: calls.__setitem__("raise", calls["raise"] + 1) or object(),
    )
    monkeypatch.setattr(app_start, "build_environment", lambda _cfg: object())
    monkeypatch.setattr(app_start, "run_training_session", lambda *_args: None)
    monkeypatch.setattr(app_start.asyncio, "run", lambda _coroutine: None)

    assert app_start.start_training(cfg, result, ["train_network.py"], require_confirmation=False) is True
    assert calls == {"gate": 0, "raise": 1}
