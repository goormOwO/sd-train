import sys

from sd_train import cli


def test_cli_main_runs_launcher_by_default(monkeypatch) -> None:
    calls = {"launch": 0, "last": 0}

    monkeypatch.setattr(sys, "argv", ["sd-train"])
    monkeypatch.setattr(cli, "launch_main", lambda: calls.__setitem__("launch", calls["launch"] + 1))
    monkeypatch.setattr(cli, "run_last_training", lambda: calls.__setitem__("last", calls["last"] + 1))

    cli.main()
    assert calls == {"launch": 1, "last": 0}


def test_cli_main_runs_last_command(monkeypatch) -> None:
    calls = {"launch": 0, "last": 0}

    monkeypatch.setattr(sys, "argv", ["sd-train", "last"])
    monkeypatch.setattr(cli, "launch_main", lambda: calls.__setitem__("launch", calls["launch"] + 1))
    monkeypatch.setattr(cli, "run_last_training", lambda: calls.__setitem__("last", calls["last"] + 1))

    cli.main()
    assert calls == {"launch": 0, "last": 1}
