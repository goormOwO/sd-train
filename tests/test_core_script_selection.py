import json
from pathlib import Path

import pytest

from sd_train.core import script_selection
from tests.helpers import FakeResponse


def test_looks_like_main_entry() -> None:
    good = "def setup_parser():\n    pass\nif __name__ == '__main__':\n    pass\n"
    bad = "def main():\n    pass\n"
    assert script_selection._looks_like_main_entry(good) is True
    assert script_selection._looks_like_main_entry(bad) is False


def test_normalize_script_path() -> None:
    assert script_selection.normalize_script_path("train_network.py") == "train_network.py"
    with pytest.raises(ValueError):
        script_selection.normalize_script_path("../train.py")


def test_scan_train_scripts_uses_cache_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text(
        json.dumps({"cached_at": 1000.0, "scripts": ["train_network.py"], "error": None}),
        encoding="utf-8",
    )
    monkeypatch.setattr(script_selection, "_SCRIPT_SCAN_CACHE", None)
    monkeypatch.setattr(script_selection, "SCRIPT_SCAN_CACHE_FILE", cache)
    monkeypatch.setattr(script_selection.time, "time", lambda: 1005.0)
    scripts, error = script_selection.scan_train_scripts()
    assert scripts == ["train_network.py"]
    assert error is None


def test_scan_train_scripts_success_from_github(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(script_selection, "_SCRIPT_SCAN_CACHE", None)
    monkeypatch.setattr(script_selection, "SCRIPT_SCAN_CACHE_FILE", cache)
    monkeypatch.setattr(script_selection.time, "time", lambda: 2000.0)

    def _fake_get(url: str, timeout: int):  # noqa: ANN001
        if "git/trees/main" in url:
            return FakeResponse(
                200,
                payload={
                    "tree": [
                        {"path": "train_network.py", "type": "blob"},
                        {"path": "not_train.py", "type": "blob"},
                    ]
                },
            )
        if "raw.githubusercontent.com" in url and "train_network.py" in url:
            return FakeResponse(
                200,
                text="def setup_parser():\n    pass\nif __name__ == '__main__':\n    pass\n",
            )
        return FakeResponse(404, text="")

    monkeypatch.setattr(script_selection.requests, "get", _fake_get)
    scripts, error = script_selection.scan_train_scripts()
    assert scripts == ["train_network.py"]
    assert error is None


def test_scan_train_scripts_fallback_when_github_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "cache.json"
    monkeypatch.setattr(script_selection, "_SCRIPT_SCAN_CACHE", None)
    monkeypatch.setattr(script_selection, "SCRIPT_SCAN_CACHE_FILE", cache)
    monkeypatch.setattr(script_selection.requests, "get", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    scripts, error = script_selection.scan_train_scripts()
    assert scripts == script_selection.SCRIPT_FALLBACKS
    assert error is not None
