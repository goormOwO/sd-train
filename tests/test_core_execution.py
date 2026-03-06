import asyncio
from pathlib import Path
from typing import TextIO
from types import SimpleNamespace

import pytest
import toml

from sd_train.core import execution
from sd_train.core.execution import MaterializedRef
from sd_train.domain.refs import DownloadAuth
from sd_train.infra.environment.types import FileServerSession
from sd_train.infra.environment.types import RunResult
from tests.helpers import FakeEnvironment


class _FakeEnvironment(FakeEnvironment):
    pass


def test_normalize_script_path() -> None:
    assert execution._normalize_script_path("./train_network.py") == "train_network.py"
    with pytest.raises(ValueError):
        execution._normalize_script_path("../bad.py")


def test_mask_secret_and_kind_helpers() -> None:
    assert execution._mask_secret("abcd") == "****"
    assert execution._mask_secret("abcdef") == "ab**ef"
    assert execution._infer_key_kind("sample_prompts") == "prompts"
    assert execution._infer_key_kind("train_data_dir") == "datasets"
    assert execution._infer_key_kind("foo") == "misc"


def test_github_raw_url_and_pins() -> None:
    assert execution._github_raw_url("owner/repo", "abc123", "file.txt") == (
        "https://raw.githubusercontent.com/owner/repo/abc123/file.txt"
    )
    assert len(execution.PINNED_SD_SCRIPTS_COMMIT) == 40
    assert len(execution.PINNED_KOHYA_SS_COMMIT) == 40
    assert len(execution.PINNED_CIVITAI_DOWNLOADER_COMMIT) == 40
    assert execution.REMOTE_TRAIN_PYTHON_VERSION == "3.11"
    assert execution.UV_INSTALL_URL == "https://astral.sh/uv/install.sh"


def test_hash_local_path_for_file_and_dir(tmp_path: Path) -> None:
    file_path = tmp_path / "a.txt"
    file_path.write_text("x", encoding="utf-8")
    assert len(execution._hash_local_path(file_path)) == 64
    d = tmp_path / "d"
    d.mkdir()
    (d / "a.bin").write_bytes(b"1")
    assert len(execution._hash_local_path(d)) == 64


def test_hash_local_path_invalid_type_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        execution._hash_local_path(tmp_path / "missing")


def test_run_commands_raises_on_failure() -> None:
    env = _FakeEnvironment()
    env.responses["bad"] = RunResult(stdout="", stderr="x", code=2)
    with pytest.raises(RuntimeError):
        asyncio.run(execution._run_commands(env, ["echo ok", "bad"]))


def test_ensure_remote_exists() -> None:
    env = _FakeEnvironment()
    env.responses["[ -e"] = RunResult(stdout="", stderr="", code=0)
    assert asyncio.run(execution._ensure_remote_exists(env, "/x")) is True
    env.responses["[ -e"] = RunResult(stdout="", stderr="", code=1)
    assert asyncio.run(execution._ensure_remote_exists(env, "/x")) is False


def test_list_stable_remote_files_filters_temp_suffixes() -> None:
    env = _FakeEnvironment()
    env.responses["python3 - <<'PY'"] = RunResult(
        stdout="\n".join(
            [
                "/out/a.safetensors\t10\t100",
                "/out/b.tmp\t10\t100",
                "malformed",
            ]
        ),
        stderr="",
        code=0,
    )
    files = asyncio.run(execution._list_stable_remote_files(env, "/out", 10))
    assert files == [("/out/a.safetensors", 10.0, 100)]


def test_list_stable_remote_files_returns_empty_on_error() -> None:
    env = _FakeEnvironment()
    env.responses["python3 - <<'PY'"] = RunResult(stdout="", stderr="err", code=1)
    files = asyncio.run(execution._list_stable_remote_files(env, "/out", 10))
    assert files == []


def test_final_output_sync_skips_when_remote_dir_missing(tmp_path: Path) -> None:
    env = _FakeEnvironment()
    env.responses["[ -d"] = RunResult(stdout="", stderr="", code=1)
    asyncio.run(execution.final_output_sync(env, "~/.sd-train/runs/x/output", "/remote/output", tmp_path / "outputs"))
    assert env.sync_to_local_calls == []


def test_final_output_sync_runs_when_remote_dir_exists(tmp_path: Path) -> None:
    env = _FakeEnvironment()
    env.responses["[ -d"] = RunResult(stdout="", stderr="", code=0)
    local_dir = tmp_path / "outputs"
    asyncio.run(execution.final_output_sync(env, "~/.sd-train/runs/x/output", "/remote/output", local_dir))
    assert local_dir.exists()
    assert env.sync_to_local_calls == [("~/.sd-train/runs/x/output", str(local_dir), None)]


def test_periodic_output_sync_copies_new_stable_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _FakeEnvironment()
    stop = asyncio.Event()
    remote_output_abs = "/remote/output"
    local_output_dir = tmp_path / "outputs"
    called = {"n": 0}

    async def _fake_list(_environment, _remote_output_abs, _stability):  # noqa: ANN001
        called["n"] += 1
        stop.set()
        return [("/remote/output/a.safetensors", 10.0, 100)]

    monkeypatch.setattr(execution, "_list_stable_remote_files", _fake_list)
    asyncio.run(
        execution.periodic_output_sync(
            env,
            remote_output_abs=remote_output_abs,
            local_output_dir=local_output_dir,
            remote_home="/home/ubuntu",
            stop_event=stop,
            interval_seconds=1,
            stability_seconds=1,
        )
    )
    assert called["n"] == 1
    assert len(env.sync_to_local_calls) == 1
    src, dst, include_paths = env.sync_to_local_calls[0]
    assert src == "/remote/output"
    assert dst.endswith("outputs")
    assert include_paths == ("a.safetensors",)


def test_periodic_output_sync_retries_after_sync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _FakeEnvironment()
    stop = asyncio.Event()
    remote_output_abs = "/remote/output"
    local_output_dir = tmp_path / "outputs"
    calls = {"n": 0}
    sync_attempts = {"n": 0}

    async def _fake_list(_environment, _remote_output_abs, _stability):  # noqa: ANN001
        calls["n"] += 1
        if calls["n"] >= 2:
            stop.set()
        return [("/remote/output/b.safetensors", 20.0, 200)]

    original_sync = env.sync_to_local

    def _flaky_sync(src: str, dst: str, include_paths: list[str] | None = None) -> None:
        sync_attempts["n"] += 1
        if sync_attempts["n"] == 1:
            raise RuntimeError("transient failure")
        original_sync(src, dst, include_paths)

    monkeypatch.setattr(execution, "_list_stable_remote_files", _fake_list)
    monkeypatch.setattr(env, "sync_to_local", _flaky_sync)
    asyncio.run(
        execution.periodic_output_sync(
            env,
            remote_output_abs=remote_output_abs,
            local_output_dir=local_output_dir,
            remote_home="/home/ubuntu",
            stop_event=stop,
            interval_seconds=0,
            stability_seconds=1,
        )
    )
    assert calls["n"] == 2
    assert sync_attempts["n"] == 2
    assert env.sync_to_local_calls == [
        ("/remote/output", str(local_output_dir), ("b.safetensors",)),
    ]


def test_upload_local_ref_uploads_when_missing(tmp_path: Path) -> None:
    env = _FakeEnvironment()
    env.responses["[ -e"] = RunResult(stdout="", stderr="", code=1)
    local = tmp_path / "x.bin"
    local.write_bytes(b"x")
    materialized = asyncio.run(
        execution._upload_local_ref(env, "network_weights", local, "~/.sd-train", "/home/ubuntu")
    )
    assert materialized.abs_path.startswith("/home/ubuntu/.sd-train/artifacts")
    assert len(env.sync_from_local_calls) == 1


def test_upload_local_ref_skips_when_exists(tmp_path: Path) -> None:
    env = _FakeEnvironment()
    env.responses["[ -e"] = RunResult(stdout="", stderr="", code=0)
    local = tmp_path / "x.bin"
    local.write_bytes(b"x")
    asyncio.run(execution._upload_local_ref(env, "network_weights", local, "~/.sd-train", "/home/ubuntu"))
    assert env.sync_from_local_calls == []


def test_materialize_hf_ref_requires_subpath_for_file_key() -> None:
    env = _FakeEnvironment()
    with pytest.raises(ValueError):
        asyncio.run(
            execution._materialize_hf_ref(
                env,
                "network_weights",
                "model:org/repo",
                "~/.sd-train",
                "/home/ubuntu",
                DownloadAuth(),
            )
        )


def test_materialize_hf_ref_returns_existing_path() -> None:
    env = _FakeEnvironment()
    env.responses["[ -e"] = RunResult(stdout="", stderr="", code=0)
    mat = asyncio.run(
        execution._materialize_hf_ref(
            env,
            "pretrained_model_name_or_path",
            "model:org/repo::model.safetensors",
            "~/.sd-train",
            "/home/ubuntu",
            DownloadAuth(),
        )
    )
    assert mat.abs_path.endswith("model.safetensors")


def test_ensure_remote_sd_scripts_uses_pinned_commit() -> None:
    env = _FakeEnvironment()
    asyncio.run(execution._ensure_remote_sd_scripts(env, "/home/ubuntu"))
    command = env.commands[-1]
    assert execution.PINNED_SD_SCRIPTS_COMMIT in command
    assert "git pull --ff-only" not in command


def test_prepare_remote_python_venv_uses_pinned_requirement_urls() -> None:
    env = _FakeEnvironment()
    venv_abs, _sd_scripts_abs = asyncio.run(execution._prepare_remote_python_venv(env, "/home/ubuntu"))
    command = env.commands[-1]
    assert venv_abs == "/home/ubuntu/.sd-train/.venv"
    assert execution.UV_INSTALL_URL in command
    assert 'export PATH="$HOME/.local/bin:$PATH";' in command
    assert "if [ ! -x /home/ubuntu/.local/bin/uv ] && ! command -v uv >/dev/null 2>&1; then " in command
    assert "base_req_source = os.path.join(sd_scripts_abs, 'requirements.txt')" in command
    assert "linux_req_source = os.path.join(sd_scripts_abs, 'requirements_linux.txt')" in command
    assert "source_req = linux_req_source if os.path.exists(linux_req_source) else base_req_source" in command
    assert "source_label = 'requirements_linux.txt' if source_req == linux_req_source else 'requirements.txt'" in command
    assert "Installing remote sd-scripts requirements from {source_label}: {source_req}" in command
    assert "stripped in ('-e .', '--editable .', '.', '-e ./', '--editable ./', './')" in command
    assert "base_norm.append(f'-e {sd_scripts_abs}')" in command
    assert f"'python', 'install', '{execution.REMOTE_TRAIN_PYTHON_VERSION}'" in command
    assert f"'venv', '--python', '{execution.REMOTE_TRAIN_PYTHON_VERSION}'" in command
    assert "'venv', '--python', '3.11', venv_abs" in command
    assert "if os.path.exists(venv_python):" in command
    assert "Using existing virtual environment at:" in command
    assert "elif os.path.exists(venv_abs):" in command
    assert "Existing virtual environment is incomplete" in command
    assert "'pip', 'install', '--python', venv_python, '--upgrade', '-r', linux_req_path" in command
    assert "refs/heads/master" not in command


def test_materialize_hf_ref_installs_pinned_huggingface_hub_version() -> None:
    env = _FakeEnvironment()
    env.responses["[ -e"] = RunResult(stdout="", stderr="", code=1)
    asyncio.run(
        execution._materialize_hf_ref(
            env,
            "pretrained_model_name_or_path",
            "model:org/repo::model.safetensors",
            "~/.sd-train",
            "/home/ubuntu",
            DownloadAuth(),
        )
    )
    command = env.commands[-1]
    assert f"huggingface_hub=={execution.PINNED_HUGGINGFACE_HUB_VERSION}" in command
    assert "venv_python" in command
    assert 'HF_PYTHON=$(python3 - <<\'PY\'' in command
    assert '"$HF_PYTHON" - <<\'PY\'' in command
    assert "install_python = venv_python if venv_python and os.path.exists(venv_python) else sys.executable" in command
    assert "install_python," in command


def test_ensure_optional_network_modules_uses_sd_scripts_for_local_modules() -> None:
    env = _FakeEnvironment()
    asyncio.run(
        execution._ensure_optional_network_modules(
            env,
            "/home/ubuntu/.sd-train/runs/run-1/train.toml",
            "/home/ubuntu/.sd-train/.venv",
            "/home/ubuntu/.sd-train/sd-scripts",
            "/home/ubuntu/.local/bin/uv",
        )
    )
    command = env.commands[-1]
    assert "sd_scripts_abs = cfg['sd_scripts_abs']" in command
    assert "sys.path.insert(0, sd_scripts_abs)" in command
    assert "os.chdir(sd_scripts_abs)" in command
    assert "env['PYTHONPATH'] = sd_scripts_abs" in command
    assert "cwd=sd_scripts_abs" in command
    assert "local_module_roots = {'networks'}" in command
    assert "if root in local_module_roots:" in command
    assert "should come from sd-scripts" in command
    assert "not present in pinned sd-scripts checkout" in command


def test_ensure_optional_network_modules_surfaces_remote_error_detail() -> None:
    env = _FakeEnvironment()
    env.responses["SD_TRAIN_OPTIONAL="] = RunResult(stdout="", stderr="remote detail", code=1)
    with pytest.raises(RuntimeError, match="remote detail"):
        asyncio.run(
            execution._ensure_optional_network_modules(
                env,
                "/home/ubuntu/.sd-train/runs/run-1/train.toml",
                "/home/ubuntu/.sd-train/.venv",
                "/home/ubuntu/.sd-train/sd-scripts",
                "/home/ubuntu/.local/bin/uv",
            )
        )


def test_materialize_civitai_ref_rejects_dir_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _FakeEnvironment()
    import sd_train.domain.refs as refs

    monkeypatch.setattr(refs, "_resolve_civitai_download_id", lambda *_args: ("123", "a.safetensors", "ok"))
    with pytest.raises(ValueError):
        asyncio.run(
            execution._materialize_civitai_ref(
                env,
                "train_data_dir",
                "civitai:123",
                "~/.sd-train",
                "/home/ubuntu",
                DownloadAuth(civitai_api_key="k"),
            )
        )


def test_materialize_civitai_ref_downloads_pinned_script(monkeypatch: pytest.MonkeyPatch) -> None:
    env = _FakeEnvironment()
    env.responses["[ -e"] = RunResult(stdout="", stderr="", code=1)
    import sd_train.domain.refs as refs

    monkeypatch.setattr(refs, "_resolve_civitai_download_id", lambda *_args: ("123", "a.safetensors", "ok"))
    asyncio.run(
        execution._materialize_civitai_ref(
            env,
            "network_weights",
            "civitai:123",
            "~/.sd-train",
            "/home/ubuntu",
            DownloadAuth(civitai_api_key="k"),
        )
    )
    command = env.commands[-1]
    assert execution.PINNED_CIVITAI_DOWNLOADER_COMMIT in command
    assert "/main/download.py" not in command


def test_resolve_and_patch_train_toml_rejects_missing_non_ref_path(tmp_path: Path) -> None:
    env = _FakeEnvironment()
    train_file = tmp_path / "train.toml"
    train_file.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::x.safetensors",
                "dataset_config": "missing.toml",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        asyncio.run(
            execution._resolve_and_patch_train_toml(
                env,
                train_file,
                "~/.sd-train/runs/x",
                "/home/ubuntu",
                DownloadAuth(),
            )
        )


def test_resolve_and_patch_train_toml_sets_default_output_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _FakeEnvironment()
    subset_dir = tmp_path / "data" / "subset-a"
    subset_dir.mkdir(parents=True)
    train_file = tmp_path / "train.toml"
    train_file.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::x.safetensors",
                "train_data_dir": str(tmp_path / "data"),
            }
        ),
        encoding="utf-8",
    )

    async def _fake_upload(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return MaterializedRef(sync_path="~/.sd-train/artifacts/datasets/h/x", abs_path="/home/ubuntu/.sd-train/artifacts/datasets/h/x")

    monkeypatch.setattr(execution, "_upload_local_ref", _fake_upload)
    remote_train_sync, remote_train_abs, remote_output_sync = asyncio.run(
        execution._resolve_and_patch_train_toml(
            env,
            train_file,
            "~/.sd-train/runs/run-1",
            "/home/ubuntu",
            DownloadAuth(),
        )
    )

    assert remote_train_sync.endswith("/train.toml")
    assert remote_train_abs.endswith("/train.toml")
    assert remote_output_sync == "~/.sd-train/runs/run-1/output"
    assert len(env.sync_from_local_calls) == 1


def test_resolve_and_patch_train_toml_materializes_qwen3_hf_ref(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _FakeEnvironment()
    captured = {"text": ""}
    subset_dir = tmp_path / "data" / "subset-a"
    subset_dir.mkdir(parents=True)
    train_file = tmp_path / "train.toml"
    train_file.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::x.safetensors",
                "train_data_dir": str(tmp_path / "data"),
                "qwen3": "model:org/repo::split_files/text_encoders/qwen.safetensors",
            }
        ),
        encoding="utf-8",
    )

    async def _fake_upload(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return MaterializedRef(
            sync_path="~/.sd-train/artifacts/datasets/h/x",
            abs_path="/home/ubuntu/.sd-train/artifacts/datasets/h/x",
        )

    async def _fake_hf(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return MaterializedRef(
            sync_path="~/.sd-train/artifacts/models/h/qwen.safetensors",
            abs_path="/home/ubuntu/.sd-train/artifacts/models/h/qwen.safetensors",
        )

    def _capture_sync(src: str, dst: str) -> None:
        captured["text"] = Path(src).read_text(encoding="utf-8")
        env.sync_from_local_calls.append((src, dst))

    monkeypatch.setattr(execution, "_upload_local_ref", _fake_upload)
    monkeypatch.setattr(execution, "_materialize_hf_ref", _fake_hf)
    monkeypatch.setattr(env, "sync_from_local", _capture_sync)
    asyncio.run(
        execution._resolve_and_patch_train_toml(
            env,
            train_file,
            "~/.sd-train/runs/run-1",
            "/home/ubuntu",
            DownloadAuth(),
        )
    )

    assert len(env.sync_from_local_calls) == 1
    rendered = toml.loads(captured["text"])
    assert rendered["qwen3"] == "/home/ubuntu/.sd-train/artifacts/models/h/qwen.safetensors"


def test_resolve_and_patch_train_toml_materializes_explicit_hf_ref_for_unknown_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env = _FakeEnvironment()
    captured = {"text": ""}
    subset_dir = tmp_path / "data" / "subset-a"
    subset_dir.mkdir(parents=True)
    train_file = tmp_path / "train.toml"
    train_file.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::x.safetensors",
                "train_data_dir": str(tmp_path / "data"),
                "custom_text_encoder": "model:org/repo::encoder.safetensors",
            }
        ),
        encoding="utf-8",
    )

    async def _fake_upload(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return MaterializedRef(
            sync_path="~/.sd-train/artifacts/datasets/h/x",
            abs_path="/home/ubuntu/.sd-train/artifacts/datasets/h/x",
        )

    async def _fake_hf(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return MaterializedRef(
            sync_path="~/.sd-train/artifacts/models/h/encoder.safetensors",
            abs_path="/home/ubuntu/.sd-train/artifacts/models/h/encoder.safetensors",
        )

    def _capture_sync(src: str, dst: str) -> None:
        captured["text"] = Path(src).read_text(encoding="utf-8")
        env.sync_from_local_calls.append((src, dst))

    monkeypatch.setattr(execution, "_upload_local_ref", _fake_upload)
    monkeypatch.setattr(execution, "_materialize_hf_ref", _fake_hf)
    monkeypatch.setattr(env, "sync_from_local", _capture_sync)
    asyncio.run(
        execution._resolve_and_patch_train_toml(
            env,
            train_file,
            "~/.sd-train/runs/run-1",
            "/home/ubuntu",
            DownloadAuth(),
        )
    )

    rendered = toml.loads(captured["text"])
    assert rendered["custom_text_encoder"] == "/home/ubuntu/.sd-train/artifacts/models/h/encoder.safetensors"


def test_run_training_session_requires_file_server() -> None:
    env = _FakeEnvironment()
    selection = type("S", (), {"train_config_path": "train.toml", "train_script": "train_network.py"})()
    with pytest.raises(RuntimeError, match="file server unavailable"):
        asyncio.run(execution.run_training_session(env, selection, DownloadAuth()))


class _SessionEnvironment(_FakeEnvironment):
    def __init__(self) -> None:
        super().__init__()
        self.transfer_progress_calls: list[bool] = []

    async def start_file_server(self) -> FileServerSession | None:
        return FileServerSession(
            protocol="sftp",
            host="127.0.0.1",
            port=22,
            username="u",
            password="p",
            root_path="/",
            pid=1,
            url="sftp://127.0.0.1:22",
        )

    async def run(
        self,
        command: str,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> RunResult:
        _ = stdout, stderr
        self.commands.append(command)
        if 'printf "%s" "$HOME"' in command:
            return RunResult(stdout="/home/ubuntu", stderr="", code=0)
        if "[ -f " in command:
            return RunResult(stdout="", stderr="", code=0)
        if "accelerate" in command and "launch" in command:
            return RunResult(stdout="", stderr="boom", code=1)
        return RunResult(stdout="", stderr="", code=0)

    def set_transfer_progress(self, enabled: bool) -> None:
        self.transfer_progress_calls.append(enabled)


def test_run_training_session_still_runs_final_sync_on_training_failure(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _SessionEnvironment()
    train_cfg = tmp_path / "train.toml"
    train_cfg.write_text("x=1", encoding="utf-8")
    selection = SimpleNamespace(train_config_path=str(train_cfg), train_script="train_network.py")
    called = {"final": 0}

    async def _noop_commands(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return None

    async def _noop_sd_scripts(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return None

    async def _prepare_venv(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return "/home/ubuntu/.sd-train/venv", "/home/ubuntu/.sd-train/sd-scripts"

    async def _resolve_train(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return "~/.sd-train/runs/run-1/train.toml", "/home/ubuntu/.sd-train/runs/run-1/train.toml", "~/.sd-train/runs/run-1/output"

    async def _periodic(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return None

    async def _final(*_args, **_kwargs):  # noqa: ANN002, ANN003
        called["final"] += 1

    monkeypatch.setattr(execution, "_run_commands", _noop_commands)
    monkeypatch.setattr(execution, "_ensure_remote_sd_scripts", _noop_sd_scripts)
    monkeypatch.setattr(execution, "_prepare_remote_python_venv", _prepare_venv)
    monkeypatch.setattr(execution, "_resolve_and_patch_train_toml", _resolve_train)
    monkeypatch.setattr(execution, "periodic_output_sync", _periodic)
    monkeypatch.setattr(execution, "final_output_sync", _final)

    with pytest.raises(RuntimeError, match="Training command failed"):
        asyncio.run(execution.run_training_session(env, selection, DownloadAuth()))
    assert called["final"] == 1
    assert env.transfer_progress_calls == [True, False]


def test_run_training_session_keeps_progress_enabled_during_final_sync(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env = _SessionEnvironment()
    train_cfg = tmp_path / "train.toml"
    train_cfg.write_text("x=1", encoding="utf-8")
    selection = SimpleNamespace(train_config_path=str(train_cfg), train_script="train_network.py")
    observed: list[bool] = []

    async def _noop_commands(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return None

    async def _noop_sd_scripts(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return None

    async def _prepare_venv(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return "/home/ubuntu/.sd-train/venv", "/home/ubuntu/.sd-train/sd-scripts"

    async def _resolve_train(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return "~/.sd-train/runs/run-1/train.toml", "/home/ubuntu/.sd-train/runs/run-1/train.toml", "~/.sd-train/runs/run-1/output"

    async def _periodic(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return None

    async def _final(*_args, **_kwargs):  # noqa: ANN002, ANN003
        observed.append(env.transfer_progress_calls[-1])

    async def _run_ok(command: str, stdout: TextIO | None = None, stderr: TextIO | None = None) -> RunResult:
        _ = stdout, stderr
        if 'printf "%s" "$HOME"' in command:
            return RunResult(stdout="/home/ubuntu", stderr="", code=0)
        if "[ -f " in command:
            return RunResult(stdout="", stderr="", code=0)
        if "accelerate" in command and "launch" in command:
            return RunResult(stdout="", stderr="", code=0)
        return RunResult(stdout="", stderr="", code=0)

    monkeypatch.setattr(env, "run", _run_ok)
    monkeypatch.setattr(execution, "_run_commands", _noop_commands)
    monkeypatch.setattr(execution, "_ensure_remote_sd_scripts", _noop_sd_scripts)
    monkeypatch.setattr(execution, "_prepare_remote_python_venv", _prepare_venv)
    monkeypatch.setattr(execution, "_resolve_and_patch_train_toml", _resolve_train)
    monkeypatch.setattr(execution, "periodic_output_sync", _periodic)
    monkeypatch.setattr(execution, "final_output_sync", _final)

    asyncio.run(execution.run_training_session(env, selection, DownloadAuth()))

    assert observed == [True]
    assert env.transfer_progress_calls == [True, False]


def test_run_training_session_swallows_final_sync_error(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _SessionEnvironment()
    train_cfg = tmp_path / "train.toml"
    train_cfg.write_text("x=1", encoding="utf-8")
    selection = SimpleNamespace(train_config_path=str(train_cfg), train_script="train_network.py")

    async def _noop_commands(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return None

    async def _noop_sd_scripts(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return None

    async def _prepare_venv(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return "/home/ubuntu/.sd-train/venv", "/home/ubuntu/.sd-train/sd-scripts"

    async def _resolve_train(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return "~/.sd-train/runs/run-1/train.toml", "/home/ubuntu/.sd-train/runs/run-1/train.toml", "~/.sd-train/runs/run-1/output"

    async def _periodic(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return None

    async def _final_raise(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("sync failed")

    async def _run_ok(command: str, stdout: TextIO | None = None, stderr: TextIO | None = None) -> RunResult:
        _ = stdout, stderr
        if 'printf "%s" "$HOME"' in command:
            return RunResult(stdout="/home/ubuntu", stderr="", code=0)
        if "[ -f " in command:
            return RunResult(stdout="", stderr="", code=0)
        if "accelerate" in command and "launch" in command:
            return RunResult(stdout="", stderr="", code=0)
        return RunResult(stdout="", stderr="", code=0)

    monkeypatch.setattr(env, "run", _run_ok)
    monkeypatch.setattr(execution, "_run_commands", _noop_commands)
    monkeypatch.setattr(execution, "_ensure_remote_sd_scripts", _noop_sd_scripts)
    monkeypatch.setattr(execution, "_prepare_remote_python_venv", _prepare_venv)
    monkeypatch.setattr(execution, "_resolve_and_patch_train_toml", _resolve_train)
    monkeypatch.setattr(execution, "periodic_output_sync", _periodic)
    monkeypatch.setattr(execution, "final_output_sync", _final_raise)

    asyncio.run(execution.run_training_session(env, selection, DownloadAuth()))
