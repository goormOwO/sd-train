import asyncio
from pathlib import Path

from sd_train.infra.environment.local_env import LocalEnvironment


def test_local_environment_run_executes_shell_command() -> None:
    env = LocalEnvironment()
    result = asyncio.run(env.run("printf 'hello'"))
    assert result.code == 0
    assert result.stdout == "hello"


def test_local_environment_start_file_server_returns_dummy_session() -> None:
    env = LocalEnvironment()
    session = asyncio.run(env.start_file_server())
    assert session is not None
    assert session.protocol == "local"
    assert session.root_path.endswith(".sd-train")


def test_local_environment_sync_from_local_copies_file_and_directory(tmp_path: Path) -> None:
    env = LocalEnvironment()
    home = Path.home()

    source_file = tmp_path / "weights.safetensors"
    source_file.write_text("x", encoding="utf-8")
    env.sync_from_local(str(source_file), "~/tmp/sd-train-local-test/file.bin")
    assert (home / "tmp" / "sd-train-local-test" / "file.bin").read_text(encoding="utf-8") == "x"

    source_dir = tmp_path / "dataset"
    source_dir.mkdir()
    (source_dir / "a.txt").write_text("a", encoding="utf-8")
    (source_dir / "nested").mkdir()
    (source_dir / "nested" / "b.txt").write_text("b", encoding="utf-8")
    env.sync_from_local(str(source_dir), "~/tmp/sd-train-local-test/data")
    assert (home / "tmp" / "sd-train-local-test" / "data" / "a.txt").read_text(encoding="utf-8") == "a"
    assert (home / "tmp" / "sd-train-local-test" / "data" / "nested" / "b.txt").read_text(encoding="utf-8") == "b"


def test_local_environment_sync_to_local_supports_include_paths(tmp_path: Path) -> None:
    env = LocalEnvironment()
    home = Path.home()
    source_root = home / "tmp" / "sd-train-local-test-sync"
    (source_root / "sub").mkdir(parents=True, exist_ok=True)
    (source_root / "a.txt").write_text("a", encoding="utf-8")
    (source_root / "sub" / "b.txt").write_text("b", encoding="utf-8")

    destination = tmp_path / "output"
    env.sync_to_local(str(source_root), str(destination), ["sub/b.txt"])

    assert not (destination / "a.txt").exists()
    assert (destination / "sub" / "b.txt").read_text(encoding="utf-8") == "b"
