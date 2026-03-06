from pathlib import Path
from typing import TextIO

from sd_train.infra.environment.base import Environment
from sd_train.infra.environment.types import FileServerSession, RunResult


class FakeResponse:
    def __init__(
        self,
        status_code: int,
        text: str = "",
        payload: dict | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._payload

    def close(self) -> None:
        return


class FakeEnvironment(Environment):
    def __init__(self) -> None:
        self.sync_to_local_calls: list[tuple[str, str, tuple[str, ...] | None]] = []
        self.sync_from_local_calls: list[tuple[str, str]] = []
        self.commands: list[str] = []
        self.responses: dict[str, RunResult] = {}

    async def connect(self) -> None:
        return

    async def close(self) -> None:
        return

    async def run(
        self,
        command: str,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> RunResult:
        _ = stdout, stderr
        self.commands.append(command)
        for key, response in self.responses.items():
            if key in command:
                return response
        return RunResult(stdout="", stderr="", code=0)

    async def health_check(self) -> bool:
        return True

    async def __aenter__(self) -> "FakeEnvironment":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        _ = exc_type, exc, tb
        return

    async def start_file_server(self) -> FileServerSession | None:
        return None

    async def stop_file_server(self) -> None:
        return

    async def file_server_status(self) -> FileServerSession | None:
        return None

    def sync_to_local(self, src: str, dst: str, include_paths: list[str] | None = None) -> None:
        normalized = tuple(include_paths) if include_paths is not None else None
        self.sync_to_local_calls.append((src, dst, normalized))

    def sync_from_local(self, src: str, dst: str) -> None:
        self.sync_from_local_calls.append((src, dst))


def create_tagger_dataset_single(root: Path) -> Path:
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    image = dataset / "img.png"
    image.write_bytes(b"img")
    (dataset / "img.txt").write_text("dog, cat", encoding="utf-8")
    return dataset


def create_tagger_dataset_pair(root: Path) -> Path:
    dataset = root / "dataset"
    dataset.mkdir(parents=True)
    (dataset / "a.png").write_bytes(b"img")
    (dataset / "b.jpg").write_bytes(b"img")
    (dataset / "ignore.txt").write_text("x", encoding="utf-8")
    (dataset / "a.txt").write_text("cat, dog", encoding="utf-8")
    (dataset / "b.txt").write_text("cat, bird", encoding="utf-8")
    return dataset
