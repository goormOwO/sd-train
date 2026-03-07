import asyncio
import os
import shutil
from pathlib import Path
from typing import TextIO

from sd_train.infra.environment.base import Environment
from sd_train.infra.environment.types import FileServerSession, RunResult


class LocalEnvironment(Environment):
    def __init__(self) -> None:
        self._root_path = str(Path("~/.sd-train").expanduser())
        self._file_server = FileServerSession(
            protocol="local",
            host="localhost",
            port=0,
            username="local",
            password="",
            root_path=self._root_path,
            pid=None,
            url=f"file://{self._root_path}",
        )

    async def connect(self) -> None:
        return

    async def close(self) -> None:
        return

    async def __aenter__(self) -> "LocalEnvironment":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        _ = exc_type, exc, tb
        await self.stop_file_server()
        await self.close()

    async def run(
        self,
        command: str,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> RunResult:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        async def read_stream(
            reader: asyncio.StreamReader | None,
            stream: TextIO | None,
        ) -> str:
            if reader is None:
                return ""
            chunks: list[str] = []
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace")
                chunks.append(text)
                if stream is not None:
                    stream.write(text)
                    stream.flush()
            return "".join(chunks)

        stdout_text, stderr_text = await asyncio.gather(
            read_stream(process.stdout, stdout),
            read_stream(process.stderr, stderr),
        )
        code = await process.wait()
        return RunResult(stdout=stdout_text, stderr=stderr_text, code=code)

    async def health_check(self) -> bool:
        return True

    async def start_file_server(self) -> FileServerSession | None:
        Path(self._root_path).mkdir(parents=True, exist_ok=True)
        return self._file_server

    async def stop_file_server(self) -> None:
        return

    async def file_server_status(self) -> FileServerSession | None:
        return self._file_server

    def sync_from_local(self, src: str, dst: str) -> None:
        destination = self._resolve_sync_path(dst)
        self._copy_path(Path(src), destination)

    def sync_to_local(self, src: str, dst: str, include_paths: list[str] | None = None) -> None:
        source = self._resolve_sync_path(src)
        destination = Path(dst).expanduser()
        if include_paths:
            destination.mkdir(parents=True, exist_ok=True)
            for rel_path in include_paths:
                source_path = source / rel_path
                self._copy_path(source_path, destination / rel_path)
            return
        self._copy_path(source, destination)

    def set_transfer_progress(self, enabled: bool) -> None:
        _ = enabled
        return

    def _resolve_sync_path(self, path: str) -> Path:
        expanded = os.path.expanduser(path)
        if path == "~":
            return Path.home()
        if path.startswith("~/"):
            return Path(expanded)
        return Path(expanded)

    def _copy_path(self, source: Path, destination: Path) -> None:
        if not source.exists():
            raise FileNotFoundError(f"Local transfer source not found: {source}")
        if source.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                self._copy_path(child, destination / child.name)
            return
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
