import os
import asyncio
import shlex
import secrets
import time
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional, TextIO

import asyncssh
import paramiko
from rclone_python import rclone as rclone_client
from rclone_python import utils as rclone_utils

from sd_train.infra.environment.base import Environment
from sd_train.infra.environment.types import FileServerSession, RunResult


@dataclass
class SSHResolvedConfig:
    host: str
    user: Optional[str]
    port: int
    identity_files: list[str]


def _resolve_ssh_config(
    host: str, user: str | None, port: int, identity_file: str | None
) -> SSHResolvedConfig:
    config = paramiko.SSHConfig()
    config_path = os.path.expanduser("~/.ssh/config")
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            config.parse(f)

    config_dict = config.lookup(host)
    resolved_user = config_dict.get("user", user)
    resolved_host = config_dict.get("hostname", host)
    resolved_port = int(config_dict.get("port", port))

    resolved_identity_files: list[str] = []
    if identity_file is not None:
        resolved_identity_files = [os.path.expanduser(identity_file)]
    else:
        value = config_dict.get("identityfile")
        if isinstance(value, list) and value:
            existing: list[str] = []
            fallback: list[str] = []
            for item in value:
                if not isinstance(item, str):
                    continue
                candidate = os.path.expanduser(item)
                if os.path.exists(candidate):
                    existing.append(candidate)
                else:
                    fallback.append(candidate)
            resolved_identity_files = existing if existing else fallback
        elif isinstance(value, str):
            resolved_identity_files = [os.path.expanduser(value)]

    return SSHResolvedConfig(
        host=resolved_host,
        user=resolved_user,
        port=resolved_port,
        identity_files=resolved_identity_files,
    )


class SSH(Environment):
    def __init__(
        self,
        host: str,
        user: str | None = None,
        port: int = 22,
        identity_file: str | None = None,
    ):
        self._resolved = _resolve_ssh_config(host, user, port, identity_file)
        self._conn: asyncssh.SSHClientConnection | None = None
        self._file_server: FileServerSession | None = None
        self._show_transfer_progress = False

    def connection_target(self) -> str:
        user = self._resolved.user or "unknown"
        return f"{user}@{self._resolved.host}:{self._resolved.port}"

    def set_transfer_progress(self, enabled: bool) -> None:
        self._show_transfer_progress = enabled

    def transfer_target(self) -> tuple[str | None, str, int, str | None]:
        identity_file: str | None = None
        for candidate in self._resolved.identity_files:
            if os.path.exists(candidate):
                identity_file = candidate
                break
        return (
            self._resolved.user,
            self._resolved.host,
            self._resolved.port,
            identity_file,
        )

    async def __aenter__(self) -> "SSH":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.stop_file_server()
        await self.close()

    async def connect(self) -> None:
        if self._conn is not None:
            return

        # Try key candidates in order like OpenSSH fallback logic.
        candidates: list[list[str] | None] = []
        for key in self._resolved.identity_files:
            candidates.append([key])
        # Final fallback: let asyncssh use its default key discovery/agent.
        candidates.append(None)

        deadline = time.monotonic() + 120.0
        last_error: Exception | None = None
        while True:
            saw_permission_denied = False
            for client_keys in candidates:
                try:
                    kwargs = {
                        "port": self._resolved.port,
                        "known_hosts": None,
                    }
                    if self._resolved.user is not None:
                        kwargs["username"] = self._resolved.user
                    if client_keys is not None:
                        kwargs["client_keys"] = client_keys
                    self._conn = await asyncssh.connect(self._resolved.host, **kwargs)
                    return
                except asyncssh.PermissionDenied as exc:
                    last_error = exc
                    saw_permission_denied = True
                    continue
                except (OSError, asyncssh.Error) as exc:
                    last_error = exc
                    continue

            # Authentication failures are unlikely to recover by retrying.
            if saw_permission_denied and isinstance(last_error, asyncssh.PermissionDenied):
                raise last_error

            if time.monotonic() >= deadline:
                break
            await asyncio.sleep(1)

        if last_error is not None:
            raise last_error
        raise RuntimeError("Failed to establish SSH connection")

    async def close(self) -> None:
        if self._conn is None:
            return
        self._conn.close()
        await self._conn.wait_closed()
        self._conn = None

    async def run(
        self,
        command: str,
        stdout: TextIO | None = None,
        stderr: TextIO | None = None,
    ) -> RunResult:
        if self._conn is None:
            raise RuntimeError("SSH is not connected")

        process = await self._conn.create_process(
            command,
            encoding="utf-8",
            stderr=-1,
        )

        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        async def read_stream(
            reader: asyncssh.SSHReader[str],
            sink: list[str],
            stream: TextIO | None,
        ) -> None:
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                sink.append(chunk)
                if stream is not None:
                    stream.write(chunk)
                    stream.flush()

        await asyncio.gather(
            read_stream(process.stdout, stdout_chunks, stdout),
            read_stream(process.stderr, stderr_chunks, stderr),
        )

        completed = await process.wait(check=False)
        exit_status = completed.exit_status if completed.exit_status is not None else 1
        return RunResult(
            stdout="".join(stdout_chunks),
            stderr="".join(stderr_chunks),
            code=exit_status,
        )

    async def health_check(self) -> bool:
        result = await self.run("true")
        return result.code == 0

    async def file_server_status(self) -> FileServerSession | None:
        if self._file_server is None:
            return None
        pid = self._file_server.pid
        if pid is None:
            self._file_server = None
            return None
        result = await self.run(f"kill -0 {pid}")
        if result.code != 0:
            self._file_server = None
            return None
        return self._file_server

    async def start_file_server(self) -> FileServerSession | None:
        status = await self.file_server_status()
        if status is not None:
            return status

        check = await self.run("command -v rclone >/dev/null 2>&1")
        if check.code != 0:
            return None

        username = f"u{secrets.token_hex(6)}"
        password = secrets.token_urlsafe(18)
        root_path = "~/.sd-train"
        port = await self._pick_random_listen_port()
        if port is None:
            return None

        await self.run(f"mkdir -p {root_path}")

        quoted_user = shlex.quote(username)
        quoted_pass = shlex.quote(password)
        serve_cmd = (
            "nohup rclone serve webdav ~/.sd-train "
            f"--addr 0.0.0.0:{port} "
            f"--user {quoted_user} --pass {quoted_pass} "
            "> ~/.sd-train/rclone-serve.log 2>&1 & echo $!"
        )
        start_result = await self.run(serve_cmd)
        if start_result.code != 0:
            return None

        pid_text = start_result.stdout.strip().splitlines()
        pid = None
        if pid_text:
            try:
                pid = int(pid_text[-1].strip())
            except ValueError:
                pid = None

        if pid is None:
            return None

        auth_user = shlex.quote(username)
        auth_pass = shlex.quote(password)
        health_cmd = (
            f"curl -fsS -u {auth_user}:{auth_pass} "
            f"http://127.0.0.1:{port}/ >/dev/null"
        )
        healthy = False
        for _ in range(5):
            health = await self.run(health_cmd)
            if health.code == 0:
                healthy = True
                break
            await asyncio.sleep(0.5)

        if not healthy:
            await self.run(f"kill {pid} >/dev/null 2>&1 || true")
            return None

        self._file_server = FileServerSession(
            protocol="webdav",
            host=self._resolved.host,
            port=port,
            username=username,
            password=password,
            root_path=root_path,
            pid=pid,
            url=f"http://{self._resolved.host}:{port}/",
        )
        return self._file_server

    async def _pick_random_listen_port(self) -> int | None:
        # Use Python's socket bind(0) remotely to choose an available ephemeral port.
        result = await self.run(
            "python3 -c 'import socket;s=socket.socket();s.bind((\"127.0.0.1\",0));print(s.getsockname()[1]);s.close()'"
        )
        if result.code != 0:
            return None

        text = result.stdout.strip()
        try:
            port = int(text.splitlines()[-1])
        except (ValueError, IndexError):
            return None
        if port < 1 or port > 65535:
            return None
        return port

    async def stop_file_server(self) -> None:
        if self._file_server is None:
            return
        pid = self._file_server.pid
        if pid is not None:
            await self.run(f"kill {pid} >/dev/null 2>&1 || true")
        self._file_server = None

    def sync_from_local(self, src: str, dst: str):
        self._rclone_transfer(src, dst, upload=True)

    def sync_to_local(self, src: str, dst: str, include_paths: list[str] | None = None):
        self._rclone_transfer(src, dst, upload=False, include_paths=include_paths)

    def _rclone_transfer(
        self,
        src: str,
        dst: str,
        upload: bool,
        include_paths: list[str] | None = None,
    ) -> None:
        session = self._file_server
        if session is None:
            raise RuntimeError("rclone file server is not started")
        if not rclone_client.is_installed():
            raise RuntimeError("Local rclone command not found")

        previous_config = rclone_utils.Config().config_path
        temp_config = tempfile.NamedTemporaryFile(
            mode="w", suffix=".conf", delete=False, encoding="utf-8"
        )
        temp_config.close()
        remote_name = f"sdxl_{secrets.token_hex(4)}"

        temp_files_from_path: str | None = None
        try:
            obscured_password = self._obscure_rclone_password(session.password)
            rclone_client.set_config_file(temp_config.name)
            rclone_client.create_remote(
                remote_name=remote_name,
                remote_type="webdav",
                url=session.url.rstrip("/"),
                vendor="other",
                user=session.username,
                **{"pass": obscured_password},
            )

            remote_path = self._normalize_remote_path(src if not upload else dst)
            remote_spec = f"{remote_name}:{remote_path}"
            transfer_args = self._high_bandwidth_rclone_args()

            if upload:
                local_path = src
                if os.path.isdir(local_path):
                    rclone_client.sync(
                        src_path=local_path,
                        dest_path=remote_spec,
                        show_progress=self._show_transfer_progress,
                        args=transfer_args,
                    )
                else:
                    rclone_client.copyto(
                        in_path=local_path,
                        out_path=remote_spec,
                        show_progress=self._show_transfer_progress,
                        args=transfer_args,
                    )
                return

            local_path = dst
            if include_paths:
                os.makedirs(local_path, exist_ok=True)
                with tempfile.NamedTemporaryFile(
                    mode="w", suffix=".txt", delete=False, encoding="utf-8"
                ) as temp_files_from:
                    for path in include_paths:
                        temp_files_from.write(path)
                        temp_files_from.write("\n")
                    temp_files_from_path = temp_files_from.name
                rclone_client.copy(
                    in_path=remote_spec,
                    out_path=local_path,
                    show_progress=self._show_transfer_progress,
                    args=[*transfer_args, "--files-from", temp_files_from_path],
                )
                return
            destination_is_dir = local_path.endswith(os.sep) or os.path.isdir(local_path)
            if destination_is_dir:
                rclone_client.copy(
                    in_path=remote_spec,
                    out_path=local_path,
                    show_progress=self._show_transfer_progress,
                    args=transfer_args,
                )
            else:
                rclone_client.copyto(
                    in_path=remote_spec,
                    out_path=local_path,
                    show_progress=self._show_transfer_progress,
                    args=transfer_args,
                )
        except Exception as exc:
            mode = "upload" if upload else "download"
            raise RuntimeError(f"rclone {mode} failed: {exc}") from exc
        finally:
            if previous_config is not None:
                rclone_client.set_config_file(str(previous_config))
            else:
                rclone_utils.Config().config_path = None
            try:
                os.remove(temp_config.name)
            except OSError:
                pass
            if temp_files_from_path is not None:
                try:
                    os.remove(temp_files_from_path)
                except OSError:
                    pass

    def _obscure_rclone_password(self, value: str) -> str:
        # rclone config stores secrets in obscured form; plain values can cause auth failures.
        result = rclone_utils.run_rclone_cmd(
            "obscure",
            args=["--", value],
            raise_errors=True,
        )
        if len(result) != 2:
            raise RuntimeError("Unexpected rclone obscure response")
        stdout = result[0]
        obscured = stdout.strip()
        if not obscured:
            raise RuntimeError("Failed to obscure rclone password")
        return obscured

    def _high_bandwidth_rclone_args(self) -> list[str]:
        # Tuned for large-file transfer on high-latency/high-bandwidth links.
        return [
            "--transfers",
            "32",
            "--checkers",
            "64",
            "--multi-thread-streams",
            "16",
            "--multi-thread-cutoff",
            "16M",
            "--buffer-size",
            "64M",
            "--contimeout",
            "10s",
            "--timeout",
            "1m",
            "--retries",
            "5",
            "--low-level-retries",
            "20",
        ]

    def _normalize_remote_path(self, path: str) -> str:
        session = self._file_server
        if session is None:
            return path.lstrip("/")

        value = path.strip()
        root = os.path.expanduser(session.root_path)
        expanded = os.path.expanduser(value)

        if expanded == root:
            return ""
        if expanded.startswith(root + os.sep):
            return expanded[len(root) + 1 :]
        if value.startswith(session.root_path):
            trimmed = value[len(session.root_path) :]
            return trimmed.lstrip("/")
        return value.lstrip("/")
