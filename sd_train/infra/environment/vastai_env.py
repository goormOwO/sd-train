import ast
import asyncio
import importlib
import re
import secrets
import shlex
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, cast

import asyncssh
from pydantic import BaseModel

from sd_train.infra.environment.ssh_env import SSH
from sd_train.infra.environment.types import FileServerSession

_vastai_sdk: Any = importlib.import_module("vastai_sdk")
VastAI = cast(type[Any], getattr(_vastai_sdk, "VastAI"))
RCLONE_INTERNAL_PORT = 8081


class VastAIConfig(BaseModel):
    api_key: str
    offer_query: str = (
        "gpu_name=RTX_5090 inet_up>=1024 inet_down>=1024 reliability>0.98 direct_port_count>=2 external=true rentable=true verified=true"
    )
    order: str = "dph"
    disk: int = 50


class VastAIInstance(SSH):
    def __init__(
        self,
        client: "VastAIClient",
        id: int,
        identity_file: str,
        temp_key_dir: str,
    ):
        self.client = client
        self.id = id
        self._temp_key_dir = temp_key_dir
        self._destroyed = False
        ssh_host, ssh_port = self._wait_for_ssh_endpoint()
        super().__init__(
            host=ssh_host, port=ssh_port, user="root", identity_file=identity_file
        )

    def _wait_for_ssh_endpoint(self) -> tuple[str, int]:
        deadline = time.time() + 300
        last_raw: Any = None
        while True:
            last_raw = self.client.client.show_instance(id=self.id)
            endpoint = self._extract_ssh_endpoint(last_raw)
            if endpoint is not None:
                return endpoint
            if time.time() >= deadline:
                raise RuntimeError(
                    f"Timed out waiting for VastAI SSH endpoint (instance={self.id}, last={last_raw!r})"
                )
            time.sleep(1)

    def _extract_ssh_endpoint(self, raw: Any) -> tuple[str, int] | None:
        if not isinstance(raw, dict):
            return None

        ssh_host_raw = raw.get("ssh_host")
        if not ssh_host_raw:
            return None
        ssh_host = str(ssh_host_raw)

        # Prefer explicit mapped SSH port when present.
        ports_raw = raw.get("ports")
        if isinstance(ports_raw, dict):
            mappings = ports_raw.get("22/tcp")
            if isinstance(mappings, list) and len(mappings) > 0:
                first = mappings[0]
                if isinstance(first, dict):
                    mapped = first.get("HostPort")
                    try:
                        mapped_port = int(str(mapped))
                    except (TypeError, ValueError):
                        mapped_port = 0
                    if mapped_port > 0:
                        return ssh_host, mapped_port

        ssh_port_raw = raw.get("ssh_port")
        try:
            ssh_port = int(str(ssh_port_raw))
        except (TypeError, ValueError):
            ssh_port = 0
        image_runtype = str(raw.get("image_runtype", "")).lower()
        if ssh_port > 0 and "jupyter" in image_runtype:
            ssh_port += 1
        if ssh_port > 0:
            return ssh_host, ssh_port
        return None

    async def connect(self) -> None:
        deadline = time.monotonic() + 120.0
        last_permission_error: Exception | None = None
        while True:
            try:
                await super().connect()
                return
            except asyncssh.PermissionDenied as exc:
                last_permission_error = exc
                if time.monotonic() >= deadline:
                    self._destroy_instance_and_tempdir()
                    raise RuntimeError(
                        f"Permission denied for user root on host {self._resolved.host}"
                    ) from exc
                await asyncio.sleep(1)
            except Exception:
                self._destroy_instance_and_tempdir()
                raise

        # Unreachable, but keeps typing explicit if loop logic changes later.
        if last_permission_error is not None:
            raise last_permission_error

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            await super().__aexit__(exc_type, exc, tb)
        finally:
            self._destroy_instance_and_tempdir()

    def _destroy_instance_and_tempdir(self) -> None:
        if not self._destroyed:
            try:
                self.client.client.destroy_instance(id=self.id)
            finally:
                self._destroyed = True
        shutil.rmtree(self._temp_key_dir, ignore_errors=True)

    async def start_file_server(self) -> FileServerSession | None:
        status = await self.file_server_status()
        if status is not None:
            return status

        check = await self.run("command -v rclone >/dev/null 2>&1")
        if check.code != 0:
            return None

        bind_port, external_host, external_port = await self._resolve_public_http_port()
        if bind_port is None or external_host is None or external_port is None:
            return None

        username = f"u{secrets.token_hex(6)}"
        password = secrets.token_urlsafe(18)
        root_path = "~/.sd-train"

        await self.run(f"mkdir -p {root_path}")

        quoted_user = shlex.quote(username)
        quoted_pass = shlex.quote(password)
        serve_cmd = (
            "nohup rclone serve webdav ~/.sd-train "
            f"--addr 0.0.0.0:{bind_port} "
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
            f"http://127.0.0.1:{bind_port}/ >/dev/null"
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
            host=external_host,
            port=external_port,
            username=username,
            password=password,
            root_path=root_path,
            pid=pid,
            url=f"http://{external_host}:{external_port}/",
        )
        return self._file_server

    async def _resolve_public_http_port(
        self,
    ) -> tuple[int | None, str | None, int | None]:
        instance_info = self.client.client.show_instance(id=self.id)
        if not isinstance(instance_info, dict):
            return None, None, None

        public_host = str(
            instance_info.get("public_ipaddr") or instance_info.get("ssh_host") or ""
        )
        if not public_host:
            return None, None, None

        # Prefer env exposed by Vast inside the running instance.
        env_probe = await self.run(
            f'python3 -c \'import os;print(os.environ.get("VAST_TCP_PORT_{RCLONE_INTERNAL_PORT}", ""))\''
        )
        env_port_raw = env_probe.stdout.strip() if env_probe.code == 0 else ""
        if env_port_raw:
            try:
                return RCLONE_INTERNAL_PORT, public_host, int(str(env_port_raw))
            except ValueError:
                pass

        ports = instance_info.get("ports")
        if not isinstance(ports, dict):
            return None, None, None
        mapping_key = f"{RCLONE_INTERNAL_PORT}/tcp"
        host_mappings = ports.get(mapping_key)
        if not isinstance(host_mappings, list) or len(host_mappings) == 0:
            return None, None, None
        mapping = host_mappings[0]
        if not isinstance(mapping, dict):
            return None, None, None
        host_port_raw = mapping.get("HostPort")
        try:
            host_port = int(str(host_port_raw))
        except (TypeError, ValueError):
            return None, None, None
        return RCLONE_INTERNAL_PORT, public_host, host_port


class VastAIClient:
    def __init__(self, config: VastAIConfig):
        self.config = config
        self.client = VastAI(api_key=config.api_key)

    def get_offer(
        self,
        query: str | None = None,
        order: str | None = None,
        disk: int | None = None,
    ) -> int:
        final_query = self._enforce_direct_port_query(query or self.config.offer_query)
        offers = self.client.search_offers(
            type="on-demand",
            query=final_query,
            order=order or self.config.order,
            storage=disk if disk is not None else self.config.disk,
        )
        if not isinstance(offers, list) or len(offers) == 0:
            raise RuntimeError("No VastAI offers found for the requested query")
        offer = offers[0]
        if not isinstance(offer, dict) or "id" not in offer:
            raise RuntimeError(f"Unexpected VastAI offer response: {offers!r}")
        return int(offer["id"])

    def create_instance(self, offer_id: int, disk: int | None = None) -> VastAIInstance:
        label = self._build_instance_label()
        before_ids = self._list_instance_ids()
        instance_id: int | None = None
        try:
            instance = self.client.create_instance(
                id=offer_id,
                disk=disk if disk is not None else self.config.disk,
                template_hash="305ac3ffd3e42e0d9ad1f4ae14729ec2",
                direct=True,
                env=f"-p {RCLONE_INTERNAL_PORT}:{RCLONE_INTERNAL_PORT}",
                label=label,
            )

            if isinstance(instance, dict):
                instance_id_raw = instance.get("new_contract")
                if instance_id_raw is not None:
                    instance_id = int(instance_id_raw)

            if instance_id is None:
                instance_id = self._parse_instance_id_from_last_output()
            if instance_id is None:
                instance_id = self._wait_for_instance_id_by_label(
                    label=label, timeout_sec=60
                )
            if instance_id is None:
                instance_id = self._wait_for_new_instance_id(before_ids, timeout_sec=60)
            if instance_id is None:
                last_output = getattr(self.client, "last_output", "")
                raise RuntimeError(
                    "VastAI instance creation was requested but no new instance id was detected. "
                    f"raw_response={instance!r} last_output={last_output!r}"
                )

            self.client.start_instance(id=instance_id)
            private_key_path, public_key, key_dir = self._create_temp_ssh_keypair()
            try:
                self._attach_ssh_key(instance_id=instance_id, ssh_public_key=public_key)
            except Exception:
                shutil.rmtree(key_dir, ignore_errors=True)
                raise
            return VastAIInstance(
                client=self,
                id=instance_id,
                identity_file=private_key_path,
                temp_key_dir=key_dir,
            )
        except BaseException:
            if instance_id is None:
                instance_id = self._wait_for_instance_id_by_label(
                    label=label, timeout_sec=5
                )
            if instance_id is not None:
                try:
                    self.client.destroy_instance(id=instance_id)
                except Exception:
                    pass
            raise

    def _parse_instance_id_from_last_output(self) -> int | None:
        output = str(getattr(self.client, "last_output", "") or "")
        if not output:
            return None
        # Typical output example: "Started. {'success': True, 'new_contract': 7835610}"
        matched = re.search(r"\{.*\}", output, re.DOTALL)
        if matched is None:
            return None
        try:
            payload = ast.literal_eval(matched.group(0))
        except (ValueError, SyntaxError):
            return None
        if not isinstance(payload, dict):
            return None
        value = payload.get("new_contract")
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def _list_instance_ids(self) -> set[int]:
        rows = self.client.show_instances()
        ids: set[int] = set()
        if not isinstance(rows, list):
            return ids
        for row in rows:
            if not isinstance(row, dict):
                continue
            value = row.get("id")
            try:
                ids.add(int(str(value)))
            except (TypeError, ValueError):
                continue
        return ids

    def _list_instances(self) -> list[dict[str, Any]]:
        rows = self.client.show_instances()
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
        return []

    def _wait_for_new_instance_id(
        self, before_ids: set[int], timeout_sec: int = 60
    ) -> int | None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            after_ids = self._list_instance_ids()
            created = [value for value in after_ids if value not in before_ids]
            if len(created) == 1:
                return created[0]
            if len(created) > 1:
                created.sort()
                return created[-1]
            time.sleep(1)
        return None

    def _wait_for_instance_id_by_label(
        self, label: str, timeout_sec: int = 60
    ) -> int | None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            matches: list[int] = []
            for row in self._list_instances():
                if str(row.get("label", "")) != label:
                    continue
                value = row.get("id")
                try:
                    matches.append(int(str(value)))
                except (TypeError, ValueError):
                    continue
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                matches.sort()
                return matches[-1]
            time.sleep(1)
        return None

    def _build_instance_label(self) -> str:
        return f"sd-train-{int(time.time())}-{secrets.token_hex(4)}"

    def _create_temp_ssh_keypair(self) -> tuple[str, str, str]:
        temp_dir = tempfile.mkdtemp(prefix="sdxl-vastai-key-")
        private_key = Path(temp_dir) / "id_ed25519"
        command = [
            "ssh-keygen",
            "-t",
            "ed25519",
            "-N",
            "",
            "-f",
            str(private_key),
            "-q",
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except FileNotFoundError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                "ssh-keygen is required to create temporary VastAI identity"
            ) from exc
        except subprocess.CalledProcessError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise RuntimeError(
                f"Failed to create temporary VastAI identity: {exc.stderr}"
            ) from exc

        public_key = private_key.with_suffix(".pub").read_text(encoding="utf-8").strip()
        return str(private_key), public_key, temp_dir

    def _attach_ssh_key(self, instance_id: int, ssh_public_key: str) -> None:
        deadline = time.time() + 60
        last_error: Exception | None = None
        while time.time() < deadline:
            try:
                self.client.attach_ssh(instance_id=instance_id, ssh_key=ssh_public_key)
                return
            except Exception as exc:
                last_error = exc
                time.sleep(1)
        if last_error is not None:
            raise RuntimeError(
                f"Failed to attach temporary SSH key to instance {instance_id}"
            ) from last_error
        raise RuntimeError(
            f"Failed to attach temporary SSH key to instance {instance_id}"
        )

    def _enforce_direct_port_query(self, query: str) -> str:
        # Remove any existing direct_port_count constraint and force >=2.
        normalized = re.sub(
            r"(?<!\S)direct_port_count\s*(?:=|!=|>=|<=|>|<|in|notin)\s*(?:\[[^\]]*\]|\"[^\"]*\"|'[^']*'|\S+)",
            "",
            query,
            flags=re.IGNORECASE,
        )
        normalized = " ".join(normalized.split())
        if normalized:
            return f"{normalized} direct_port_count>=2"
        return "direct_port_count>=2"
