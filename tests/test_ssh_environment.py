import asyncio

from sd_train.infra.environment import ssh_env
from sd_train.infra.environment.ssh_env import SSH


def test_ssh_connect_omits_username_when_unset(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeConnection:
        def close(self) -> None:
            return

        async def wait_closed(self) -> None:
            return

    async def _fake_connect(host: str, **kwargs):  # noqa: ANN003
        captured["host"] = host
        captured["kwargs"] = kwargs
        return FakeConnection()

    monkeypatch.setattr(
        ssh_env,
        "_resolve_ssh_config",
        lambda *_args: ssh_env.SSHResolvedConfig(host="wsl", user=None, port=22, identity_files=[]),
    )
    monkeypatch.setattr(ssh_env.asyncssh, "connect", _fake_connect)

    env = SSH(host="wsl", user=None, port=22, identity_file=None)
    asyncio.run(env.connect())

    assert captured["host"] == "wsl"
    assert captured["kwargs"] == {"port": 22, "known_hosts": None}

    asyncio.run(env.close())
