from sd_train.core import environment_setup
from sd_train.core.environment_setup import build_environment, env_to_model
from sd_train.infra.environment.ssh_env import SSH


def test_build_environment_for_ssh() -> None:
    model = env_to_model(
        {"name": "dev", "type": "ssh", "host": "127.0.0.1", "user": "me", "port": 22}
    )
    env = build_environment(model)
    assert isinstance(env, SSH)


def test_build_environment_for_vastai(monkeypatch) -> None:
    class FakeInstance:
        pass

    class FakeVastClient:
        def __init__(self, _config) -> None:
            return

        def get_offer(self, query: str | None, order: str | None, disk: int | None) -> int:
            return 123

        def create_instance(self, offer_id: int, disk: int | None = None) -> FakeInstance:
            assert offer_id == 123
            return FakeInstance()

    monkeypatch.setattr(environment_setup, "VastAIClient", FakeVastClient)

    model = env_to_model({"name": "vast", "type": "vastai", "api_key": "x"})
    env = build_environment(model)
    assert isinstance(env, FakeInstance)
