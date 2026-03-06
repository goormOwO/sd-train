from typing import Any

from sd_train.config.models import EnvironmentConfig, SSHEnvironmentConfig, VastAIEnvironmentConfig
from sd_train.infra.environment.base import Environment
from sd_train.infra.environment.ssh_env import SSH
from sd_train.infra.environment.vastai_env import VastAIClient, VastAIConfig


def env_to_model(raw: dict[str, Any]) -> EnvironmentConfig:
    if str(raw.get("type", "ssh")).lower() == "vastai":
        return VastAIEnvironmentConfig(**raw)
    return SSHEnvironmentConfig(**raw)


def build_environment(config: EnvironmentConfig) -> Environment:
    if isinstance(config, SSHEnvironmentConfig):
        return SSH(
            host=config.host,
            user=config.user or None,
            port=config.port,
            identity_file=config.identity_file or None,
        )

    client = VastAIClient(
        VastAIConfig(
            api_key=config.api_key,
            offer_query=config.offer_query,
            order=config.order,
            disk=config.disk,
        )
    )
    offer_id = client.get_offer(query=config.offer_query, order=config.order, disk=config.disk)
    return client.create_instance(offer_id=offer_id, disk=config.disk)
