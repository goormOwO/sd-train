from typing import Any, Literal

from pydantic import BaseModel, Field


DEFAULT_LOCAL_ENVIRONMENT_NAME = "local"
DEFAULT_VASTAI_OFFER_QUERY = (
    "gpu_name=RTX_5090 inet_up>=1024 inet_down>=1024 reliability>0.98 "
    "direct_port_count>=2 external=true rentable=true verified=true"
)


class LastSelection(BaseModel):
    environment_name: str = ""
    train_config_path: str = ""
    train_script: str = ""


class SSHEnvironmentConfig(BaseModel):
    name: str
    type: Literal["ssh"] = "ssh"
    host: str = ""
    user: str = ""
    port: int = 22
    identity_file: str = ""


class VastAIEnvironmentConfig(BaseModel):
    name: str
    type: Literal["vastai"] = "vastai"
    api_key: str = ""
    offer_query: str = DEFAULT_VASTAI_OFFER_QUERY
    order: str = "dph"
    disk: int = 50


class LocalEnvironmentConfig(BaseModel):
    name: str = DEFAULT_LOCAL_ENVIRONMENT_NAME
    type: Literal["local"] = "local"


EnvironmentConfig = LocalEnvironmentConfig | SSHEnvironmentConfig | VastAIEnvironmentConfig


class TaggerConfig(BaseModel):
    dataset_dir: str = ""
    model: str = "SmilingWolf/wd-vit-tagger-v3"
    threshold: float = 0.4
    batch: int = 1


class AppConfig(BaseModel):
    environments: list[dict[str, Any]] = Field(default_factory=list)
    last: LastSelection = Field(default_factory=LastSelection)
    tagger: TaggerConfig = Field(default_factory=TaggerConfig)
    other_options: dict[str, str] = Field(default_factory=dict)


def default_local_environment() -> dict[str, str]:
    return {
        "name": DEFAULT_LOCAL_ENVIRONMENT_NAME,
        "type": "local",
    }


def normalize_app_config(config: AppConfig) -> AppConfig:
    other_environments = [
        dict(env)
        for env in config.environments
        if not (
            str(env.get("type", "")).lower() == "local"
            or str(env.get("name", "")).strip() == DEFAULT_LOCAL_ENVIRONMENT_NAME
        )
    ]
    config.environments = [default_local_environment(), *other_environments]

    valid_names = {
        str(env.get("name", "")).strip() for env in config.environments if str(env.get("name", "")).strip()
    }
    current_name = config.last.environment_name.strip()
    if not current_name or current_name not in valid_names:
        config.last.environment_name = DEFAULT_LOCAL_ENVIRONMENT_NAME
    return config
