from sd_train.domain.refs import DownloadAuth


def mask_secret(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def build_download_auth(other_options: dict[str, str]) -> DownloadAuth:
    return DownloadAuth(
        hf_token=str(other_options.get("hf_token", "")).strip(),
        civitai_api_key=str(other_options.get("civitai_api_key", "")).strip(),
    )
