from sd_train.core.source_auth import build_download_auth, mask_secret


def test_mask_secret_behavior() -> None:
    assert mask_secret("abcd") == "****"
    assert mask_secret("abcdef") == "ab**ef"


def test_build_download_auth_trims_values() -> None:
    auth = build_download_auth({"hf_token": "  hf  ", "civitai_api_key": "  civ  "})
    assert auth.hf_token == "hf"
    assert auth.civitai_api_key == "civ"
