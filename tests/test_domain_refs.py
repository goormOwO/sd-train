from pathlib import Path

import pytest

from tests.helpers import FakeResponse
from sd_train.domain.refs import (
    DownloadAuth,
    _check_hf_ref_access,
    _check_civitai_ref_access,
    _collect_external_refs_from_train_config,
    _extract_civitai_filename_from_location,
    _resolve_civitai_download_id,
    looks_like_civitai_ref,
    looks_like_explicit_remote_ref,
    looks_like_hf_ref,
    parse_civitai_ref,
    parse_hf_ref,
    verify_external_refs_accessibility,
)


def test_parse_hf_ref_with_revision_and_subpath() -> None:
    ref = parse_hf_ref("model:org/repo@main::weights/model.safetensors")
    assert ref.repo_type == "model"
    assert ref.repo_id == "org/repo"
    assert ref.revision == "main"
    assert ref.subpath == "weights/model.safetensors"


def test_parse_civitai_ref_with_filename() -> None:
    resolved, filename = parse_civitai_ref("civitai:46846::lora.safetensors")
    assert resolved == "46846"
    assert filename == "lora.safetensors"


def test_ref_shape_predicates() -> None:
    assert looks_like_hf_ref("org/repo")
    assert looks_like_civitai_ref("civitai:46846")
    assert looks_like_explicit_remote_ref("model:org/repo::x.safetensors")
    assert not looks_like_hf_ref(str(Path("/tmp/not-a-ref")))


def test_parse_hf_ref_invalid_shape_raises() -> None:
    with pytest.raises(ValueError):
        parse_hf_ref("invalid-repo-id")


def test_parse_civitai_ref_from_url_model_version() -> None:
    resolved, filename = parse_civitai_ref(
        "civitai:https://civitai.com/models/1234?modelVersionId=5678::abc.safetensors"
    )
    assert resolved == "5678"
    assert filename == "abc.safetensors"


def test_extract_civitai_filename_from_location() -> None:
    name = _extract_civitai_filename_from_location(
        "https://x.example/file?response-content-disposition=attachment%3B%20filename%3D%22model.safetensors%22"
    )
    assert name == "model.safetensors"


def test_check_hf_ref_access_subpath_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return FakeResponse(status_code=404)

    import sd_train.domain.refs as refs

    monkeypatch.setattr(refs.requests, "get", _fake_get)
    result = _check_hf_ref_access("model:org/repo::missing.bin", DownloadAuth())
    assert result.ok is False
    assert "Not found" in result.detail


def test_collect_external_refs_from_train_config(tmp_path: Path) -> None:
    train_file = tmp_path / "train.toml"
    local_file = tmp_path / "exists.safetensors"
    local_file.write_text("x", encoding="utf-8")
    train_file.write_text(
        "\n".join(
            [
                f'network_weights = "{local_file}"',
                'pretrained_model_name_or_path = "model:org/repo::model.safetensors"',
                'vae = "civitai:46846"',
                'output_dir = "./outputs"',
            ]
        ),
        encoding="utf-8",
    )

    refs = _collect_external_refs_from_train_config(train_file)
    assert refs == [
        ("pretrained_model_name_or_path", "model:org/repo::model.safetensors", "hf"),
        ("vae", "civitai:46846", "civitai"),
    ]


def test_collect_external_refs_from_train_config_includes_explicit_ref_for_unknown_key(
    tmp_path: Path,
) -> None:
    train_file = tmp_path / "train.toml"
    train_file.write_text(
        '\n'.join(['custom_text_encoder = "model:org/repo::encoder.safetensors"']),
        encoding="utf-8",
    )

    refs = _collect_external_refs_from_train_config(train_file)
    assert refs == [("custom_text_encoder", "model:org/repo::encoder.safetensors", "hf")]


def test_verify_external_refs_accessibility_mixed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import sd_train.domain.refs as refs

    train_file = tmp_path / "train.toml"
    train_file.write_text(
        "\n".join(
            [
                'pretrained_model_name_or_path = "model:org/repo::model.safetensors"',
                'vae = "civitai:46846"',
            ]
        ),
        encoding="utf-8",
    )

    def _fake_hf(_value: str, _auth: DownloadAuth):  # noqa: ANN001
        return refs.ExternalRefCheck(key="", ref="x", provider="hf", ok=True, detail="OK")

    def _fake_civitai(_value: str, _auth: DownloadAuth):  # noqa: ANN001
        return refs.ExternalRefCheck(key="", ref="y", provider="civitai", ok=False, detail="bad")

    monkeypatch.setattr(refs, "_check_hf_ref_access", _fake_hf)
    monkeypatch.setattr(refs, "_check_civitai_ref_access", _fake_civitai)
    checks = verify_external_refs_accessibility(train_file, DownloadAuth(hf_token="a", civitai_api_key="b"))
    assert [check.key for check in checks] == ["pretrained_model_name_or_path", "vae"]
    assert checks[0].ok is True
    assert checks[1].ok is False


def test_check_hf_ref_access_repo_unauthorized(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_get(*_args, **_kwargs):  # noqa: ANN002, ANN003
        return FakeResponse(status_code=401)

    import sd_train.domain.refs as refs

    monkeypatch.setattr(refs.requests, "get", _fake_get)
    result = _check_hf_ref_access("model:org/repo", DownloadAuth())
    assert result.ok is False
    assert "Unauthorized" in result.detail


def test_resolve_civitai_download_id_missing_key_raises() -> None:
    with pytest.raises(RuntimeError, match="Missing CivitAI API key"):
        _resolve_civitai_download_id("civitai:123", DownloadAuth())


def test_check_civitai_ref_access_returns_failure_on_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    import sd_train.domain.refs as refs

    def _boom(*_args, **_kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("unauthorized")

    monkeypatch.setattr(refs, "_resolve_civitai_download_id", _boom)
    check = _check_civitai_ref_access("civitai:123", DownloadAuth(civitai_api_key="x"))
    assert check.ok is False
    assert "unauthorized" in check.detail
