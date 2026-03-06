from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qs, unquote, urlparse
import re

import requests
import toml

from sd_train.domain.path_rules import (
    FORCE_REMOTE_OUTPUT_KEYS,
    is_path_value_key,
    sanitize_component,
)


@dataclass
class HFRef:
    repo_type: str
    repo_id: str
    revision: str | None
    subpath: str | None


@dataclass
class DownloadAuth:
    hf_token: str = ""
    civitai_api_key: str = ""


@dataclass
class ExternalRefCheck:
    key: str
    ref: str
    provider: Literal["hf", "civitai"]
    ok: bool
    detail: str


def parse_hf_ref(value: str) -> HFRef:
    ref = value.strip()
    prefix = "model"
    if ref.startswith("dataset:"):
        prefix = "dataset"
        ref = ref.removeprefix("dataset:")
    elif ref.startswith("model:"):
        prefix = "model"
        ref = ref.removeprefix("model:")

    subpath: str | None = None
    if "::" in ref:
        ref, subpath = ref.split("::", 1)
        if subpath:
            subpath = subpath.strip()

    revision: str | None = None
    if "@" in ref:
        ref, revision = ref.split("@", 1)
        revision = revision.strip() or None

    repo_id = ref.strip()
    if not re.match(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", repo_id):
        raise ValueError(f"Invalid Hugging Face reference: {value}")

    return HFRef(repo_type=prefix, repo_id=repo_id, revision=revision, subpath=subpath)


def looks_like_hf_ref(value: str) -> bool:
    raw = value.strip()
    if not raw:
        return False
    if raw.startswith(("model:", "dataset:")):
        return True
    if raw.startswith(("./", "../", "/", "~")):
        return False
    repo_candidate = raw
    if "::" in repo_candidate:
        repo_candidate = repo_candidate.split("::", 1)[0]
    if "@" in repo_candidate:
        repo_candidate = repo_candidate.split("@", 1)[0]
    return re.match(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$", repo_candidate) is not None


def parse_civitai_ref(value: str) -> tuple[str, str | None]:
    raw = value.strip()
    if not raw.startswith("civitai:"):
        raise ValueError(f"Invalid CivitAI reference: {value}")

    payload = raw.removeprefix("civitai:").strip()
    filename: str | None = None
    if "::" in payload:
        payload, filename = payload.split("::", 1)
        filename = sanitize_component(filename.strip()) if filename.strip() else None
    payload = payload.strip()
    if not payload:
        raise ValueError(f"Invalid CivitAI reference: {value}")

    if payload.isdigit():
        return payload, filename

    parsed = urlparse(payload)
    if not parsed.scheme:
        raise ValueError(f"Invalid CivitAI reference: {value}")

    qs = parse_qs(parsed.query)
    model_version = qs.get("modelVersionId")
    if model_version and model_version[0].isdigit():
        return model_version[0], filename

    match = re.search(r"/api/download/models/(\d+)", parsed.path)
    if match:
        return match.group(1), filename

    match = re.search(r"/models/(\d+)", parsed.path)
    if match:
        return match.group(1), filename

    raise ValueError(
        f"Unable to determine CivitAI model/version id from reference: {value}"
    )


def looks_like_civitai_ref(value: str) -> bool:
    return value.strip().startswith("civitai:")


def looks_like_explicit_remote_ref(value: str) -> bool:
    raw = value.strip()
    return raw.startswith(("model:", "dataset:", "civitai:"))


def _extract_civitai_filename_from_location(location: str | None) -> str | None:
    if not location:
        return None
    parsed = urlparse(location)
    qs = parse_qs(parsed.query)
    content_disposition = qs.get("response-content-disposition", [None])[0]
    if isinstance(content_disposition, str) and "filename=" in content_disposition:
        raw_name = content_disposition.split("filename=", 1)[1].strip('"')
        name = unquote(raw_name).strip()
        if name:
            return sanitize_component(name)
    path_name = Path(parsed.path).name.strip()
    if path_name:
        return sanitize_component(unquote(path_name))
    return None


def _resolve_civitai_download_id(civitai_raw: str, auth: DownloadAuth) -> tuple[str, str | None, str]:
    candidate_id, filename = parse_civitai_ref(civitai_raw)
    token = auth.civitai_api_key.strip()
    if not token:
        raise RuntimeError("Missing CivitAI API key in Other Options.")

    headers = {"Authorization": f"Bearer {token}"}
    download_url = f"https://civitai.com/api/download/models/{candidate_id}"
    response = requests.get(download_url, headers=headers, timeout=20, allow_redirects=False)
    if response.status_code in {200, 301, 302, 303, 307, 308}:
        redirect_name = _extract_civitai_filename_from_location(response.headers.get("Location"))
        return candidate_id, filename or redirect_name, "direct modelVersionId"
    if response.status_code in {401, 403}:
        raise RuntimeError("Unauthorized. Check CivitAI API key in Other Options.")
    if response.status_code != 404:
        raise RuntimeError(f"CivitAI download check failed: HTTP {response.status_code}")

    model_url = f"https://civitai.com/api/v1/models/{candidate_id}"
    model_response = requests.get(model_url, headers=headers, timeout=20)
    if model_response.status_code in {401, 403}:
        raise RuntimeError("Unauthorized. Check CivitAI API key in Other Options.")
    if model_response.status_code == 404:
        raise RuntimeError("Model/version not found.")
    if model_response.status_code != 200:
        raise RuntimeError(f"CivitAI model lookup failed: HTTP {model_response.status_code}")

    payload = model_response.json()
    versions = payload.get("modelVersions")
    if not isinstance(versions, list) or not versions:
        raise RuntimeError("No modelVersions found for given CivitAI model id.")
    first = versions[0]
    if not isinstance(first, dict) or not isinstance(first.get("id"), int):
        raise RuntimeError("Invalid CivitAI modelVersions payload.")
    resolved = str(first["id"])
    resolved_download_url = f"https://civitai.com/api/download/models/{resolved}"
    resolved_resp = requests.get(
        resolved_download_url, headers=headers, timeout=20, allow_redirects=False
    )
    redirect_name = _extract_civitai_filename_from_location(resolved_resp.headers.get("Location"))
    return resolved, filename or redirect_name, f"resolved from model id {candidate_id}"


def _collect_external_refs_from_train_config(
    train_config_path: Path,
) -> list[tuple[str, str, Literal["hf", "civitai"]]]:
    data = toml.load(train_config_path)
    if not isinstance(data, dict):
        return []

    refs: list[tuple[str, str, Literal["hf", "civitai"]]] = []
    for key, value in data.items():
        if not isinstance(value, str):
            continue
        if not is_path_value_key(key) and not looks_like_explicit_remote_ref(value):
            continue
        if key in FORCE_REMOTE_OUTPUT_KEYS:
            continue

        raw_value = value.strip()
        if not raw_value:
            continue

        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = (train_config_path.parent / candidate).resolve()
        if candidate.exists():
            continue

        if looks_like_hf_ref(raw_value):
            refs.append((key, raw_value, "hf"))
            continue
        if looks_like_civitai_ref(raw_value):
            refs.append((key, raw_value, "civitai"))
            continue
    return refs


def _check_hf_ref_access(hf_raw: str, auth: DownloadAuth) -> ExternalRefCheck:
    hf_ref = parse_hf_ref(hf_raw)
    headers: dict[str, str] = {}
    token = auth.hf_token.strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    if hf_ref.subpath:
        revision = hf_ref.revision or "main"
        base = "https://huggingface.co"
        if hf_ref.repo_type == "dataset":
            url = f"{base}/datasets/{hf_ref.repo_id}/resolve/{revision}/{hf_ref.subpath}"
        else:
            url = f"{base}/{hf_ref.repo_id}/resolve/{revision}/{hf_ref.subpath}"
        response = requests.get(url, headers=headers, timeout=20, stream=True, allow_redirects=True)
        response.close()
        if response.status_code == 200:
            return ExternalRefCheck(key="", ref=hf_raw, provider="hf", ok=True, detail=f"OK ({hf_ref.repo_id}::{hf_ref.subpath})")
        if response.status_code in {401, 403}:
            return ExternalRefCheck(key="", ref=hf_raw, provider="hf", ok=False, detail="Unauthorized. Check HF Token in Other Options.")
        if response.status_code == 404:
            return ExternalRefCheck(key="", ref=hf_raw, provider="hf", ok=False, detail="Not found. Check repo/revision/subpath.")
        return ExternalRefCheck(key="", ref=hf_raw, provider="hf", ok=False, detail=f"HTTP {response.status_code}")

    if hf_ref.repo_type == "dataset":
        url = f"https://huggingface.co/api/datasets/{hf_ref.repo_id}"
    else:
        url = f"https://huggingface.co/api/models/{hf_ref.repo_id}"
    params: dict[str, str] = {}
    if hf_ref.revision:
        params["revision"] = hf_ref.revision
    response = requests.get(url, headers=headers, params=params, timeout=20)
    if response.status_code == 200:
        return ExternalRefCheck(key="", ref=hf_raw, provider="hf", ok=True, detail=f"OK ({hf_ref.repo_id})")
    if response.status_code in {401, 403}:
        return ExternalRefCheck(key="", ref=hf_raw, provider="hf", ok=False, detail="Unauthorized. Check HF Token in Other Options.")
    if response.status_code == 404:
        return ExternalRefCheck(key="", ref=hf_raw, provider="hf", ok=False, detail="Repository not found or not accessible.")
    return ExternalRefCheck(key="", ref=hf_raw, provider="hf", ok=False, detail=f"HTTP {response.status_code}")


def _check_civitai_ref_access(civitai_raw: str, auth: DownloadAuth) -> ExternalRefCheck:
    try:
        model_id, _filename, resolution = _resolve_civitai_download_id(civitai_raw, auth)
        return ExternalRefCheck(
            key="",
            ref=civitai_raw,
            provider="civitai",
            ok=True,
            detail=f"OK (download id: {model_id}, {resolution})",
        )
    except RuntimeError as exc:
        return ExternalRefCheck(key="", ref=civitai_raw, provider="civitai", ok=False, detail=str(exc))


def verify_external_refs_accessibility(train_config_path: Path, auth: DownloadAuth) -> list[ExternalRefCheck]:
    refs = _collect_external_refs_from_train_config(train_config_path)
    results: list[ExternalRefCheck] = []
    for key, value, provider in refs:
        if provider == "hf":
            result = _check_hf_ref_access(value, auth)
        else:
            result = _check_civitai_ref_access(value, auth)
        result.key = key
        results.append(result)
    return results
