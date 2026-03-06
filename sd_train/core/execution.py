import asyncio
import hashlib
import json
import os
import shlex
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml
from colorama import Fore, Style

from sd_train.domain.path_rules import (
    FORCE_REMOTE_OUTPUT_KEYS,
    expected_hf_mode_for_key,
    expected_local_mode_for_key,
    is_path_value_key,
    sanitize_component,
    to_abs_remote,
    to_sync_remote,
)
from sd_train.domain.refs import (
    DownloadAuth,
    looks_like_explicit_remote_ref,
    looks_like_civitai_ref,
    looks_like_hf_ref,
    parse_hf_ref,
)
from sd_train.infra.environment.base import Environment
from sd_train.infra.environment.ssh_env import SSH

BASE_COMMANDS = ["mkdir -p ~/.sd-train"]
TEMP_FILE_SUFFIXES = (".tmp", ".part", ".lock")
SYNC_INTERVAL_SECONDS = 15
SYNC_STABILITY_SECONDS = 20
PINNED_SD_SCRIPTS_COMMIT = "1a3ec9ea745fe9883551dfca5c947ea3d6aa68c7"
PINNED_KOHYA_SS_COMMIT = "4161d1d80ad554f7801c584632665d6825994062"
PINNED_CIVITAI_DOWNLOADER_COMMIT = "cef9dc1db2469c232ce74b0cf4254a2af97e390f"
PINNED_HUGGINGFACE_HUB_VERSION = "0.36.2"
REMOTE_TRAIN_PYTHON_VERSION = "3.11"
UV_INSTALL_URL = "https://astral.sh/uv/install.sh"


@dataclass
class MaterializedRef:
    sync_path: str
    abs_path: str


def _mask_secret(value: str) -> str:
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}{'*' * (len(value) - 4)}{value[-2:]}"


def _normalize_script_path(script: str) -> str:
    normalized = script.strip().replace("\\", "/")
    normalized = normalized.lstrip("/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized:
        raise ValueError("Train script path is empty")
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    if any(part == ".." for part in parts):
        raise ValueError("Train script must be a relative path inside sd-scripts")
    return "/".join(parts)


async def _run_commands(environment: Environment, commands: list[str]) -> None:
    for command in commands:
        print(f"{Fore.YELLOW}$ {command}{Style.RESET_ALL}")
        result = await environment.run(command, stdout=sys.stdout, stderr=sys.stderr)
        if result.code != 0:
            raise RuntimeError(f"Command failed ({result.code}): {command}")


def _set_transfer_progress(environment: Environment, enabled: bool) -> None:
    setter = getattr(environment, "set_transfer_progress", None)
    if callable(setter):
        setter(enabled)


async def _ensure_remote_sd_scripts(environment: Environment, remote_home: str) -> None:
    sd_scripts_abs = to_abs_remote("~/.sd-train/sd-scripts", remote_home)
    command = (
        "command -v git >/dev/null 2>&1 || { echo 'git not found on remote' >&2; exit 2; }; "
        f"if [ ! -d {shlex.quote(sd_scripts_abs)}/.git ]; then "
        f"git clone --filter=blob:none https://github.com/kohya-ss/sd-scripts.git {shlex.quote(sd_scripts_abs)}; "
        "fi; "
        f"cd {shlex.quote(sd_scripts_abs)} && "
        f"git fetch --depth 1 origin {PINNED_SD_SCRIPTS_COMMIT} && "
        f"git checkout --detach {PINNED_SD_SCRIPTS_COMMIT} && "
        "git submodule sync --recursive && "
        "git submodule update --init --recursive --depth 1"
    )
    result = await environment.run(command, stdout=sys.stdout, stderr=sys.stderr)
    if result.code != 0:
        raise RuntimeError(
            "Failed to prepare remote sd-scripts repository. "
            "Check git/network availability on remote."
        )


async def _prepare_remote_python_venv(
    environment: Environment, remote_home: str
) -> tuple[str, str]:
    setup_root_abs = to_abs_remote("~/.sd-train", remote_home)
    sd_scripts_abs = to_abs_remote("~/.sd-train/sd-scripts", remote_home)
    venv_abs = to_abs_remote("~/.sd-train/.venv", remote_home)
    uv_bin_abs = to_abs_remote("~/.local/bin/uv", remote_home)
    setup_cfg = {
        "setup_root_abs": setup_root_abs,
        "sd_scripts_abs": sd_scripts_abs,
        "venv_abs": venv_abs,
        "uv_bin_abs": uv_bin_abs,
    }
    setup_cfg_json = json.dumps(setup_cfg, ensure_ascii=True)
    command = (
        "command -v python3 >/dev/null 2>&1 || { echo 'python3 not found on remote' >&2; exit 2; }; "
        "export PATH=\"$HOME/.local/bin:$PATH\"; "
        f"if [ ! -x {shlex.quote(uv_bin_abs)} ] && ! command -v uv >/dev/null 2>&1; then "
        "command -v curl >/dev/null 2>&1 || { echo 'curl not found on remote' >&2; exit 4; }; "
        f"curl -LsSf {shlex.quote(UV_INSTALL_URL)} | sh; "
        "fi; "
        f"test -x {shlex.quote(uv_bin_abs)} || command -v uv >/dev/null 2>&1 || "
        "{ echo 'uv not found after install' >&2; exit 5; }; "
        f"test -f {shlex.quote(sd_scripts_abs + '/requirements.txt')} || "
        "{ echo 'requirements.txt not found in sd-scripts' >&2; exit 3; }; "
        f"SD_TRAIN_SETUP={shlex.quote(setup_cfg_json)} python3 - <<'PY'\n"
        "import json\n"
        "import os\n"
        "import subprocess\n"
        "import tempfile\n"
        "\n"
        "cfg = json.loads(__import__('os').environ['SD_TRAIN_SETUP'])\n"
        "setup_root_abs = cfg['setup_root_abs']\n"
        "sd_scripts_abs = cfg['sd_scripts_abs']\n"
        "venv_abs = cfg['venv_abs']\n"
        "uv_bin = cfg['uv_bin_abs']\n"
        "if not os.path.exists(uv_bin):\n"
        "    uv_bin = 'uv'\n"
        "venv_python = venv_abs + '/bin/python'\n"
        "base_req_source = os.path.join(sd_scripts_abs, 'requirements.txt')\n"
        "linux_req_source = os.path.join(sd_scripts_abs, 'requirements_linux.txt')\n"
        "\n"
        "def run(cmd, *, cwd=None):\n"
        "    subprocess.run(cmd, check=True, cwd=cwd)\n"
        "\n"
        "os.makedirs(setup_root_abs, exist_ok=True)\n"
        f"run([uv_bin, 'python', 'install', {REMOTE_TRAIN_PYTHON_VERSION!r}])\n"
        "if os.path.exists(venv_python):\n"
        "    print(f'Using existing virtual environment at: {venv_abs}')\n"
        "elif os.path.exists(venv_abs):\n"
        "    raise RuntimeError(\n"
        "        f'Existing virtual environment is incomplete at {venv_abs}; '\n"
        "        'remove it manually if you want to recreate it'\n"
        "    )\n"
        "else:\n"
        f"    run([uv_bin, 'venv', '--python', {REMOTE_TRAIN_PYTHON_VERSION!r}, venv_abs], cwd=setup_root_abs)\n"
        "run([uv_bin, 'pip', 'install', '--python', venv_python, '--upgrade', 'pip', 'setuptools', 'wheel'])\n"
        "source_req = linux_req_source if os.path.exists(linux_req_source) else base_req_source\n"
        "source_label = 'requirements_linux.txt' if source_req == linux_req_source else 'requirements.txt'\n"
        "print(f'Installing remote sd-scripts requirements from {source_label}: {source_req}')\n"
        "with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_linux:\n"
        "    linux_req_path = tmp_linux.name\n"
        "    tmp_linux.write(open(source_req, 'r', encoding='utf-8', errors='ignore').read())\n"
        "with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_base:\n"
        "    base_req_path = tmp_base.name\n"
        "    tmp_base.write(open(base_req_source, 'r', encoding='utf-8', errors='ignore').read())\n"
        "base_raw = open(base_req_path, 'r', encoding='utf-8', errors='ignore').read().splitlines()\n"
        "base_norm: list[str] = []\n"
        "for line in base_raw:\n"
        "    stripped = line.strip()\n"
        "    if stripped in ('-e .', '--editable .', '.', '-e ./', '--editable ./', './'):\n"
        "        base_norm.append(f'-e {sd_scripts_abs}')\n"
        "        continue\n"
        "    if './sd-scripts' in stripped:\n"
        "        if stripped.startswith('-e ') or stripped.startswith('--editable '):\n"
        "            base_norm.append(f'-e {sd_scripts_abs}')\n"
        "        else:\n"
        "            base_norm.append(sd_scripts_abs)\n"
        "        continue\n"
        "    base_norm.append(line)\n"
        "open(base_req_path, 'w', encoding='utf-8').write('\\n'.join(base_norm) + '\\n')\n"
        "raw_text = open(linux_req_path, 'r', encoding='utf-8', errors='ignore').read().splitlines()\n"
        "normalized: list[str] = []\n"
        "for line in raw_text:\n"
        "    stripped = line.strip()\n"
        "    if not stripped or stripped.startswith('#'):\n"
        "        normalized.append(line)\n"
        "        continue\n"
        "    if stripped == '-r requirements.txt':\n"
        "        normalized.append(f'-r {base_req_path}')\n"
        "        continue\n"
        "    if stripped in ('-e .', '--editable .', '.', '-e ./', '--editable ./', './'):\n"
        "        normalized.append(f'-e {sd_scripts_abs}')\n"
        "        continue\n"
        "    if './sd-scripts' in stripped:\n"
        "        if stripped.startswith('-e ') or stripped.startswith('--editable '):\n"
        "            normalized.append(f'-e {sd_scripts_abs}')\n"
        "        else:\n"
        "            normalized.append(sd_scripts_abs)\n"
        "        continue\n"
        "    normalized.append(line)\n"
        "open(linux_req_path, 'w', encoding='utf-8').write('\\n'.join(normalized) + '\\n')\n"
        "run([uv_bin, 'pip', 'install', '--python', venv_python, '--upgrade', '-r', linux_req_path])\n"
        "os.unlink(linux_req_path)\n"
        "os.unlink(base_req_path)\n"
        "PY"
    )
    result = await environment.run(command, stdout=sys.stdout, stderr=sys.stderr)
    if result.code != 0:
        raise RuntimeError(
            "Failed to prepare remote python venv and install requirements. "
            "Tried installing requirements from the remote sd-scripts checkout "
            "(requirements.txt and optional requirements_linux.txt)."
        )
    return venv_abs, sd_scripts_abs


def _hash_local_path(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    if path.is_dir():
        for child in sorted(path.rglob("*")):
            if child.is_dir():
                continue
            rel = child.relative_to(path).as_posix()
            stat = child.stat()
            hasher.update(rel.encode("utf-8"))
            hasher.update(str(stat.st_size).encode("utf-8"))
            hasher.update(str(stat.st_mtime_ns).encode("utf-8"))
        return hasher.hexdigest()

    raise ValueError(f"Unsupported local path type: {path}")


def _infer_key_kind(key: str) -> str:
    lowered = key.lower()
    if "prompt" in lowered:
        return "prompts"
    if "data" in lowered:
        return "datasets"
    if "vae" in lowered:
        return "vae"
    if "model" in lowered or "weight" in lowered:
        return "models"
    if "config" in lowered:
        return "configs"
    return "misc"


def _github_raw_url(repo: str, commit: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"


async def _ensure_remote_exists(environment: Environment, abs_path: str) -> bool:
    result = await environment.run(f"[ -e {shlex.quote(abs_path)} ]")
    return result.code == 0


async def _upload_local_ref(
    environment: Environment,
    key: str,
    local_path: Path,
    remote_root_sync: str,
    remote_home: str,
) -> MaterializedRef:
    content_hash = _hash_local_path(local_path)
    kind = _infer_key_kind(key)
    leaf = sanitize_component(local_path.name)
    remote_sync = f"{remote_root_sync}/artifacts/{kind}/{content_hash}/{leaf}"
    remote_abs = to_abs_remote(remote_sync, remote_home)

    if not await _ensure_remote_exists(environment, remote_abs):
        print(f"{Fore.CYAN}Uploading local reference: {local_path} -> {remote_sync}{Style.RESET_ALL}")
        await asyncio.to_thread(environment.sync_from_local, str(local_path), remote_sync)

    return MaterializedRef(sync_path=remote_sync, abs_path=remote_abs)


async def _materialize_hf_ref(
    environment: Environment,
    key: str,
    hf_raw: str,
    remote_root_sync: str,
    remote_home: str,
    auth: DownloadAuth,
) -> MaterializedRef:
    hf_ref = parse_hf_ref(hf_raw)
    mode = expected_hf_mode_for_key(key, has_subpath=hf_ref.subpath is not None)

    if mode == "file" and hf_ref.subpath is None:
        raise ValueError(f"HF ref for key '{key}' must include subpath with '::'. got: {hf_raw}")

    hash_seed = hashlib.sha256(hf_raw.encode("utf-8")).hexdigest()
    kind = _infer_key_kind(key)
    if mode == "dir":
        remote_sync = f"{remote_root_sync}/artifacts/{kind}/{hash_seed}/snapshot"
    else:
        filename = sanitize_component(Path(hf_ref.subpath or "payload").name)
        remote_sync = f"{remote_root_sync}/artifacts/{kind}/{hash_seed}/{filename}"
    remote_abs = to_abs_remote(remote_sync, remote_home)

    if await _ensure_remote_exists(environment, remote_abs):
        return MaterializedRef(sync_path=remote_sync, abs_path=remote_abs)

    cfg = {
        "repo_type": hf_ref.repo_type,
        "repo_id": hf_ref.repo_id,
        "revision": hf_ref.revision,
        "subpath": hf_ref.subpath,
        "mode": mode,
        "target": remote_abs,
        "hf_token": auth.hf_token.strip(),
        "venv_python": to_abs_remote("~/.sd-train/.venv/bin/python", remote_home),
    }
    cfg_json = json.dumps(cfg, ensure_ascii=True)
    command = (
        f"export HF_CFG={shlex.quote(cfg_json)}; "
        "HF_PYTHON=$(python3 - <<'PY'\n"
        "import json\n"
        "import os\n"
        "cfg = json.loads(os.environ['HF_CFG'])\n"
        "venv_python = cfg.get('venv_python', '')\n"
        "print(venv_python if venv_python and os.path.exists(venv_python) else 'python3')\n"
        "PY\n"
        "); "
        "\"$HF_PYTHON\" - <<'PY'\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "from importlib.metadata import PackageNotFoundError, version\n"
        "cfg = json.loads(os.environ['HF_CFG'])\n"
        "target = pathlib.Path(cfg['target'])\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "repo_type = cfg['repo_type']\n"
        "repo_id = cfg['repo_id']\n"
        "revision = cfg['revision']\n"
        "subpath = cfg['subpath']\n"
        "mode = cfg['mode']\n"
        "token = (cfg.get('hf_token') or '').strip() or None\n"
        "venv_python = (cfg.get('venv_python') or '').strip()\n"
        "if token:\n"
        "    os.environ['HF_TOKEN'] = token\n"
        "    os.environ['HUGGINGFACE_HUB_TOKEN'] = token\n"
        f"expected_version = {PINNED_HUGGINGFACE_HUB_VERSION!r}\n"
        f"pinned_requirement = {'huggingface_hub==' + PINNED_HUGGINGFACE_HUB_VERSION!r}\n"
        "install_python = venv_python if venv_python and os.path.exists(venv_python) else sys.executable\n"
        "installed_version = None\n"
        "try:\n"
        "    installed_version = version('huggingface_hub')\n"
        "except PackageNotFoundError:\n"
        "    installed_version = None\n"
        "if installed_version != expected_version:\n"
        "    subprocess.run([\n"
        "        install_python,\n"
        "        '-m',\n"
        "        'pip',\n"
        "        'install',\n"
        "        '--upgrade',\n"
        "        pinned_requirement,\n"
        "    ], check=True)\n"
        "from huggingface_hub import hf_hub_download, snapshot_download\n"
        "if mode == 'dir':\n"
        "    snapshot_download(repo_id=repo_id, repo_type=repo_type, revision=revision, local_dir=str(target), local_dir_use_symlinks=False, token=token)\n"
        "else:\n"
        "    if not subpath:\n"
        "        raise RuntimeError('missing subpath for file mode')\n"
        "    downloaded = pathlib.Path(hf_hub_download(repo_id=repo_id, repo_type=repo_type, revision=revision, filename=subpath, local_dir=str(target.parent), local_dir_use_symlinks=False, token=token))\n"
        "    if downloaded != target:\n"
        "        downloaded.replace(target)\n"
        "PY"
    )

    print(f"{Fore.CYAN}Downloading HF reference on remote: {hf_raw} -> {remote_abs}{Style.RESET_ALL}")
    result = await environment.run(command, stdout=sys.stdout, stderr=sys.stderr)
    if result.code != 0:
        raise RuntimeError(f"Failed to materialize HF reference '{hf_raw}' for key '{key}'")

    return MaterializedRef(sync_path=remote_sync, abs_path=remote_abs)


async def _materialize_civitai_ref(
    environment: Environment,
    key: str,
    civitai_raw: str,
    remote_root_sync: str,
    remote_home: str,
    auth: DownloadAuth,
) -> MaterializedRef:
    from sd_train.domain.refs import _resolve_civitai_download_id

    model_id, resolved_filename, _resolution = _resolve_civitai_download_id(civitai_raw, auth)
    mode = expected_local_mode_for_key(key)
    if mode == "dir":
        raise ValueError(f"CivitAI ref for key '{key}' must resolve to a file. got: {civitai_raw}")

    hash_seed = hashlib.sha256(civitai_raw.encode("utf-8")).hexdigest()
    kind = _infer_key_kind(key)
    filename = resolved_filename or f"civitai-{model_id}.safetensors"
    remote_sync = f"{remote_root_sync}/artifacts/{kind}/{hash_seed}/{sanitize_component(filename)}"
    remote_abs = to_abs_remote(remote_sync, remote_home)
    if await _ensure_remote_exists(environment, remote_abs):
        return MaterializedRef(sync_path=remote_sync, abs_path=remote_abs)

    token = auth.civitai_api_key.strip()

    cfg = {
        "model_id": model_id,
        "target": remote_abs,
        "civitai_api_key": token,
        "downloader_url": _github_raw_url(
            "ashleykleynhans/civitai-downloader",
            PINNED_CIVITAI_DOWNLOADER_COMMIT,
            "download.py",
        ),
    }
    cfg_json = json.dumps(cfg, ensure_ascii=True)
    command = (
        f"CIVITAI_CFG={shlex.quote(cfg_json)} python3 - <<'PY'\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import subprocess\n"
        "import sys\n"
        "import urllib.request\n"
        "cfg = json.loads(os.environ['CIVITAI_CFG'])\n"
        "target = pathlib.Path(cfg['target'])\n"
        "target.parent.mkdir(parents=True, exist_ok=True)\n"
        "script = pathlib.Path.home() / '.sd-train' / 'bin' / 'civitai-downloader.py'\n"
        "script.parent.mkdir(parents=True, exist_ok=True)\n"
        "urllib.request.urlretrieve(cfg['downloader_url'], str(script))\n"
        "script.chmod(0o755)\n"
        "before = {p for p in target.parent.glob('*') if p.is_file()}\n"
        "env = os.environ.copy()\n"
        "env['CIVITAI_TOKEN'] = cfg['civitai_api_key']\n"
        "subprocess.run([sys.executable, str(script), str(cfg['model_id']), str(target.parent)], env=env, check=True)\n"
        "after = [p for p in target.parent.glob('*') if p.is_file()]\n"
        "new_files = [p for p in after if p not in before]\n"
        "chosen = None\n"
        "if new_files:\n"
        "    chosen = max(new_files, key=lambda p: p.stat().st_mtime)\n"
        "else:\n"
        "    candidates = sorted(after, key=lambda p: p.stat().st_mtime, reverse=True)\n"
        "    if candidates:\n"
        "        chosen = candidates[0]\n"
        "if chosen is None:\n"
        "    print('CivitAI download completed but no file found', file=sys.stderr)\n"
        "    sys.exit(4)\n"
        "if chosen != target:\n"
        "    chosen.replace(target)\n"
        "PY"
    )

    print(f"{Fore.CYAN}Downloading CivitAI reference on remote: {civitai_raw} -> {remote_abs}{Style.RESET_ALL}")
    result = await environment.run(command, stdout=sys.stdout, stderr=sys.stderr)
    if result.code != 0:
        raise RuntimeError(f"Failed to materialize CivitAI reference '{civitai_raw}' for key '{key}'")

    return MaterializedRef(sync_path=remote_sync, abs_path=remote_abs)


async def _resolve_and_patch_train_toml(
    environment: Environment,
    train_config_path: Path,
    remote_run_sync: str,
    remote_home: str,
    auth: DownloadAuth,
) -> tuple[str, str, str]:
    data = toml.load(train_config_path)
    if not isinstance(data, dict):
        raise ValueError("train.toml must decode to a table")

    remote_root_sync = "~/.sd-train"
    remote_output_sync = f"{remote_run_sync}/output"
    remote_logging_sync = f"{remote_run_sync}/logging"
    remote_output_abs = to_abs_remote(remote_output_sync, remote_home)
    remote_logging_abs = to_abs_remote(remote_logging_sync, remote_home)

    for key, value in list(data.items()):
        if not isinstance(value, str):
            continue

        if key in FORCE_REMOTE_OUTPUT_KEYS:
            data[key] = remote_output_abs if key == "output_dir" else remote_logging_abs
            continue

        if not is_path_value_key(key) and not looks_like_explicit_remote_ref(value):
            continue

        raw_value = value.strip()
        if not raw_value:
            continue

        candidate = Path(raw_value).expanduser()
        if not candidate.is_absolute():
            candidate = (train_config_path.parent / candidate).resolve()

        if candidate.exists():
            mode = expected_local_mode_for_key(key)
            if mode == "dir" and not candidate.is_dir():
                raise ValueError(f"'{key}' expects directory: {candidate}")
            if mode == "file" and not candidate.is_file():
                raise ValueError(f"'{key}' expects file: {candidate}")
            materialized = await _upload_local_ref(environment, key, candidate, remote_root_sync, remote_home)
            data[key] = materialized.abs_path
            continue

        if not looks_like_hf_ref(raw_value) and not looks_like_civitai_ref(raw_value):
            raise FileNotFoundError(f"Local path for key '{key}' does not exist: {candidate}")

        if looks_like_hf_ref(raw_value):
            materialized = await _materialize_hf_ref(environment, key, raw_value, remote_root_sync, remote_home, auth)
        else:
            materialized = await _materialize_civitai_ref(environment, key, raw_value, remote_root_sync, remote_home, auth)
        data[key] = materialized.abs_path

    if "output_dir" not in data:
        data["output_dir"] = remote_output_abs
    if "logging_dir" not in data:
        data["logging_dir"] = remote_logging_abs

    rendered = toml.dumps(data)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False, encoding="utf-8") as temp:
        temp.write(rendered)
        temp_path = Path(temp.name)

    remote_train_sync = f"{remote_run_sync}/train.toml"
    remote_train_abs = to_abs_remote(remote_train_sync, remote_home)
    try:
        await asyncio.to_thread(environment.sync_from_local, str(temp_path), remote_train_sync)
    finally:
        temp_path.unlink(missing_ok=True)

    return remote_train_sync, remote_train_abs, remote_output_sync


async def _list_stable_remote_files(
    environment: Environment,
    remote_output_abs: str,
    stability_seconds: int,
) -> list[tuple[str, float, int]]:
    script = (
        f"OUT={shlex.quote(remote_output_abs)} STABILITY={int(stability_seconds)} python3 - <<'PY'\n"
        "import os\n"
        "import sys\n"
        "import time\n"
        "root = os.environ['OUT']\n"
        "stability = int(os.environ['STABILITY'])\n"
        "now = time.time()\n"
        "if not os.path.isdir(root):\n"
        "    sys.exit(0)\n"
        "for base, _dirs, files in os.walk(root):\n"
        "    for name in files:\n"
        "        if name.endswith(('.tmp', '.part', '.lock')):\n"
        "            continue\n"
        "        path = os.path.join(base, name)\n"
        "        try:\n"
        "            st = os.stat(path)\n"
        "        except OSError:\n"
        "            continue\n"
        "        if now - st.st_mtime < stability:\n"
        "            continue\n"
        "        print(f'{path}\t{st.st_mtime}\t{st.st_size}')\n"
        "PY"
    )
    result = await environment.run(script)
    if result.code != 0:
        return []

    items: list[tuple[str, float, int]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        path = parts[0]
        try:
            mtime = float(parts[1])
            size = int(parts[2])
        except ValueError:
            continue
        if Path(path).name.endswith(TEMP_FILE_SUFFIXES):
            continue
        items.append((path, mtime, size))
    return items


async def periodic_output_sync(
    environment: Environment,
    remote_output_abs: str,
    local_output_dir: Path,
    remote_home: str,
    stop_event: asyncio.Event,
    interval_seconds: int,
    stability_seconds: int,
) -> None:
    seen: dict[str, tuple[float, int]] = {}
    remote_output_sync = to_sync_remote(remote_output_abs, remote_home)
    while not stop_event.is_set():
        stable = await _list_stable_remote_files(environment, remote_output_abs, stability_seconds)
        pending_remote_files: list[tuple[str, float, int, str]] = []
        for remote_abs, mtime, size in stable:
            previous = seen.get(remote_abs)
            if previous is not None and previous == (mtime, size):
                continue

            rel = os.path.relpath(remote_abs, remote_output_abs)
            if rel.startswith(".."):
                continue
            pending_remote_files.append((remote_abs, mtime, size, rel))

        if pending_remote_files:
            local_output_dir.mkdir(parents=True, exist_ok=True)
            pending_rel_paths = [item[3] for item in pending_remote_files]
            try:
                await asyncio.to_thread(
                    environment.sync_to_local,
                    remote_output_sync,
                    str(local_output_dir),
                    pending_rel_paths,
                )
                for remote_abs, mtime, size, _rel in pending_remote_files:
                    seen[remote_abs] = (mtime, size)
            except Exception as exc:
                print(
                    f"{Fore.YELLOW}Periodic sync skipped for {remote_output_sync}: {exc}{Style.RESET_ALL}"
                )

        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_seconds)
        except asyncio.TimeoutError:
            continue


async def final_output_sync(
    environment: Environment,
    remote_output_sync: str,
    remote_output_abs: str,
    local_output_dir: Path,
) -> None:
    exists = await environment.run(f"[ -d {shlex.quote(remote_output_abs)} ]")
    if exists.code != 0:
        print(
            f"{Fore.YELLOW}Remote output directory not found. Skip final sync: {remote_output_abs}{Style.RESET_ALL}"
        )
        return
    local_output_dir.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(environment.sync_to_local, remote_output_sync, str(local_output_dir))


async def _ensure_optional_network_modules(
    environment: Environment,
    remote_train_abs: str,
    remote_venv_abs: str,
    remote_sd_scripts_abs: str,
    remote_uv_bin_abs: str,
) -> None:
    cfg = {
        "train_config": remote_train_abs,
        "venv_python": remote_venv_abs + "/bin/python",
        "uv_bin": remote_uv_bin_abs,
        "sd_scripts_abs": remote_sd_scripts_abs,
    }
    cfg_json = json.dumps(cfg, ensure_ascii=True)
    command = (
        f"SD_TRAIN_OPTIONAL={shlex.quote(cfg_json)} python3 - <<'PY'\n"
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import re\n"
        "import subprocess\n"
        "\n"
        "cfg = json.loads(__import__('os').environ['SD_TRAIN_OPTIONAL'])\n"
        "train_config = cfg['train_config']\n"
        "venv_python = cfg['venv_python']\n"
        "uv_bin = cfg['uv_bin']\n"
        "if not os.path.exists(uv_bin):\n"
        "    uv_bin = 'uv'\n"
        "sd_scripts_abs = cfg['sd_scripts_abs']\n"
        "sd_scripts_path = pathlib.Path(sd_scripts_abs)\n"
        "\n"
        "with open(train_config, 'rb') as f:\n"
        "    try:\n"
        "        import tomllib\n"
        "        data = tomllib.load(f)\n"
        "    except Exception:\n"
        "        data = {}\n"
        "network_module = str(data.get('network_module', '')).strip().lower()\n"
        "optimizer_type = str(data.get('optimizer_type', '')).strip().lower()\n"
        "if network_module:\n"
        "    print(f'Checking network module dependencies for: {network_module}')\n"
        "if optimizer_type:\n"
        "    print(f'Checking optimizer dependencies for: {optimizer_type}')\n"
        "\n"
        "known_packages = {\n"
        "    'lycoris': 'lycoris_lora',\n"
        "    'lycoris_lora': 'lycoris_lora',\n"
        "    'xformers': 'xformers',\n"
        "    'bitsandbytes': 'bitsandbytes',\n"
        "    'dadaptation': 'dadaptation',\n"
        "    'prodigyopt': 'prodigyopt',\n"
        "    'schedulefree': 'schedulefree',\n"
        "}\n"
        "local_module_roots = {'networks'}\n"
        "import_probe = (\n"
        "    'import os\\n'\n"
        "    'import sys\\n'\n"
        "    'import importlib\\n'\n"
        "    'import traceback\\n'\n"
        "    f'sd_scripts_abs={sd_scripts_abs!r}\\n'\n"
        "    f'module={network_module!r}\\n'\n"
        "    'sys.path.insert(0, sd_scripts_abs)\\n'\n"
        "    'os.chdir(sd_scripts_abs)\\n'\n"
        "    'try:\\n'\n"
        "    '    importlib.import_module(module)\\n'\n"
        "    'except Exception:\\n'\n"
        "    '    traceback.print_exc()\\n'\n"
        "    '    raise\\n'\n"
        ")\n"
        "\n"
        "def run_probe():\n"
        "    env = os.environ.copy()\n"
        "    existing = env.get('PYTHONPATH', '').strip()\n"
        "    env['PYTHONPATH'] = sd_scripts_abs if not existing else sd_scripts_abs + os.pathsep + existing\n"
        "    return subprocess.run(\n"
        "        [venv_python, '-c', import_probe],\n"
        "        capture_output=True,\n"
        "        text=True,\n"
        "        cwd=sd_scripts_abs,\n"
        "        env=env,\n"
        "    )\n"
        "\n"
        "if not network_module:\n"
        "    raise SystemExit(0)\n"
        "\n"
        "result = run_probe()\n"
        "if result.returncode == 0:\n"
        "    raise SystemExit(0)\n"
        "\n"
        "output = (result.stderr or '') + '\\n' + (result.stdout or '')\n"
        "missing_matches = re.findall(r\"No module named '([^']+)'\", output)\n"
        "if not missing_matches:\n"
        "    raise RuntimeError(\n"
        "        'Failed importing network_module, but no missing module could be detected.\\n'\n"
        "        + output.strip()\n"
        "    )\n"
        "\n"
        "installed_any = False\n"
        "for missing in missing_matches:\n"
        "    root = missing.split('.', 1)[0]\n"
        "    if root in local_module_roots:\n"
        "        candidate = sd_scripts_path / root\n"
        "        if candidate.exists():\n"
        "            detail = output.strip()\n"
        "            raise RuntimeError(\n"
        "                f'Module {missing} should come from sd-scripts at {candidate}, '\n"
        "                'but it is still not importable. Check network_module spelling and pinned sd-scripts commit.\\n'\n"
        "                + detail\n"
        "            )\n"
        "        raise RuntimeError(\n"
        "            f'Module {missing} is not present in pinned sd-scripts checkout under {candidate}. '\n"
        "            'Check network_module spelling or update the pinned sd-scripts commit.'\n"
        "        )\n"
        "    package = known_packages.get(root, root)\n"
        "    print(f'Missing python module: {missing}. Trying uv pip install {package}...')\n"
        "    pip_result = subprocess.run([\n"
        "        uv_bin,\n"
        "        'pip',\n"
        "        'install',\n"
        "        '--python',\n"
        "        venv_python,\n"
        "        package,\n"
        "    ], check=False)\n"
        "    if pip_result.returncode == 0:\n"
        "        installed_any = True\n"
        "\n"
        "if not installed_any:\n"
        "    raise RuntimeError('Unable to install any missing packages for network_module import.')\n"
        "\n"
        "verify = run_probe()\n"
        "if verify.returncode != 0:\n"
        "    verify_out = (verify.stderr or '') + '\\n' + (verify.stdout or '')\n"
        "    raise RuntimeError('network_module import still failing after auto-install.\\n' + verify_out.strip())\n"
        "PY"
    )
    result = await environment.run(command, stdout=sys.stdout, stderr=sys.stderr)
    if result.code != 0:
        detail = (result.stderr or "").strip() or (result.stdout or "").strip()
        raise RuntimeError(
            "Failed to ensure optional training modules from train config "
            "(e.g., lycoris)."
            + (f"\n{detail}" if detail else "")
        )


async def run_training_session(environment: Environment, selection: Any, auth: DownloadAuth) -> None:
    if isinstance(environment, SSH):
        print(f"{Fore.CYAN}Connecting SSH: {environment.connection_target()}{Style.RESET_ALL}")

    async with environment:
        server = await environment.start_file_server()
        if server is None:
            raise RuntimeError("rclone file server unavailable")

        print(
            f"{Fore.CYAN}rclone serve ready: {server.url} (user={server.username}, pass={_mask_secret(server.password)}){Style.RESET_ALL}"
        )
        _set_transfer_progress(environment, True)
        await _run_commands(environment, BASE_COMMANDS)

        home_result = await environment.run('printf "%s" "$HOME"')
        if home_result.code != 0 or not home_result.stdout.strip():
            raise RuntimeError("Failed to resolve remote HOME path")
        remote_home = home_result.stdout.strip()
        print(f"{Fore.CYAN}Preparing remote sd-scripts repository...{Style.RESET_ALL}")
        await _ensure_remote_sd_scripts(environment, remote_home)
        print(f"{Fore.CYAN}Preparing remote python venv and requirements...{Style.RESET_ALL}")
        remote_venv_abs, remote_sd_scripts_abs = await _prepare_remote_python_venv(environment, remote_home)
        remote_uv_bin_abs = to_abs_remote("~/.local/bin/uv", remote_home)

        run_id = f"run-{int(time.time())}-{os.getpid()}"
        remote_run_sync = f"~/.sd-train/runs/{run_id}"
        remote_run_abs = to_abs_remote(remote_run_sync, remote_home)
        mkdir_cmd = (
            f"mkdir -p {shlex.quote(remote_run_abs)} "
            f"{shlex.quote(to_abs_remote('~/.sd-train/artifacts', remote_home))}"
        )
        await _run_commands(environment, [mkdir_cmd])

        train_config_path = Path(selection.train_config_path).expanduser().resolve()
        if not train_config_path.is_file():
            raise ValueError(f"Train config file not found: {train_config_path}")

        remote_train_sync, remote_train_abs, remote_output_sync = await _resolve_and_patch_train_toml(
            environment,
            train_config_path,
            remote_run_sync,
            remote_home,
            auth,
        )
        remote_output_abs = to_abs_remote(remote_output_sync, remote_home)
        await _ensure_optional_network_modules(
            environment,
            remote_train_abs,
            remote_venv_abs,
            remote_sd_scripts_abs,
            remote_uv_bin_abs,
        )

        script = _normalize_script_path(selection.train_script)
        remote_script_abs = to_abs_remote(f"~/.sd-train/sd-scripts/{script}", remote_home)
        script_exists = await environment.run(f"[ -f {shlex.quote(remote_script_abs)} ]")
        if script_exists.code != 0:
            raise RuntimeError(f"Selected train script does not exist on remote: {script}")

        local_output_dir = Path("outputs") / run_id
        stop_event = asyncio.Event()
        sync_task = asyncio.create_task(
            periodic_output_sync(
                environment,
                remote_output_abs=remote_output_abs,
                local_output_dir=local_output_dir,
                remote_home=remote_home,
                stop_event=stop_event,
                interval_seconds=SYNC_INTERVAL_SECONDS,
                stability_seconds=SYNC_STABILITY_SECONDS,
            )
        )

        train_command = (
            f"cd {shlex.quote(remote_sd_scripts_abs)} && "
            f"{shlex.quote(remote_venv_abs + '/bin/accelerate')} launch "
            f"{shlex.quote(script)} --config_file {shlex.quote(remote_train_abs)}"
        )
        print(f"{Fore.CYAN}Starting training script: {script}{Style.RESET_ALL}")

        try:
            result = await environment.run(train_command, stdout=sys.stdout, stderr=sys.stderr)
            if result.code != 0:
                raise RuntimeError(f"Training command failed with code {result.code}")
        finally:
            stop_event.set()
            await sync_task
            print(f"{Fore.CYAN}Running final output sync...{Style.RESET_ALL}")
            try:
                await final_output_sync(environment, remote_output_sync, remote_output_abs, local_output_dir)
            except Exception as exc:
                print(f"{Fore.YELLOW}Final output sync skipped due to error: {exc}{Style.RESET_ALL}")
            finally:
                _set_transfer_progress(environment, False)
