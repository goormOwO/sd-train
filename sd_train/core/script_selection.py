import ast
import json
import time
import warnings
from pathlib import Path

import requests

SCRIPT_SCAN_CACHE_TTL_SECONDS = 24 * 60 * 60
SCRIPT_FALLBACKS = ["train_network.py", "sd_train_network.py"]
SCRIPT_SCAN_CACHE_FILE = Path(".cache") / "train-script-scan.json"

_SCRIPT_SCAN_CACHE: tuple[float, list[str], str | None] | None = None


def _looks_like_main_entry(content: str) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(content)
    except SyntaxError:
        return False

    has_setup_parser = False
    has_main_guard = False

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "setup_parser":
            has_setup_parser = True
        if isinstance(node, ast.If):
            test = node.test
            if (
                isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and isinstance(test.ops[0], ast.Eq)
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], ast.Constant)
                and test.comparators[0].value == "__main__"
            ):
                has_main_guard = True

    return has_setup_parser and has_main_guard


def scan_train_scripts() -> tuple[list[str], str | None]:
    now = time.time()
    global _SCRIPT_SCAN_CACHE
    if _SCRIPT_SCAN_CACHE is not None:
        cached_at, scripts, error = _SCRIPT_SCAN_CACHE
        if now - cached_at < SCRIPT_SCAN_CACHE_TTL_SECONDS:
            return scripts, error

    if SCRIPT_SCAN_CACHE_FILE.exists():
        try:
            payload = json.loads(SCRIPT_SCAN_CACHE_FILE.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                cached_at_raw = payload.get("cached_at", 0)
                scripts_raw = payload.get("scripts", [])
                error_raw = payload.get("error")
                if isinstance(cached_at_raw, (int, float)) and isinstance(scripts_raw, list):
                    scripts = [s for s in scripts_raw if isinstance(s, str)]
                    error = error_raw if isinstance(error_raw, str) else None
                    if scripts and now - float(cached_at_raw) < SCRIPT_SCAN_CACHE_TTL_SECONDS:
                        _SCRIPT_SCAN_CACHE = (float(cached_at_raw), scripts, error)
                        return scripts, error
        except Exception:
            pass

    url = "https://api.github.com/repos/kohya-ss/sd-scripts/git/trees/main?recursive=1"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        payload = response.json()
        tree = payload.get("tree", [])
        if not isinstance(tree, list):
            raise RuntimeError("Invalid GitHub tree response")

        candidates: list[str] = []
        for node in tree:
            if not isinstance(node, dict):
                continue
            path = node.get("path")
            node_type = node.get("type")
            if not isinstance(path, str) or node_type != "blob":
                continue
            base_name = path.rsplit("/", 1)[-1]
            if not base_name.endswith(".py"):
                continue
            if "train" not in base_name.lower():
                continue
            candidates.append(path)

        scripts: list[str] = []
        for path in sorted(candidates):
            raw_url = "https://raw.githubusercontent.com/kohya-ss/sd-scripts/main/" + path
            raw_response = requests.get(raw_url, timeout=15)
            if raw_response.status_code != 200:
                continue
            if _looks_like_main_entry(raw_response.text):
                scripts.append(path)

        if not scripts:
            raise RuntimeError("No train scripts found in GitHub scan")

        _SCRIPT_SCAN_CACHE = (now, scripts, None)
        SCRIPT_SCAN_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCRIPT_SCAN_CACHE_FILE.write_text(
            json.dumps({"cached_at": now, "scripts": scripts, "error": None}, ensure_ascii=True),
            encoding="utf-8",
        )
        return scripts, None
    except Exception as exc:
        message = f"GitHub script scan failed: {exc}"
        fallback = SCRIPT_FALLBACKS.copy()
        _SCRIPT_SCAN_CACHE = (now, fallback, message)
        SCRIPT_SCAN_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        SCRIPT_SCAN_CACHE_FILE.write_text(
            json.dumps({"cached_at": now, "scripts": fallback, "error": message}, ensure_ascii=True),
            encoding="utf-8",
        )
        return fallback, message


def normalize_script_path(script: str) -> str:
    value = script.strip()
    if not value:
        raise ValueError("Train script is empty")
    if value.startswith("/"):
        raise ValueError("Train script must be a relative path inside sd-scripts")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("Train script has invalid path segments")
    return value
