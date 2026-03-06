from dataclasses import dataclass
from pathlib import Path
import json

import toml

from sd_train.domain.path_rules import (
    FORCE_REMOTE_OUTPUT_KEYS,
    expected_hf_mode_for_key,
    expected_local_mode_for_key,
    is_path_value_key,
    resolve_path_from_config,
)
from sd_train.domain.refs import (
    looks_like_civitai_ref,
    looks_like_explicit_remote_ref,
    looks_like_hf_ref,
    parse_civitai_ref,
    parse_hf_ref,
)

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


@dataclass
class PreflightReport:
    train_config_path: Path
    train_script: str
    dataset_mode: str
    dataset_roots: list[Path]
    subset_dirs: list[Path]
    image_count: int
    caption_count: int
    sample_prompts_count: int | None
    max_train_steps: int | None
    train_batch_size: int | None
    save_every_n_steps: int | None
    sample_every_n_steps: int | None
    output_name: str | None


def _count_images_under(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            total += 1
    return total


def _count_captions_for_images(path: Path, caption_extension: str) -> int:
    total = 0
    for image in path.rglob("*"):
        if not image.is_file() or image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        caption = image.with_suffix(caption_extension)
        if caption.is_file():
            total += 1
    return total


def _load_local_dataset_subsets(
    dataset_config_path: Path, train_config_path: Path
) -> tuple[list[Path], list[Path]]:
    suffix = dataset_config_path.suffix.lower()
    if suffix == ".json":
        payload = json.loads(dataset_config_path.read_text(encoding="utf-8"))
    else:
        payload = toml.load(dataset_config_path)
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid dataset_config format: {dataset_config_path}")

    dataset_roots: list[Path] = []
    subset_dirs: list[Path] = []
    datasets = payload.get("datasets")
    if not isinstance(datasets, list):
        raise ValueError("dataset_config must include a 'datasets' list")

    for dataset in datasets:
        if not isinstance(dataset, dict):
            continue
        subsets = dataset.get("subsets")
        if not isinstance(subsets, list):
            continue
        for subset in subsets:
            if not isinstance(subset, dict):
                continue
            image_dir_raw = subset.get("image_dir")
            if isinstance(image_dir_raw, str) and image_dir_raw.strip():
                image_dir = resolve_path_from_config(image_dir_raw, dataset_config_path)
                if not image_dir.is_dir():
                    raise ValueError(f"subset.image_dir is not a directory: {image_dir}")
                dataset_roots.append(image_dir)
                subset_dirs.append(image_dir)

            metadata_file_raw = subset.get("metadata_file")
            if isinstance(metadata_file_raw, str) and metadata_file_raw.strip():
                metadata_file = resolve_path_from_config(metadata_file_raw, dataset_config_path)
                if not metadata_file.is_file():
                    raise ValueError(f"subset.metadata_file is not a file: {metadata_file}")
                dataset_roots.append(metadata_file.parent)

    uniq_roots = sorted(set(dataset_roots), key=lambda p: str(p))
    uniq_subsets = sorted(set(subset_dirs), key=lambda p: str(p))
    return uniq_roots, uniq_subsets


def validate_train_config_lightweight(train_config_path: Path, train_script: str) -> PreflightReport:
    data = toml.load(train_config_path)
    if not isinstance(data, dict):
        raise ValueError("train.toml must decode to a table")

    if "pretrained_model_name_or_path" not in data:
        raise ValueError("Missing required key: pretrained_model_name_or_path")

    dataset_config = data.get("dataset_config")
    if dataset_config is None and "train_data_dir" not in data:
        raise ValueError("Missing required key: train_data_dir (or provide dataset_config)")

    dataset_mode = "train_data_dir"
    dataset_roots: list[Path] = []
    subset_dirs: list[Path] = []
    caption_extension = data.get("caption_extension")
    if not isinstance(caption_extension, str) or not caption_extension.strip():
        caption_extension = ".txt"

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
            mode = expected_local_mode_for_key(key)
            if mode == "dir" and not candidate.is_dir():
                raise ValueError(f"'{key}' expects directory: {candidate}")
            if mode == "file" and not candidate.is_file():
                raise ValueError(f"'{key}' expects file: {candidate}")
            continue

        if not looks_like_hf_ref(raw_value) and not looks_like_civitai_ref(raw_value):
            raise FileNotFoundError(f"Local path for key '{key}' does not exist: {candidate}")
        if looks_like_hf_ref(raw_value):
            hf_ref = parse_hf_ref(raw_value)
            expected = expected_hf_mode_for_key(key, has_subpath=hf_ref.subpath is not None)
            if expected == "file" and hf_ref.subpath is None:
                raise ValueError(
                    f"HF ref for key '{key}' must include subpath with '::'. got: {raw_value}"
                )
        else:
            parse_civitai_ref(raw_value)

    if isinstance(dataset_config, str) and dataset_config.strip():
        dataset_config_path = resolve_path_from_config(dataset_config, train_config_path)
        if dataset_config_path.exists():
            if not dataset_config_path.is_file():
                raise ValueError(f"'dataset_config' expects file path: {dataset_config_path}")
            dataset_mode = "dataset_config"
            dataset_roots, subset_dirs = _load_local_dataset_subsets(dataset_config_path, train_config_path)
        else:
            parse_hf_ref(dataset_config)
            dataset_mode = "dataset_config(hf)"
    else:
        train_data_raw = data.get("train_data_dir")
        if not isinstance(train_data_raw, str) or not train_data_raw.strip():
            raise ValueError("Missing required key: train_data_dir")
        train_data_dir = resolve_path_from_config(train_data_raw, train_config_path)
        if not train_data_dir.is_dir():
            raise ValueError(f"'train_data_dir' is not a directory: {train_data_dir}")
        dataset_roots = [train_data_dir]

        for child in sorted(train_data_dir.iterdir()):
            if not child.is_dir():
                continue
            if _count_images_under(child) > 0:
                subset_dirs.append(child)
        if not subset_dirs:
            raise ValueError(
                "No subset directory with images found under train_data_dir. "
                "Expected structure: train_data_dir/<subset_dir>/images..."
            )

    image_count = 0
    caption_count = 0
    scan_targets = subset_dirs if subset_dirs else dataset_roots
    for target in scan_targets:
        image_count += _count_images_under(target)
        caption_count += _count_captions_for_images(target, caption_extension)
    if scan_targets and image_count == 0:
        raise ValueError("Dataset contains no image files.")

    sample_prompts_count: int | None = None
    sample_prompts_raw = data.get("sample_prompts")
    if isinstance(sample_prompts_raw, str) and sample_prompts_raw.strip():
        sample_prompts_path = resolve_path_from_config(sample_prompts_raw, train_config_path)
        if sample_prompts_path.is_file():
            sample_prompts_count = len(
                [
                    line
                    for line in sample_prompts_path.read_text(
                        encoding="utf-8", errors="ignore"
                    ).splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
            )

    return PreflightReport(
        train_config_path=train_config_path,
        train_script=train_script,
        dataset_mode=dataset_mode,
        dataset_roots=dataset_roots,
        subset_dirs=subset_dirs,
        image_count=image_count,
        caption_count=caption_count,
        sample_prompts_count=sample_prompts_count,
        max_train_steps=data.get("max_train_steps") if isinstance(data.get("max_train_steps"), int) else None,
        train_batch_size=data.get("train_batch_size") if isinstance(data.get("train_batch_size"), int) else None,
        save_every_n_steps=data.get("save_every_n_steps") if isinstance(data.get("save_every_n_steps"), int) else None,
        sample_every_n_steps=data.get("sample_every_n_steps") if isinstance(data.get("sample_every_n_steps"), int) else None,
        output_name=data.get("output_name") if isinstance(data.get("output_name"), str) else None,
    )
