import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")


@dataclass
class TaggerModelConfig:
    model: str = "SmilingWolf/wd-vit-tagger-v3"
    threshold: float = 0.4
    batch: int = 1


@dataclass
class TagStatsSnapshot:
    total_images: int
    captioned_images: int
    tags: list[tuple[str, int]]


@dataclass
class TaggerRunSummary:
    command: str
    dataset_dir: str
    processed_images: int
    changed_captions: int
    failed_images: int
    duration_seconds: float
    failed_paths: list[str]
    message: str = ""


class Caption:
    def __init__(self, tags: list[str]):
        self._tags = [tag.strip() for tag in tags if tag.strip()]

    @staticmethod
    def format(tag: str) -> str:
        return tag.replace("_", " ").replace("(", r"\(").replace(")", r"\)").strip()

    @staticmethod
    def _caption_path(image_path: Path) -> Path:
        return image_path.with_suffix(".txt")

    @staticmethod
    def from_image(image_path: Path) -> "Caption":
        path = Caption._caption_path(image_path)
        if not path.exists():
            return Caption([])
        return Caption(path.read_text(encoding="utf-8").split(","))

    @staticmethod
    def from_tags(tags: Iterable[str]) -> "Caption":
        return Caption([tag.strip() for tag in tags])

    def tags(self) -> list[str]:
        return list(self._tags)

    def add(self, tag: str) -> bool:
        value = tag.strip()
        if not value or value in self._tags:
            return False
        self._tags.append(value)
        return True

    def remove(self, tag: str) -> bool:
        value = tag.strip()
        if value not in self._tags:
            return False
        self._tags.remove(value)
        return True

    def front(self, tag: str) -> bool:
        value = tag.strip()
        if value not in self._tags:
            return False
        self._tags.remove(value)
        self._tags.insert(0, value)
        return True

    def shuffle(self) -> None:
        random.shuffle(self._tags)

    def save_for_image(self, image_path: Path) -> None:
        Caption._caption_path(image_path).write_text(
            ", ".join(self._tags), encoding="utf-8"
        )


def parse_tag_input(tags: str) -> list[str]:
    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def find_images(directory: str) -> list[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        return []

    images: list[Path] = []
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.lower().endswith(IMAGE_SUFFIXES):
                images.append(Path(base) / name)
    images.sort()
    return images


def _build_summary(
    *,
    command: str,
    dataset_dir: str,
    processed_images: int,
    changed_captions: int,
    failed_paths: list[str],
    started: float,
    message: str = "",
) -> TaggerRunSummary:
    return TaggerRunSummary(
        command=command,
        dataset_dir=dataset_dir,
        processed_images=processed_images,
        changed_captions=changed_captions,
        failed_images=len(failed_paths),
        failed_paths=failed_paths,
        duration_seconds=max(0.0, time.time() - started),
        message=message,
    )


def count_overwrite_candidates(directory: str) -> int:
    count = 0
    for image_path in find_images(directory):
        if image_path.with_suffix(".txt").exists():
            count += 1
    return count


def collect_stats(directory: str) -> TagStatsSnapshot:
    images = find_images(directory)
    counts: dict[str, int] = {}
    captioned_images = 0

    for image in images:
        caption = Caption.from_image(image)
        tags = caption.tags()
        if tags:
            captioned_images += 1
        for tag in tags:
            counts[tag] = counts.get(tag, 0) + 1

    sorted_tags = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    return TagStatsSnapshot(
        total_images=len(images),
        captioned_images=captioned_images,
        tags=sorted_tags,
    )


def _load_model(config: TaggerModelConfig) -> tuple[Any, Any, Any]:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    import timm
    from timm.data import create_transform, resolve_model_data_config

    model = timm.create_model(f"hf-hub:{config.model}", pretrained=True)
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)
    tags_csv = hf_hub_download(config.model, "selected_tags.csv")
    tags_df = pd.read_csv(tags_csv)
    return model, transform, tags_df


def auto_tag(
    directory: str,
    config: TaggerModelConfig,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> TaggerRunSummary:
    from PIL import Image
    import torch

    started = time.time()
    images = find_images(directory)
    if not images:
        return _build_summary(
            command="tag",
            dataset_dir=directory,
            processed_images=0,
            changed_captions=0,
            failed_paths=[],
            started=started,
        )

    model, transform, tags_df = _load_model(config)

    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    model = model.to(device)
    model.eval()

    changed = 0
    failed: list[str] = []

    batch_size = max(1, int(config.batch))
    completed = 0
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        pil_images: list[Any] = []
        opened: list[Any] = []
        valid_images: list[Path] = []

        for path in batch:
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img)
                opened.append(img)
                valid_images.append(path)
            except Exception:
                failed.append(str(path))

        if not valid_images:
            continue

        try:
            tensors = [transform(image) for image in pil_images]
            inputs = torch.stack(tensors, dim=0).to(device)
            with torch.no_grad():
                outputs = model(inputs)
                probs = torch.nn.functional.sigmoid(outputs).cpu()
        except Exception:
            failed.extend(str(path) for path in valid_images)
            for image in opened:
                image.close()
            continue

        for image_index, path in enumerate(valid_images):
            caption = Caption.from_image(path)
            prev_tags = caption.tags()
            image_probs = probs[image_index]
            for tag_index, prob in enumerate(image_probs):
                if float(prob) < config.threshold:
                    continue
                raw_tag = str(tags_df.iloc[tag_index]["name"])
                caption.add(Caption.format(raw_tag))

            next_tags = caption.tags()
            if prev_tags != next_tags:
                caption.save_for_image(path)
                changed += 1
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(images), str(path))

        for image in opened:
            image.close()

        for _ in range(len(batch) - len(valid_images)):
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(images), "")

    return _build_summary(
        command="tag",
        dataset_dir=directory,
        processed_images=len(images),
        changed_captions=changed,
        failed_paths=failed,
        started=started,
    )


def add_tags(directory: str, tags: list[str]) -> TaggerRunSummary:
    started = time.time()
    images = find_images(directory)
    changed = 0
    failed: list[str] = []
    for image in images:
        try:
            caption = Caption.from_image(image)
            mutated = False
            for tag in tags:
                mutated = caption.add(tag) or mutated
            if mutated:
                caption.save_for_image(image)
                changed += 1
        except Exception:
            failed.append(str(image))
    return _build_summary(
        command="add",
        dataset_dir=directory,
        processed_images=len(images),
        changed_captions=changed,
        failed_paths=failed,
        started=started,
    )


def remove_tags(directory: str, tags: list[str]) -> TaggerRunSummary:
    started = time.time()
    images = find_images(directory)
    changed = 0
    failed: list[str] = []
    for image in images:
        try:
            caption = Caption.from_image(image)
            mutated = False
            for tag in tags:
                mutated = caption.remove(tag) or mutated
            if mutated:
                caption.save_for_image(image)
                changed += 1
        except Exception:
            failed.append(str(image))
    return _build_summary(
        command="remove",
        dataset_dir=directory,
        processed_images=len(images),
        changed_captions=changed,
        failed_paths=failed,
        started=started,
    )


def remove_single_tag(directory: str, tag: str) -> TaggerRunSummary:
    return remove_tags(directory, [tag])


def rename_single_tag(directory: str, old_tag: str, new_tag: str) -> TaggerRunSummary:
    started = time.time()
    images = find_images(directory)
    changed = 0
    failed: list[str] = []
    source = old_tag.strip()
    target = new_tag.strip()
    if not source or not target or source == target:
        return _build_summary(
            command="rename",
            dataset_dir=directory,
            processed_images=len(images),
            changed_captions=0,
            failed_paths=[],
            started=started,
            message="No-op rename request.",
        )

    for image in images:
        try:
            caption = Caption.from_image(image)
            before = caption.tags()
            if source not in before:
                continue
            replaced = [target if tag == source else tag for tag in before]
            deduped: list[str] = []
            for tag in replaced:
                if tag not in deduped:
                    deduped.append(tag)
            if before != deduped:
                Caption.from_tags(deduped).save_for_image(image)
                changed += 1
        except Exception:
            failed.append(str(image))
    return _build_summary(
        command="rename",
        dataset_dir=directory,
        processed_images=len(images),
        changed_captions=changed,
        failed_paths=failed,
        started=started,
    )


def delete_all_tags(directory: str) -> TaggerRunSummary:
    started = time.time()
    images = find_images(directory)
    changed = 0
    failed: list[str] = []
    for image in images:
        try:
            caption = Caption.from_image(image)
            if not caption.tags():
                continue
            Caption.from_tags([]).save_for_image(image)
            changed += 1
        except Exception:
            failed.append(str(image))
    return _build_summary(
        command="delete_all",
        dataset_dir=directory,
        processed_images=len(images),
        changed_captions=changed,
        failed_paths=failed,
        started=started,
    )


def front_tags(directory: str, tags: list[str]) -> TaggerRunSummary:
    started = time.time()
    images = find_images(directory)
    changed = 0
    failed: list[str] = []
    for image in images:
        try:
            caption = Caption.from_image(image)
            before = caption.tags()
            for tag in reversed(tags):
                caption.front(tag)
            if before != caption.tags():
                caption.save_for_image(image)
                changed += 1
        except Exception:
            failed.append(str(image))
    return _build_summary(
        command="front",
        dataset_dir=directory,
        processed_images=len(images),
        changed_captions=changed,
        failed_paths=failed,
        started=started,
    )


def shuffle_tags(directory: str) -> TaggerRunSummary:
    started = time.time()
    images = find_images(directory)
    changed = 0
    failed: list[str] = []
    for image in images:
        try:
            caption = Caption.from_image(image)
            before = caption.tags()
            if not before:
                continue
            caption.shuffle()
            if before != caption.tags():
                caption.save_for_image(image)
                changed += 1
        except Exception:
            failed.append(str(image))
    return _build_summary(
        command="shuffle",
        dataset_dir=directory,
        processed_images=len(images),
        changed_captions=changed,
        failed_paths=failed,
        started=started,
    )
