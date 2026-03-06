from pathlib import Path

from tests.helpers import create_tagger_dataset_pair
from sd_train.tagger.core import (
    Caption,
    count_overwrite_candidates,
    find_images,
    remove_single_tag,
    rename_single_tag,
)


def test_find_images_filters_by_suffix_and_missing_dir(tmp_path: Path) -> None:
    dataset = create_tagger_dataset_pair(tmp_path)
    images = find_images(str(dataset))
    assert [image.name for image in images] == ["a.png", "b.jpg"]
    assert find_images(str(tmp_path / "missing")) == []


def test_count_overwrite_candidates_counts_existing_captions(tmp_path: Path) -> None:
    dataset = create_tagger_dataset_pair(tmp_path)
    assert count_overwrite_candidates(str(dataset)) == 2


def test_rename_single_tag_noop_message(tmp_path: Path) -> None:
    dataset = create_tagger_dataset_pair(tmp_path)
    summary = rename_single_tag(str(dataset), "cat", "cat")
    assert summary.changed_captions == 0
    assert summary.message == "No-op rename request."


def test_rename_single_tag_dedupes_tags(tmp_path: Path) -> None:
    dataset = create_tagger_dataset_pair(tmp_path)
    # a.txt: cat, dog -> dog, b.txt: cat, bird -> bird
    summary = rename_single_tag(str(dataset), "cat", "dog")
    assert summary.changed_captions >= 1
    first = Caption.from_image(dataset / "a.png").tags()
    assert first == ["dog"]


def test_remove_single_tag_updates_caption(tmp_path: Path) -> None:
    dataset = create_tagger_dataset_pair(tmp_path)
    summary = remove_single_tag(str(dataset), "bird")
    assert summary.changed_captions == 1
    second = Caption.from_image(dataset / "b.jpg").tags()
    assert "bird" not in second
