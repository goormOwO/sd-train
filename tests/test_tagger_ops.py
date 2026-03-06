from pathlib import Path

from tests.helpers import create_tagger_dataset_single
from sd_train.tagger.core import (
    add_tags,
    collect_stats,
    delete_all_tags,
    front_tags,
    remove_tags,
    shuffle_tags,
)


def _caption(dataset: Path) -> str:
    return (dataset / "img.txt").read_text(encoding="utf-8")


def test_caption_ops_regression_smoke(tmp_path: Path) -> None:
    dataset = create_tagger_dataset_single(tmp_path)

    add_tags(str(dataset), ["flower"])
    assert "flower" in _caption(dataset)

    remove_tags(str(dataset), ["dog"])
    assert "dog" not in _caption(dataset)

    front_tags(str(dataset), ["flower"])
    assert _caption(dataset).startswith("flower")

    shuffle_tags(str(dataset))
    tags = {tag.strip() for tag in _caption(dataset).split(",") if tag.strip()}
    assert tags == {"flower", "cat"}

    delete_all_tags(str(dataset))
    assert _caption(dataset).strip() == ""

    stats = collect_stats(str(dataset))
    assert stats.total_images == 1
