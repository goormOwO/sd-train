from pathlib import Path

import pytest
import toml

from sd_train.domain.preflight import validate_train_config_lightweight


def _write_image(path: Path) -> None:
    path.write_bytes(b"fake")


def test_preflight_succeeds_with_train_data_dir(tmp_path: Path) -> None:
    subset = tmp_path / "dataset" / "subset-a"
    subset.mkdir(parents=True)
    _write_image(subset / "a.png")
    (subset / "a.txt").write_text("tag1, tag2", encoding="utf-8")

    train_toml = tmp_path / "train.toml"
    train_toml.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::model.safetensors",
                "train_data_dir": str(tmp_path / "dataset"),
            }
        ),
        encoding="utf-8",
    )

    report = validate_train_config_lightweight(train_toml, "train_network.py")
    assert report.image_count == 1
    assert report.caption_count == 1


def test_preflight_succeeds_with_dataset_config(tmp_path: Path) -> None:
    subset = tmp_path / "images"
    subset.mkdir(parents=True)
    _write_image(subset / "b.png")

    dataset_cfg = tmp_path / "dataset.toml"
    dataset_cfg.write_text(
        toml.dumps({"datasets": [{"subsets": [{"image_dir": str(subset)}]}]}),
        encoding="utf-8",
    )

    train_toml = tmp_path / "train.toml"
    train_toml.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::model.safetensors",
                "dataset_config": str(dataset_cfg),
            }
        ),
        encoding="utf-8",
    )

    report = validate_train_config_lightweight(train_toml, "train_network.py")
    assert report.dataset_mode == "dataset_config"
    assert report.image_count == 1


def test_preflight_fails_when_required_key_missing(tmp_path: Path) -> None:
    train_toml = tmp_path / "train.toml"
    train_toml.write_text(toml.dumps({"train_data_dir": "./dataset"}), encoding="utf-8")

    try:
        validate_train_config_lightweight(train_toml, "train_network.py")
    except ValueError as exc:
        assert "pretrained_model_name_or_path" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_preflight_fails_when_subset_has_no_images(tmp_path: Path) -> None:
    subset = tmp_path / "dataset" / "subset-a"
    subset.mkdir(parents=True)
    (subset / "readme.txt").write_text("not image", encoding="utf-8")
    train_toml = tmp_path / "train.toml"
    train_toml.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::model.safetensors",
                "train_data_dir": str(tmp_path / "dataset"),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No subset directory with images"):
        validate_train_config_lightweight(train_toml, "train_network.py")


def test_preflight_dataset_config_hf_mode(tmp_path: Path) -> None:
    train_toml = tmp_path / "train.toml"
    train_toml.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::model.safetensors",
                "dataset_config": "dataset:org/repo::sub/dataset.toml",
            }
        ),
        encoding="utf-8",
    )
    report = validate_train_config_lightweight(train_toml, "train_network.py")
    assert report.dataset_mode == "dataset_config(hf)"


def test_preflight_hf_ref_without_subpath_for_file_key_fails(tmp_path: Path) -> None:
    subset = tmp_path / "dataset" / "subset-a"
    subset.mkdir(parents=True)
    _write_image(subset / "a.png")
    train_toml = tmp_path / "train.toml"
    train_toml.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "org/repo",
                "train_data_dir": str(tmp_path / "dataset"),
                "network_weights": "model:org/repo",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must include subpath"):
        validate_train_config_lightweight(train_toml, "train_network.py")


def test_preflight_accepts_explicit_ref_for_unknown_key(tmp_path: Path) -> None:
    subset = tmp_path / "dataset" / "subset-a"
    subset.mkdir(parents=True)
    _write_image(subset / "a.png")
    train_toml = tmp_path / "train.toml"
    train_toml.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::model.safetensors",
                "train_data_dir": str(tmp_path / "dataset"),
                "custom_text_encoder": "model:org/repo",
            }
        ),
        encoding="utf-8",
    )

    report = validate_train_config_lightweight(train_toml, "train_network.py")
    assert report.image_count == 1


def test_preflight_sample_prompts_count_ignores_comments(tmp_path: Path) -> None:
    subset = tmp_path / "dataset" / "subset-a"
    subset.mkdir(parents=True)
    _write_image(subset / "a.png")
    prompts = tmp_path / "sample.txt"
    prompts.write_text("# comment\n\nhello\nworld\n", encoding="utf-8")
    train_toml = tmp_path / "train.toml"
    train_toml.write_text(
        toml.dumps(
            {
                "pretrained_model_name_or_path": "model:org/repo::model.safetensors",
                "train_data_dir": str(tmp_path / "dataset"),
                "sample_prompts": str(prompts),
            }
        ),
        encoding="utf-8",
    )

    report = validate_train_config_lightweight(train_toml, "train_network.py")
    assert report.sample_prompts_count == 2
