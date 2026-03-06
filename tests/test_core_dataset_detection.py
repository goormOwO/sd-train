from pathlib import Path

from sd_train.core.dataset_detection import guess_dataset_dir_from_train_config


def test_guess_dataset_dir_missing_file_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "missing.toml"
    assert guess_dataset_dir_from_train_config(str(path)) == ""


def test_guess_dataset_dir_invalid_toml_returns_empty(tmp_path: Path) -> None:
    path = tmp_path / "train.toml"
    path.write_text("invalid=[", encoding="utf-8")
    assert guess_dataset_dir_from_train_config(str(path)) == ""


def test_guess_dataset_dir_resolves_relative_path(tmp_path: Path) -> None:
    dataset = tmp_path / "data"
    dataset.mkdir()
    train = tmp_path / "train.toml"
    train.write_text('train_data_dir = "./data"\n', encoding="utf-8")
    assert guess_dataset_dir_from_train_config(str(train)) == str(dataset.resolve())
