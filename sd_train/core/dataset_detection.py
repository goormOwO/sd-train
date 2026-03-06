from pathlib import Path

import toml


def guess_dataset_dir_from_train_config(train_config_path: str) -> str:
    path = Path(train_config_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()
    if not path.is_file():
        return ""
    try:
        data = toml.load(path)
    except Exception:
        return ""
    value = data.get("train_data_dir")
    if not isinstance(value, str) or not value.strip():
        return ""
    dataset = Path(value.strip()).expanduser()
    if not dataset.is_absolute():
        dataset = (path.parent / dataset).resolve()
    return str(dataset)
