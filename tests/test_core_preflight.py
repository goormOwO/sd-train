from pathlib import Path

from sd_train.core.preflight import (
    build_external_ref_failure_message,
    build_preflight_summary,
    run_preflight_checks,
)
from sd_train.domain.preflight import PreflightReport
from sd_train.domain.refs import DownloadAuth, ExternalRefCheck


def _sample_report(path: Path) -> PreflightReport:
    return PreflightReport(
        train_config_path=path,
        train_script="train_network.py",
        dataset_mode="train_data_dir",
        dataset_roots=[path.parent / "dataset"],
        subset_dirs=[path.parent / "dataset" / "subset-a"],
        image_count=5,
        caption_count=4,
        sample_prompts_count=2,
        max_train_steps=1000,
        train_batch_size=2,
        save_every_n_steps=100,
        sample_every_n_steps=50,
        output_name="out",
    )


def test_build_external_ref_failure_message_only_for_failed_checks() -> None:
    checks = [
        ExternalRefCheck(key="a", ref="model:org/repo", provider="hf", ok=True, detail="ok"),
        ExternalRefCheck(key="b", ref="civitai:123", provider="civitai", ok=False, detail="unauthorized"),
    ]
    message = build_external_ref_failure_message(checks)
    assert "External reference check failed" in message
    assert "[civitai] b: civitai:123" in message
    assert "unauthorized" in message


def test_build_preflight_summary_contains_main_fields(tmp_path: Path) -> None:
    report = _sample_report(tmp_path / "train.toml")
    checks = [
        ExternalRefCheck(key="pretrained_model_name_or_path", ref="model:org/repo::x", provider="hf", ok=True, detail="OK")
    ]
    summary = build_preflight_summary(report, checks)
    assert "train_script: train_network.py" in summary
    assert "image_count: 5" in summary
    assert "sample_prompts: 2" in summary
    assert "external_refs_checked: 1" in summary


def test_build_preflight_summary_without_optionals(tmp_path: Path) -> None:
    report = PreflightReport(
        train_config_path=tmp_path / "train.toml",
        train_script="train.py",
        dataset_mode="dataset_config",
        dataset_roots=[],
        subset_dirs=[],
        image_count=0,
        caption_count=0,
        sample_prompts_count=None,
        max_train_steps=None,
        train_batch_size=None,
        save_every_n_steps=None,
        sample_every_n_steps=None,
        output_name=None,
    )
    summary = build_preflight_summary(report, [])
    assert "dataset_roots" not in summary
    assert "sample_prompts" not in summary
    assert "external_refs_checked" not in summary


def test_run_preflight_checks_delegates(monkeypatch, tmp_path: Path) -> None:
    report = _sample_report(tmp_path / "train.toml")
    checks = [ExternalRefCheck(key="k", ref="r", provider="hf", ok=True, detail="OK")]
    import sd_train.core.preflight as preflight

    monkeypatch.setattr(preflight, "validate_train_config_lightweight", lambda *_args: report)
    monkeypatch.setattr(preflight, "verify_external_refs_accessibility", lambda *_args: checks)
    out_report, out_checks = run_preflight_checks(tmp_path / "train.toml", "train_network.py", DownloadAuth())
    assert out_report is report
    assert out_checks is checks
