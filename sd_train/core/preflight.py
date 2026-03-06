from pathlib import Path

from sd_train.domain.preflight import PreflightReport, validate_train_config_lightweight
from sd_train.domain.refs import DownloadAuth, ExternalRefCheck, verify_external_refs_accessibility


def run_preflight_checks(
    train_config_path: Path,
    script: str,
    auth: DownloadAuth,
) -> tuple[PreflightReport, list[ExternalRefCheck]]:
    report = validate_train_config_lightweight(train_config_path, script)
    checks = verify_external_refs_accessibility(train_config_path, auth)
    return report, checks


def build_external_ref_failure_message(checks: list[ExternalRefCheck]) -> str:
    failed_checks = [check for check in checks if not check.ok]
    if not failed_checks:
        return ""
    lines = ["External reference check failed:"]
    for check in failed_checks:
        lines.append(f"- [{check.provider}] {check.key}: {check.ref}")
        lines.append(f"  {check.detail}")
    return "\n".join(lines)


def build_preflight_summary(report: PreflightReport, checks: list[ExternalRefCheck]) -> str:
    summary_lines: list[str] = [
        f"train_config: {report.train_config_path}",
        f"train_script: {report.train_script}",
        f"dataset_mode: {report.dataset_mode}",
    ]
    if report.dataset_roots:
        roots = ", ".join(str(path) for path in report.dataset_roots[:3])
        extra = f" (+{len(report.dataset_roots) - 3})" if len(report.dataset_roots) > 3 else ""
        summary_lines.append(f"dataset_roots: {roots}{extra}")
    if report.subset_dirs:
        summary_lines.append(f"subset_dirs: {len(report.subset_dirs)}")
    summary_lines.append(f"image_count: {report.image_count}")
    summary_lines.append(f"caption_count: {report.caption_count}")
    if report.sample_prompts_count is not None:
        summary_lines.append(f"sample_prompts: {report.sample_prompts_count}")
    if report.max_train_steps is not None:
        summary_lines.append(f"max_train_steps: {report.max_train_steps}")
    if report.train_batch_size is not None:
        summary_lines.append(f"train_batch_size: {report.train_batch_size}")
    if report.save_every_n_steps is not None:
        summary_lines.append(f"save_every_n_steps: {report.save_every_n_steps}")
    if report.sample_every_n_steps is not None:
        summary_lines.append(f"sample_every_n_steps: {report.sample_every_n_steps}")
    if report.output_name:
        summary_lines.append(f"output_name: {report.output_name}")
    if checks:
        summary_lines.append(f"external_refs_checked: {len(checks)}")
        summary_lines.extend([f"  - [{check.provider}] {check.key}: {check.detail}" for check in checks])
    return "\n".join(summary_lines)
