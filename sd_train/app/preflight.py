from dataclasses import dataclass
from pathlib import Path

from sd_train.core.preflight import (
    build_external_ref_failure_message,
    build_preflight_summary,
    run_preflight_checks,
)
from sd_train.domain.preflight import PreflightReport
from sd_train.domain.refs import DownloadAuth, ExternalRefCheck
from sd_train.ui.apps.launcher import PreflightReviewApp


@dataclass
class PreflightGateResult:
    report: PreflightReport
    checks: list[ExternalRefCheck]


def run_preflight_gate(
    train_config_path: Path,
    script: str,
    auth: DownloadAuth,
) -> PreflightGateResult | None:
    print("\x1b[36mRunning local preflight validation...\x1b[0m")
    try:
        report, checks = run_preflight_checks(train_config_path, script, auth)
    except Exception as exc:
        message = str(exc)
        if "External reference" in message:
            message = f"External reference check failed: {message}"
        PreflightReviewApp(summary="", error_message=message).run()
        return None

    failure_message = build_external_ref_failure_message(checks)
    if failure_message:
        PreflightReviewApp(summary="", error_message=failure_message).run()
        return None

    proceed = PreflightReviewApp(summary=build_preflight_summary(report, checks)).run()
    if not proceed:
        return None

    return PreflightGateResult(report=report, checks=checks)
