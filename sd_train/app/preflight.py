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


def run_preflight_or_raise(
    train_config_path: Path,
    script: str,
    auth: DownloadAuth,
) -> PreflightGateResult:
    print("\x1b[36mRunning local preflight validation...\x1b[0m")
    try:
        report, checks = run_preflight_checks(train_config_path, script, auth)
    except Exception as exc:
        message = str(exc)
        if "External reference" in message:
            message = f"External reference check failed: {message}"
        raise RuntimeError(message) from exc

    failure_message = build_external_ref_failure_message(checks)
    if failure_message:
        raise RuntimeError(failure_message)

    return PreflightGateResult(report=report, checks=checks)


def run_preflight_gate(
    train_config_path: Path,
    script: str,
    auth: DownloadAuth,
) -> PreflightGateResult | None:
    try:
        gate = run_preflight_or_raise(train_config_path, script, auth)
    except Exception as exc:
        PreflightReviewApp(summary="", error_message=str(exc)).run()
        return None

    proceed = PreflightReviewApp(summary=build_preflight_summary(gate.report, gate.checks)).run()
    if not proceed:
        return None

    return gate
