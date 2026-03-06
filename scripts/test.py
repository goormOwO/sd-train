import subprocess
import sys


def main() -> None:
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov=sd_train",
                "--cov-config=.coveragerc",
                "--cov-report=term-missing",
                "-q",
            ]
        )
    )
