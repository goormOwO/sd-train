import argparse

from sd_train.app.launcher import main as launch_main
from sd_train.app.launcher import run_last_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sd-train")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("last", help="Start training immediately with the last saved settings")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "last":
        run_last_training()
        return
    launch_main()


if __name__ == "__main__":
    main()
