from __future__ import annotations

import argparse

from .pipeline import build_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="block-detect",
        description="Workspace scaffold for daily blocked-image detection.",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Create workspace directories and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pipeline = build_pipeline()
    pipeline.prepare()

    if args.prepare_only:
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

