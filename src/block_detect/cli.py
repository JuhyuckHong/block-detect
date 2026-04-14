from __future__ import annotations

import argparse
from dataclasses import replace

from .config import load_settings
from .pipeline import DetectionPipeline


def build_pipeline(settings=None) -> DetectionPipeline:
    return DetectionPipeline(settings=settings or load_settings())


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
    parser.add_argument(
        "--date",
        help="Target capture date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--dropbox-path",
        help="Override the resolved Dropbox day path.",
    )
    parser.add_argument(
        "--dark-threshold",
        type=int,
        help="Pixel brightness threshold to count as dark.",
    )
    parser.add_argument(
        "--dark-ratio-threshold",
        type=float,
        help="Minimum ratio of dark pixels for abnormal detection.",
    )
    parser.add_argument(
        "--mean-brightness-threshold",
        type=float,
        help="Maximum mean brightness for abnormal detection.",
    )
    parser.add_argument(
        "--download-workers",
        type=int,
        help="Number of concurrent Dropbox downloads.",
    )
    parser.add_argument(
        "--classify-workers",
        type=int,
        help="Number of concurrent image classification workers.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings()
    settings = replace(
        settings,
        dark_threshold=args.dark_threshold
        if args.dark_threshold is not None
        else settings.dark_threshold,
        dark_ratio_threshold=args.dark_ratio_threshold
        if args.dark_ratio_threshold is not None
        else settings.dark_ratio_threshold,
        mean_brightness_threshold=args.mean_brightness_threshold
        if args.mean_brightness_threshold is not None
        else settings.mean_brightness_threshold,
        download_workers=max(1, args.download_workers)
        if args.download_workers is not None
        else settings.download_workers,
        classify_workers=max(1, args.classify_workers)
        if args.classify_workers is not None
        else settings.classify_workers,
    )
    pipeline = build_pipeline(settings=settings)

    if args.prepare_only:
        pipeline.prepare()
        return 0

    if not args.date:
        parser.print_help()
        return 0

    summary = pipeline.run_day(args.date, remote_day_path=args.dropbox_path)
    print(
        " ".join(
            [
                f"date={args.date}",
                f"processed={summary.processed_count}",
                f"abnormal={summary.abnormal_count}",
                f"normal={summary.normal_count}",
                f"unknown={summary.unknown_count}",
            ]
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
