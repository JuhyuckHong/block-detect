from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any, Callable

from .classifier import BlackPixelClassifier, BlockClassifier, ClassificationResult
from .config import Settings, ensure_workspace_dirs, load_settings
from .dropbox_client import DropboxClient


@dataclass(frozen=True)
class PipelineSummary:
    processed_count: int
    abnormal_count: int
    normal_count: int
    unknown_count: int


@dataclass(frozen=True)
class PipelineRunResult:
    day: str
    remote_day_path: str
    results: list[ClassificationResult]
    summary: PipelineSummary
    report_path: Path
    classification_settings: dict[str, float]


ProgressCallback = Callable[[str, int, int], None]
TIME_PATTERN = re.compile(r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})$")
DAY_START_SECONDS = 0
DAY_END_SECONDS = (24 * 3600) - 1


@dataclass(frozen=True)
class TimeRange:
    start_seconds: int
    end_seconds: int

    def contains(self, capture_seconds: int | None) -> bool:
        if capture_seconds is None:
            return False
        return self.start_seconds <= capture_seconds <= self.end_seconds

    def report_suffix(self) -> str:
        return f"{format_seconds_for_filename(self.start_seconds)}-{format_seconds_for_filename(self.end_seconds)}"

    def display_text(self) -> str:
        return f"{format_seconds_for_display(self.start_seconds)}~{format_seconds_for_display(self.end_seconds)}"


def extract_capture_seconds(image_path: Path) -> int | None:
    stem = image_path.stem
    candidate = stem.split("_")[-1]
    match = TIME_PATTERN.search(candidate)
    if match is None:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute"))
    second = int(match.group("second"))
    if hour >= 24 or minute >= 60 or second >= 60:
        return None
    return hour * 3600 + minute * 60 + second


def parse_time_value(value: str) -> int:
    text = value.strip()
    for pattern in ("%H:%M", "%H:%M:%S", "%H%M", "%H%M%S"):
        try:
            parsed = datetime.strptime(text, pattern)
            return parsed.hour * 3600 + parsed.minute * 60 + parsed.second
        except ValueError:
            continue
    raise ValueError("Time must be in HH:MM or HH:MM:SS format.")


def build_time_range(start_time: str | None, end_time: str | None) -> TimeRange | None:
    start_text = (start_time or "").strip()
    end_text = (end_time or "").strip()
    if not start_text and not end_text:
        return None
    start_seconds = parse_time_value(start_text) if start_text else DAY_START_SECONDS
    end_seconds = parse_time_value(end_text) if end_text else DAY_END_SECONDS
    if start_seconds > end_seconds:
        raise ValueError("Start time must be earlier than or equal to end time.")
    return TimeRange(start_seconds=start_seconds, end_seconds=end_seconds)


def format_seconds_for_display(value: int) -> str:
    hour = value // 3600
    minute = (value % 3600) // 60
    second = value % 60
    if second == 0:
        return f"{hour:02d}:{minute:02d}"
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def format_seconds_for_filename(value: int) -> str:
    hour = value // 3600
    minute = (value % 3600) // 60
    second = value % 60
    return f"{hour:02d}{minute:02d}{second:02d}"


class DetectionPipeline:
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    def __init__(
        self,
        settings: Settings,
        classifier: BlockClassifier | None = None,
        dropbox_client: DropboxClient | None = None,
    ):
        self.settings = settings
        self.classifier = classifier or BlackPixelClassifier(
            dark_threshold=settings.dark_threshold,
            score_threshold=settings.score_threshold,
            roi_left_y_ratio=settings.roi_left_y_ratio,
            roi_bottom_x_ratio=settings.roi_bottom_x_ratio,
        )
        self.dropbox_client = dropbox_client or DropboxClient(settings)

    def prepare(self) -> None:
        ensure_workspace_dirs(self.settings)

    def classify_local_images(
        self,
        image_paths: list[Path],
        progress_callback: ProgressCallback | None = None,
    ) -> list[ClassificationResult]:
        if not image_paths:
            if progress_callback is not None:
                progress_callback("classify", 0, 0)
            return []

        max_workers = min(self.settings.classify_workers, len(image_paths))
        if progress_callback is not None:
            progress_callback("classify", 0, len(image_paths))
        if max_workers == 1:
            results: list[ClassificationResult] = []
            total = len(image_paths)
            for index, path in enumerate(image_paths, start=1):
                results.append(self.classifier.classify(path))
                if progress_callback is not None:
                    progress_callback("classify", index, total)
            return results

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self.classifier.classify, image_path): index
                for index, image_path in enumerate(image_paths)
            }
            results: list[ClassificationResult | None] = [None] * len(image_paths)
            completed = 0
            total = len(image_paths)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                results[index] = future.result()
                completed += 1
                if progress_callback is not None:
                    progress_callback("classify", completed, total)
            return [result for result in results if result is not None]

    def summarize(self, results: list[ClassificationResult]) -> PipelineSummary:
        abnormal = sum(1 for result in results if result.label in {"abnormal", "blocked"})
        normal = sum(1 for result in results if result.label == "normal")
        unknown = sum(1 for result in results if result.label not in {"abnormal", "blocked", "normal"})
        return PipelineSummary(
            processed_count=len(results),
            abnormal_count=abnormal,
            normal_count=normal,
            unknown_count=unknown,
        )

    def build_remote_day_path(self, day: str) -> str:
        relative_path = self.settings.dropbox_day_template.format(date=day).strip("/")
        root = self.settings.dropbox_root.rstrip("/")
        if not relative_path:
            return root or "/"
        if not root:
            return f"/{relative_path}"
        return f"{root}/{relative_path}"

    def local_day_dir(self, day: str) -> Path:
        return self.settings.inbox_dir / day

    def list_local_images(self, day: str) -> list[Path]:
        local_day_dir = self.local_day_dir(day)
        if not local_day_dir.exists():
            return []
        image_extensions = {
            extension.lower()
            for extension in getattr(self.dropbox_client, "IMAGE_EXTENSIONS", self.IMAGE_EXTENSIONS)
        }
        return sorted(
            path
            for path in local_day_dir.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )

    def filter_image_paths_by_time_range(
        self,
        image_paths: list[Path],
        time_range: TimeRange | None = None,
    ) -> list[Path]:
        if time_range is None:
            return image_paths
        return [
            image_path
            for image_path in image_paths
            if time_range.contains(extract_capture_seconds(image_path))
        ]

    def filter_remote_images_by_time_range(
        self,
        remote_images: list[Any],
        time_range: TimeRange | None = None,
    ) -> list[Any]:
        if time_range is None:
            return remote_images
        filtered = []
        for remote_image in remote_images:
            capture_seconds = extract_capture_seconds(Path(getattr(remote_image, "name", "")))
            if capture_seconds is None:
                capture_seconds = extract_capture_seconds(Path(getattr(remote_image, "path_display", "")))
            if time_range.contains(capture_seconds):
                filtered.append(remote_image)
        return filtered

    def run_day_with_details(
        self,
        day: str,
        remote_day_path: str | None = None,
        time_range: TimeRange | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineRunResult:
        self.prepare()
        resolved_remote_path = remote_day_path or self.build_remote_day_path(day)
        local_target_dir = self.settings.inbox_dir / day
        remote_paths = self.dropbox_client.list_day_images(resolved_remote_path)
        remote_paths = self.filter_remote_images_by_time_range(remote_paths, time_range=time_range)
        image_paths = self.dropbox_client.download_images(
            remote_paths,
            local_target_dir,
            progress_callback=None
            if progress_callback is None
            else lambda current, total: progress_callback("download", current, total),
        )
        results = self.classify_local_images(image_paths, progress_callback=progress_callback)
        summary = self.summarize(results)
        report_path = self.write_report(day, resolved_remote_path, results, summary, time_range=time_range)
        return PipelineRunResult(
            day=day,
            remote_day_path=resolved_remote_path,
            results=results,
            summary=summary,
            report_path=report_path,
            classification_settings={
                "score_threshold": self.settings.score_threshold,
                "roi_left_y_ratio": self.settings.roi_left_y_ratio,
                "roi_bottom_x_ratio": self.settings.roi_bottom_x_ratio,
            },
        )

    def run_local_day_with_details(
        self,
        day: str,
        remote_day_path: str | None = None,
        time_range: TimeRange | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineRunResult:
        self.prepare()
        resolved_remote_path = remote_day_path or self.build_remote_day_path(day)
        image_paths = self.filter_image_paths_by_time_range(self.list_local_images(day), time_range=time_range)
        results = self.classify_local_images(image_paths, progress_callback=progress_callback)
        summary = self.summarize(results)
        report_path = self.write_report(day, resolved_remote_path, results, summary, time_range=time_range)
        return PipelineRunResult(
            day=day,
            remote_day_path=resolved_remote_path,
            results=results,
            summary=summary,
            report_path=report_path,
            classification_settings={
                "score_threshold": self.settings.score_threshold,
                "roi_left_y_ratio": self.settings.roi_left_y_ratio,
                "roi_bottom_x_ratio": self.settings.roi_bottom_x_ratio,
            },
        )

    def run_day(
        self,
        day: str,
        remote_day_path: str | None = None,
        time_range: TimeRange | None = None,
    ) -> PipelineSummary:
        return self.run_day_with_details(day, remote_day_path=remote_day_path, time_range=time_range).summary

    def report_path_for_day(self, day: str, time_range: TimeRange | None = None) -> Path:
        report_name = day if time_range is None else f"{day}__{time_range.report_suffix()}"
        return self.settings.reports_dir / f"{report_name}.json"

    def write_report(
        self,
        day: str,
        remote_day_path: str,
        results: list[ClassificationResult],
        summary: PipelineSummary,
        time_range: TimeRange | None = None,
    ) -> Path:
        report_path = self.report_path_for_day(day, time_range=time_range)
        payload = {
            "day": day,
            "remote_day_path": remote_day_path,
            "time_range": None
            if time_range is None
            else {
                "start_seconds": time_range.start_seconds,
                "end_seconds": time_range.end_seconds,
                "display": time_range.display_text(),
            },
            "classification_settings": {
                "score_threshold": self.settings.score_threshold,
                "roi_left_y_ratio": self.settings.roi_left_y_ratio,
                "roi_bottom_x_ratio": self.settings.roi_bottom_x_ratio,
            },
            "summary": {
                "processed_count": summary.processed_count,
                "abnormal_count": summary.abnormal_count,
                "normal_count": summary.normal_count,
                "unknown_count": summary.unknown_count,
            },
            "results": [
                {
                    "image_path": str(result.image_path),
                    "label": result.label,
                    "score": result.score,
                    "reason": result.reason,
                }
                for result in results
            ],
        }
        report_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return report_path


def build_pipeline(settings: Settings | None = None) -> DetectionPipeline:
    return DetectionPipeline(settings=settings or load_settings())


def load_saved_run(
    day: str,
    settings: Settings | None = None,
    time_range: TimeRange | None = None,
) -> PipelineRunResult:
    resolved_settings = settings or load_settings()
    report_name = day if time_range is None else f"{day}__{time_range.report_suffix()}"
    report_path = resolved_settings.reports_dir / f"{report_name}.json"
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    results = [
        ClassificationResult(
            image_path=Path(item["image_path"]),
            label=str(item["label"]),
            score=float(item["score"]),
            reason=str(item["reason"]),
        )
        for item in payload.get("results", [])
    ]

    summary_payload = payload.get("summary", {})
    classification_settings_payload = payload.get("classification_settings", {})
    summary = PipelineSummary(
        processed_count=int(summary_payload.get("processed_count", len(results))),
        abnormal_count=int(summary_payload.get("abnormal_count", 0)),
        normal_count=int(summary_payload.get("normal_count", 0)),
        unknown_count=int(summary_payload.get("unknown_count", 0)),
    )

    return PipelineRunResult(
        day=str(payload.get("day", day)),
        remote_day_path=str(payload.get("remote_day_path", "")),
        results=results,
        summary=summary,
        report_path=report_path,
        classification_settings={
            "score_threshold": float(
                classification_settings_payload.get(
                    "score_threshold",
                    resolved_settings.score_threshold,
                )
            ),
            "roi_left_y_ratio": float(
                classification_settings_payload.get(
                    "roi_left_y_ratio",
                    classification_settings_payload.get(
                        "roi_line_offset_ratio",
                        resolved_settings.roi_left_y_ratio,
                    ),
                )
            ),
            "roi_bottom_x_ratio": float(
                classification_settings_payload.get(
                    "roi_bottom_x_ratio",
                    resolved_settings.roi_bottom_x_ratio,
                )
            ),
        },
    )
