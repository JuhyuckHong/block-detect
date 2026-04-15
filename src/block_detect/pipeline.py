from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

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
            roi_line_offset_ratio=settings.roi_line_offset_ratio,
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

    def run_day_with_details(
        self,
        day: str,
        remote_day_path: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineRunResult:
        self.prepare()
        resolved_remote_path = remote_day_path or self.build_remote_day_path(day)
        local_target_dir = self.settings.inbox_dir / day
        remote_paths = self.dropbox_client.list_day_images(resolved_remote_path)
        image_paths = self.dropbox_client.download_images(
            remote_paths,
            local_target_dir,
            progress_callback=None
            if progress_callback is None
            else lambda current, total: progress_callback("download", current, total),
        )
        results = self.classify_local_images(image_paths, progress_callback=progress_callback)
        summary = self.summarize(results)
        report_path = self.write_report(day, resolved_remote_path, results, summary)
        return PipelineRunResult(
            day=day,
            remote_day_path=resolved_remote_path,
            results=results,
            summary=summary,
            report_path=report_path,
            classification_settings={
                "score_threshold": self.settings.score_threshold,
                "roi_line_offset_ratio": self.settings.roi_line_offset_ratio,
            },
        )

    def run_local_day_with_details(
        self,
        day: str,
        remote_day_path: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> PipelineRunResult:
        self.prepare()
        resolved_remote_path = remote_day_path or self.build_remote_day_path(day)
        image_paths = self.list_local_images(day)
        results = self.classify_local_images(image_paths, progress_callback=progress_callback)
        summary = self.summarize(results)
        report_path = self.write_report(day, resolved_remote_path, results, summary)
        return PipelineRunResult(
            day=day,
            remote_day_path=resolved_remote_path,
            results=results,
            summary=summary,
            report_path=report_path,
            classification_settings={
                "score_threshold": self.settings.score_threshold,
                "roi_line_offset_ratio": self.settings.roi_line_offset_ratio,
            },
        )

    def run_day(self, day: str, remote_day_path: str | None = None) -> PipelineSummary:
        return self.run_day_with_details(day, remote_day_path=remote_day_path).summary

    def write_report(
        self,
        day: str,
        remote_day_path: str,
        results: list[ClassificationResult],
        summary: PipelineSummary,
    ) -> Path:
        report_path = self.settings.reports_dir / f"{day}.json"
        payload = {
            "day": day,
            "remote_day_path": remote_day_path,
            "classification_settings": {
                "score_threshold": self.settings.score_threshold,
                "roi_line_offset_ratio": self.settings.roi_line_offset_ratio,
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


def load_saved_run(day: str, settings: Settings | None = None) -> PipelineRunResult:
    resolved_settings = settings or load_settings()
    report_path = resolved_settings.reports_dir / f"{day}.json"
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
            "roi_line_offset_ratio": float(
                classification_settings_payload.get(
                    "roi_line_offset_ratio",
                    resolved_settings.roi_line_offset_ratio,
                )
            ),
        },
    )
