from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from .classifier import BlackPixelClassifier, BlockClassifier, ClassificationResult
from .config import Settings, ensure_workspace_dirs, load_settings
from .dropbox_client import DropboxClient


@dataclass(frozen=True)
class PipelineSummary:
    processed_count: int
    abnormal_count: int
    normal_count: int
    unknown_count: int


class DetectionPipeline:
    def __init__(
        self,
        settings: Settings,
        classifier: BlockClassifier | None = None,
        dropbox_client: DropboxClient | None = None,
    ):
        self.settings = settings
        self.classifier = classifier or BlackPixelClassifier(
            dark_threshold=settings.dark_threshold,
            dark_ratio_threshold=settings.dark_ratio_threshold,
            mean_brightness_threshold=settings.mean_brightness_threshold,
        )
        self.dropbox_client = dropbox_client or DropboxClient(settings)

    def prepare(self) -> None:
        ensure_workspace_dirs(self.settings)

    def classify_local_images(self, image_paths: list[Path]) -> list[ClassificationResult]:
        return [self.classifier.classify(path) for path in image_paths]

    def summarize(self, results: list[ClassificationResult]) -> PipelineSummary:
        abnormal = sum(1 for result in results if result.label == "abnormal")
        normal = sum(1 for result in results if result.label == "normal")
        unknown = sum(1 for result in results if result.label not in {"abnormal", "normal"})
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

    def run_day(self, day: str, remote_day_path: str | None = None) -> PipelineSummary:
        self.prepare()
        resolved_remote_path = remote_day_path or self.build_remote_day_path(day)
        local_target_dir = self.settings.inbox_dir / day
        remote_paths = self.dropbox_client.list_day_images(resolved_remote_path)
        image_paths = self.dropbox_client.download_images(remote_paths, local_target_dir)
        results = self.classify_local_images(image_paths)
        summary = self.summarize(results)
        self.write_report(day, resolved_remote_path, results, summary)
        return summary

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


def build_pipeline() -> DetectionPipeline:
    return DetectionPipeline(settings=load_settings())
