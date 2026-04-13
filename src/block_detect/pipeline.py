from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .classifier import BlockClassifier, ClassificationResult, PlaceholderClassifier
from .config import Settings, ensure_workspace_dirs, load_settings
from .dropbox_client import DropboxClient


@dataclass(frozen=True)
class PipelineSummary:
    processed_count: int
    blocked_count: int
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
        self.classifier = classifier or PlaceholderClassifier()
        self.dropbox_client = dropbox_client or DropboxClient(settings)

    def prepare(self) -> None:
        ensure_workspace_dirs(self.settings)

    def classify_local_images(self, image_paths: list[Path]) -> list[ClassificationResult]:
        return [self.classifier.classify(path) for path in image_paths]

    def summarize(self, results: list[ClassificationResult]) -> PipelineSummary:
        blocked = sum(1 for result in results if result.label == "blocked")
        normal = sum(1 for result in results if result.label == "normal")
        unknown = sum(1 for result in results if result.label not in {"blocked", "normal"})
        return PipelineSummary(
            processed_count=len(results),
            blocked_count=blocked,
            normal_count=normal,
            unknown_count=unknown,
        )


def build_pipeline() -> DetectionPipeline:
    return DetectionPipeline(settings=load_settings())

