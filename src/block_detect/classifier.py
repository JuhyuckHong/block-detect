from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, ImageStat


@dataclass(frozen=True)
class ClassificationResult:
    image_path: Path
    label: str
    score: float
    reason: str


class BlockClassifier(Protocol):
    def classify(self, image_path: Path) -> ClassificationResult:
        """Return abnormal / normal decision for a single image."""


class BlackPixelClassifier:
    def __init__(
        self,
        dark_threshold: int = 32,
        dark_ratio_threshold: float = 0.58,
        mean_brightness_threshold: float = 50.0,
    ):
        self.dark_threshold = dark_threshold
        self.dark_ratio_threshold = dark_ratio_threshold
        self.mean_brightness_threshold = mean_brightness_threshold

    def classify(self, image_path: Path) -> ClassificationResult:
        with Image.open(image_path) as image:
            grayscale = image.convert("L")
            stat = ImageStat.Stat(grayscale)
            histogram = grayscale.histogram()

        total_pixels = sum(histogram) or 1
        dark_pixels = sum(histogram[: self.dark_threshold + 1])
        dark_ratio = dark_pixels / total_pixels
        mean_brightness = stat.mean[0]
        is_abnormal = (
            dark_ratio >= self.dark_ratio_threshold
            and mean_brightness <= self.mean_brightness_threshold
        )

        return ClassificationResult(
            image_path=image_path,
            label="abnormal" if is_abnormal else "normal",
            score=dark_ratio,
            reason=(
                f"dark_ratio={dark_ratio:.4f}, "
                f"mean_brightness={mean_brightness:.2f}, "
                f"dark_threshold={self.dark_threshold}, "
                f"ratio_threshold={self.dark_ratio_threshold:.2f}, "
                f"mean_threshold={self.mean_brightness_threshold:.2f}"
            ),
        )


class PlaceholderClassifier(BlackPixelClassifier):
    """Backwards-compatible alias for the initial classifier implementation."""
