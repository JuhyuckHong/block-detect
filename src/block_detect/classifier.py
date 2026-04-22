from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image


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
    ANALYSIS_SIZE = (160, 90)

    def __init__(
        self,
        dark_threshold: int = 32,
        score_threshold: float = 0.80,
        roi_left_y_ratio: float = 0.30,
        roi_bottom_x_ratio: float = 0.40,
    ):
        self.dark_threshold = dark_threshold
        self.score_threshold = score_threshold
        self.roi_left_y_ratio = max(0.0, min(1.0, roi_left_y_ratio))
        self.roi_bottom_x_ratio = max(0.0, min(1.0, roi_bottom_x_ratio))

    def _roi_left_y_pixels(self, height: int) -> int:
        if height <= 1:
            return 0
        return int(round((height - 1) * self.roi_left_y_ratio))

    def _roi_bottom_x_pixels(self, width: int) -> int:
        if width <= 1:
            return 0
        return int(round((width - 1) * self.roi_bottom_x_ratio))

    def _is_in_lower_left_region(self, x: int, y: int, width: int, height: int) -> bool:
        if width <= 1 or height <= 1:
            return True
        left_y = self._roi_left_y_pixels(height)
        bottom_x = self._roi_bottom_x_pixels(width)
        if bottom_x <= 0:
            return x == 0 and y >= left_y
        if x > bottom_x:
            return False
        bottom_y = height - 1
        return (y - left_y) * bottom_x >= x * (bottom_y - left_y)

    def _region_pixels(self, grayscale: Image.Image) -> list[int]:
        width, height = grayscale.size
        pixels = list(grayscale.getdata())
        region_values: list[int] = []
        for y in range(height):
            row_offset = y * width
            for x in range(width):
                if self._is_in_lower_left_region(x, y, width, height):
                    region_values.append(pixels[row_offset + x])
        return region_values

    def classify(self, image_path: Path) -> ClassificationResult:
        with Image.open(image_path) as image:
            grayscale = image.convert("L").resize(self.ANALYSIS_SIZE)
            region_pixels = self._region_pixels(grayscale)

        total_pixels = len(region_pixels) or 1
        mean_brightness = sum(region_pixels) / total_pixels
        score = 1.0 - (mean_brightness / 255.0)
        is_abnormal = score >= self.score_threshold

        return ClassificationResult(
            image_path=image_path,
            label="abnormal" if is_abnormal else "normal",
            score=score,
            reason=(
                f"score={score:.4f}, "
                f"mean_brightness={mean_brightness:.2f}, "
                f"score_threshold={self.score_threshold:.2f}, "
                f"roi_left_y_ratio={self.roi_left_y_ratio:.2f}, "
                f"roi_bottom_x_ratio={self.roi_bottom_x_ratio:.2f}, "
                "roi=lower_left_edge_triangle"
            ),
        )


class PlaceholderClassifier(BlackPixelClassifier):
    """Backwards-compatible alias for the initial classifier implementation."""
