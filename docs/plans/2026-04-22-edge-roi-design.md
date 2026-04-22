# Edge ROI Design

## Goal

Detect cover occlusion from the lower-left frame edge using a fixed ROI that matches the cover travel path.

## ROI Definition

Use image coordinates with the origin at the lower-left corner for product reasoning. In PIL/image-array coordinates the origin is upper-left, so the same line is represented as:

- Left edge point: `(0, 0.30 * height)` in upper-left coordinates, equivalent to 70% vertical height from the lower-left origin.
- Bottom edge point: `(0.40 * width, height - 1)` in upper-left coordinates.
- ROI polygon: lower-left triangle bounded by the left edge, bottom edge, and the line between those two points.

## Classification

Keep the current score model:

- Convert image to grayscale.
- Resize to the existing analysis size.
- Collect pixels inside the new ROI.
- Compute mean brightness.
- Compute `score = 1.0 - mean_brightness / 255.0`.
- Classify as `abnormal` when `score >= score_threshold`.

Default threshold remains `0.80`.

## Settings

Replace the old offset-style setting with explicit endpoint settings:

- `BLOCK_DETECT_ROI_LEFT_Y_RATIO=0.30`
- `BLOCK_DETECT_ROI_BOTTOM_X_RATIO=0.40`

The classifier should clamp both ratios to valid image ranges.

## UI

Update the GUI preview overlay to draw the same new lower-left triangular ROI. Runtime controls should expose the two endpoint ratios instead of the old line offset ratio.

## Tests

Add tests for:

- Default settings expose the new ROI endpoint ratios.
- The classifier marks lower-left edge occlusion as abnormal.
- The classifier keeps dark regions outside the lower-left edge ROI normal.
- Runtime overrides and reports persist the new settings.
