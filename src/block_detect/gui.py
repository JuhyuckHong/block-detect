from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import hashlib
import math
from pathlib import Path
from queue import Empty, Queue
import re
import threading
import traceback
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageOps, ImageTk

from .classifier import ClassificationResult
from .config import Settings, load_settings
from .pipeline import PipelineRunResult, build_pipeline, load_saved_run


DEFAULT_GUI_DATE = "2026-03-30"
DEFAULT_GUI_DROPBOX_PATH = "/test_042_new/2026-03-30"
THUMBNAIL_SIZE = (240, 180)
SELECTED_PREVIEW_SIZE = (360, 360)
GALLERY_BATCH_SIZE = 8
TIME_PATTERN = re.compile(r"(?P<hour>\d{2})-(?P<minute>\d{2})-(?P<second>\d{2})$")


def is_blocked_result(result: ClassificationResult) -> bool:
    return result.label in {"abnormal", "blocked"}


def blocked_results(results: list[ClassificationResult]) -> list[ClassificationResult]:
    return [result for result in results if is_blocked_result(result)]


def format_ratio(part: int, whole: int) -> str:
    if whole <= 0:
        return "0.0% (0/0)"
    return f"{(part / whole) * 100:.1f}% ({part}/{whole})"


def render_preview_image(
    image: Image.Image,
    size: tuple[int, int],
    show_roi_overlay: bool = False,
    roi_line_offset_ratio: float = 0.12,
) -> Image.Image:
    preview = ImageOps.contain(image.convert("RGBA"), size)
    if not show_roi_overlay:
        return preview.convert("RGB")

    width, height = preview.size
    offset_pixels = int(round(max(0.0, min(0.95, roi_line_offset_ratio)) * max(0, height - 1)))
    if height <= 1 or width <= 1:
        offset_pixels = 0
    bottom_intersection_x = width - 1
    if height > 1:
        bottom_intersection_x = int(
            round(((height - 1 - offset_pixels) * (width - 1)) / (height - 1))
        )
        bottom_intersection_x = max(0, min(width - 1, bottom_intersection_x))
    overlay = Image.new("RGBA", preview.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.polygon(
        [
            (0, 0),
            (width - 1, 0),
            (width - 1, height - 1),
            (bottom_intersection_x, height - 1),
            (0, offset_pixels),
        ],
        fill=(255, 80, 80, 72),
    )
    draw.line(
        (0, offset_pixels, bottom_intersection_x, height - 1),
        fill=(255, 220, 0, 255),
        width=3,
    )
    return Image.alpha_composite(preview, overlay).convert("RGB")


def thumbnail_cache_key(image_path: Path, size: tuple[int, int]) -> str:
    stat = image_path.stat()
    digest = hashlib.sha256(
        f"{image_path.resolve()}|{stat.st_mtime_ns}|{stat.st_size}|{size[0]}x{size[1]}".encode("utf-8")
    ).hexdigest()
    return digest


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


def calculate_hour_tick_step(plot_width: int, hour_span: int, min_label_spacing: int = 24) -> int:
    if hour_span <= 0:
        return 1
    pixels_per_hour = max(1.0, plot_width / hour_span)
    return max(1, math.ceil(min_label_spacing / pixels_per_hour))


def apply_runtime_overrides(
    settings: Settings,
    *,
    download_workers: int,
    classify_workers: int,
    score_threshold: float,
    roi_line_offset_ratio: float,
) -> Settings:
    return replace(
        settings,
        download_workers=download_workers,
        classify_workers=classify_workers,
        score_threshold=score_threshold,
        roi_line_offset_ratio=roi_line_offset_ratio,
    )


class DetectionGui:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Block Detect")
        self.root.geometry("1380x900")
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.report_callback_exception = self._handle_tk_exception

        self.settings = load_settings()
        self.result_queue: Queue[tuple[str, object]] = Queue()
        self.worker_thread: threading.Thread | None = None
        self.thumbnail_refs: list[ImageTk.PhotoImage] = []
        self.selected_preview_ref: ImageTk.PhotoImage | None = None
        self.results_by_item_id: dict[str, ClassificationResult] = {}
        self.latest_results: list[ClassificationResult] = []
        self.applied_roi_line_offset_ratio = self.settings.roi_line_offset_ratio
        self.gallery_after_id: str | None = None
        self.pending_blocked_results: list[ClassificationResult] = []

        self.dropbox_path_var = tk.StringVar(value=DEFAULT_GUI_DROPBOX_PATH)
        self.download_workers_var = tk.StringVar(value=str(self.settings.download_workers))
        self.classify_workers_var = tk.StringVar(value=str(self.settings.classify_workers))
        self.score_threshold_var = tk.DoubleVar(value=self.settings.score_threshold)
        self.roi_line_offset_ratio_var = tk.DoubleVar(value=self.settings.roi_line_offset_ratio)
        self.score_threshold_label_var = tk.StringVar(value=f"{self.settings.score_threshold:.3f}")
        self.roi_offset_label_var = tk.StringVar(
            value=f"{self.settings.roi_line_offset_ratio:.3f}"
        )
        self.status_var = tk.StringVar(value="Ready.")
        self.progress_label_var = tk.StringVar(value="Idle")
        self.remote_path_var = tk.StringVar(value="-")
        self.report_path_var = tk.StringVar(value="-")

        self.summary_vars = {
            "processed": tk.StringVar(value="0"),
            "blocked": tk.StringVar(value="0"),
            "normal": tk.StringVar(value="0"),
            "unknown": tk.StringVar(value="0"),
            "ratio": tk.StringVar(value="0.0% (0/0)"),
        }

        self._build_layout()
        self.dropbox_path_var.trace_add("write", self._on_inputs_changed)
        self.score_threshold_var.trace_add("write", self._on_visual_threshold_changed)
        self.roi_line_offset_ratio_var.trace_add("write", self._on_visual_threshold_changed)
        self.root.after(100, self._poll_result_queue)
        self._update_action_button_state()

    def _safe_score_threshold(self) -> float:
        try:
            return float(self.score_threshold_var.get())
        except (tk.TclError, ValueError):
            return float(self.settings.score_threshold)

    def _safe_roi_line_offset_ratio(self) -> float:
        try:
            return float(self.roi_line_offset_ratio_var.get())
        except (tk.TclError, ValueError):
            return float(self.applied_roi_line_offset_ratio)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=3)
        self.root.rowconfigure(3, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        controls.columnconfigure(5, weight=1)
        controls.columnconfigure(9, weight=1)

        ttk.Label(controls, text="Dropbox Path").grid(row=0, column=0, sticky="w", padx=(0, 8))
        ttk.Entry(controls, textvariable=self.dropbox_path_var).grid(row=0, column=1, columnspan=5, sticky="ew")
        self.run_button = ttk.Button(controls, text="Run Detection", command=self.start_detection)
        self.run_button.grid(row=0, column=6, sticky="ew", padx=(12, 6))
        self.apply_threshold_button = ttk.Button(
            controls,
            text="Apply Threshold",
            command=self.apply_threshold_to_local,
        )
        self.apply_threshold_button.grid(row=0, column=7, sticky="ew", padx=6)
        self.load_button = ttk.Button(controls, text="Load Existing", command=self.load_existing)
        self.load_button.grid(row=0, column=8, sticky="ew", padx=6)

        ttk.Label(controls, text="Download Workers").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(10, 0))
        ttk.Entry(controls, textvariable=self.download_workers_var, width=8).grid(row=1, column=1, sticky="w", pady=(10, 0))

        ttk.Label(controls, text="Classify Workers").grid(row=1, column=2, sticky="w", padx=(16, 8), pady=(10, 0))
        ttk.Entry(controls, textvariable=self.classify_workers_var, width=8).grid(row=1, column=3, sticky="w", pady=(10, 0))

        ttk.Label(controls, text="Score Threshold").grid(row=1, column=4, sticky="w", padx=(16, 8), pady=(10, 0))
        ttk.Scale(
            controls,
            from_=0.0,
            to=1.0,
            variable=self.score_threshold_var,
            orient="horizontal",
        ).grid(row=1, column=5, sticky="ew", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.score_threshold_var, width=8).grid(row=1, column=6, sticky="w", pady=(10, 0))

        ttk.Label(controls, text="ROI Offset").grid(row=1, column=7, sticky="w", padx=(16, 8), pady=(10, 0))
        ttk.Scale(
            controls,
            from_=0.0,
            to=0.35,
            variable=self.roi_line_offset_ratio_var,
            orient="horizontal",
        ).grid(row=1, column=8, sticky="ew", pady=(10, 0))
        ttk.Entry(controls, textvariable=self.roi_line_offset_ratio_var, width=8).grid(row=1, column=9, sticky="w", pady=(10, 0))

        progress = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        progress.grid(row=1, column=0, sticky="ew")
        progress.columnconfigure(0, weight=1)

        self.progress_bar = ttk.Progressbar(progress, mode="determinate", maximum=1, value=0)
        self.progress_bar.grid(row=0, column=0, sticky="ew")
        ttk.Label(progress, textvariable=self.progress_label_var).grid(row=0, column=1, sticky="w", padx=(12, 0))
        ttk.Label(progress, textvariable=self.status_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        main_area = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        main_area.grid(row=2, column=0, sticky="nsew")
        main_area.columnconfigure(0, weight=3)
        main_area.columnconfigure(1, weight=2)
        main_area.rowconfigure(1, weight=1)

        summary = ttk.LabelFrame(main_area, text="Summary", padding=12)
        summary.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        for idx in range(5):
            summary.columnconfigure(idx, weight=1)

        self._create_summary_value(summary, 0, "Processed", self.summary_vars["processed"])
        self._create_summary_value(summary, 1, "Blocked", self.summary_vars["blocked"])
        self._create_summary_value(summary, 2, "Normal", self.summary_vars["normal"])
        self._create_summary_value(summary, 3, "Unknown", self.summary_vars["unknown"])
        self._create_summary_value(summary, 4, "Blocked Ratio", self.summary_vars["ratio"])

        chart_frame = ttk.LabelFrame(main_area, text="Score Plot", padding=8)
        chart_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        chart_frame.columnconfigure(0, weight=1)
        chart_frame.rowconfigure(0, weight=1)
        self.score_plot_canvas = tk.Canvas(
            chart_frame,
            width=320,
            highlightthickness=0,
            background="white",
        )
        self.score_plot_canvas.grid(row=0, column=0, sticky="nsew")
        self.score_plot_canvas.bind("<Configure>", self._redraw_score_plot)

        results_preview_area = ttk.Frame(main_area)
        results_preview_area.grid(row=1, column=0, sticky="nsew", padx=(0, 8), pady=(8, 0))
        results_preview_area.columnconfigure(0, weight=1)
        results_preview_area.columnconfigure(1, weight=1)
        results_preview_area.rowconfigure(0, weight=1)

        left_panel = ttk.LabelFrame(results_preview_area, text="All Results", padding=8)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left_panel.columnconfigure(0, weight=1)
        left_panel.rowconfigure(0, weight=1)

        columns = ("file", "label", "score")
        self.results_tree = ttk.Treeview(left_panel, columns=columns, show="headings", height=8)
        self.results_tree.heading("file", text="File")
        self.results_tree.heading("label", text="Label")
        self.results_tree.heading("score", text="Score")
        self.results_tree.column("file", width=180, anchor="w")
        self.results_tree.column("label", width=90, anchor="center")
        self.results_tree.column("score", width=90, anchor="e")
        self.results_tree.grid(row=0, column=0, sticky="nsew")
        self.results_tree.bind("<<TreeviewSelect>>", self._on_result_selected)

        tree_scroll = ttk.Scrollbar(left_panel, orient="vertical", command=self.results_tree.yview)
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.results_tree.configure(yscrollcommand=tree_scroll.set)

        preview_panel = ttk.LabelFrame(results_preview_area, text="Selected Preview", padding=8)
        preview_panel.grid(row=0, column=1, sticky="nsew")
        preview_panel.columnconfigure(0, weight=1)
        preview_panel.rowconfigure(0, weight=1)

        self.preview_image_label = ttk.Label(preview_panel, text="Select a row to preview.", anchor="center")
        self.preview_image_label.grid(row=0, column=0, sticky="nsew")

        preview_meta = ttk.Frame(preview_panel)
        preview_meta.grid(row=1, column=0, sticky="ew", pady=(10, 0))
        preview_meta.columnconfigure(0, weight=1)

        self.preview_name_var = tk.StringVar(value="-")
        ttk.Label(
            preview_meta,
            textvariable=self.preview_name_var,
            anchor="center",
            justify="center",
            wraplength=320,
        ).grid(row=0, column=0, sticky="ew")

        bottom_area = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        bottom_area.grid(row=3, column=0, sticky="nsew")
        bottom_area.columnconfigure(0, weight=1)
        bottom_area.rowconfigure(0, weight=1)

        right_panel = ttk.LabelFrame(bottom_area, text="Blocked Images", padding=8)
        right_panel.grid(row=0, column=0, sticky="nsew")
        right_panel.columnconfigure(0, weight=1)
        right_panel.rowconfigure(0, weight=1)

        self.blocked_canvas = tk.Canvas(right_panel, highlightthickness=0, height=260)
        self.blocked_canvas.grid(row=0, column=0, sticky="nsew")

        blocked_scroll = ttk.Scrollbar(right_panel, orient="horizontal", command=self.blocked_canvas.xview)
        blocked_scroll.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        self.blocked_canvas.configure(xscrollcommand=blocked_scroll.set)

        self.blocked_frame = ttk.Frame(self.blocked_canvas)
        self.blocked_window = self.blocked_canvas.create_window((0, 0), window=self.blocked_frame, anchor="nw")
        self.blocked_frame.bind("<Configure>", self._sync_blocked_scroll_region)
        self.blocked_canvas.bind("<Configure>", self._resize_blocked_window)

    def _create_summary_value(
        self,
        parent: ttk.LabelFrame,
        column: int,
        title: str,
        value_var: tk.StringVar,
    ) -> None:
        frame = ttk.Frame(parent, padding=6)
        frame.grid(row=0, column=column, sticky="ew")
        frame.columnconfigure(0, weight=1)
        ttk.Label(frame, text=title).grid(row=0, column=0, sticky="w")
        ttk.Label(frame, textvariable=value_var, font=("Segoe UI", 14, "bold")).grid(
            row=1,
            column=0,
            sticky="w",
            pady=(6, 0),
        )

    def start_detection(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.status_var.set("Detection is already running.")
            return

        try:
            day = self._selected_day()
            if not day:
                raise ValueError("Dropbox Path must end with a date-like folder name.")
            download_workers = max(1, int(self.download_workers_var.get().strip()))
            classify_workers = max(1, int(self.classify_workers_var.get().strip()))
            score_threshold = float(self.score_threshold_var.get())
            roi_line_offset_ratio = float(self.roi_line_offset_ratio_var.get())
        except (ValueError, tk.TclError) as exc:
            self.status_var.set(str(exc))
            return

        remote_path_override = self.dropbox_path_var.get().strip() or None
        self.status_var.set("Running detection...")
        self.progress_label_var.set("Preparing...")
        self.progress_bar.configure(maximum=1, value=0)
        self.remote_path_var.set("-")
        self.report_path_var.set("-")
        self._clear_results()
        self.run_button.state(["disabled"])

        self.worker_thread = threading.Thread(
            target=self._run_detection_worker,
            args=(
                day,
                remote_path_override,
                download_workers,
                classify_workers,
                score_threshold,
                roi_line_offset_ratio,
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def _saved_run_paths(self) -> tuple[Path, Path]:
        day = self._selected_day()
        return (
            self.settings.reports_dir / f"{day}.json",
            self.settings.inbox_dir / day,
        )

    def _write_error_log(self, *, context: str, error_text: str) -> Path:
        logs_dir = self.settings.workspace_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        log_path = logs_dir / f"block-detect-error-{timestamp}.log"
        dropbox_path = self.dropbox_path_var.get().strip()
        selected_day = self._selected_day()
        payload = "\n".join(
            [
                f"Timestamp: {datetime.now().isoformat(timespec='seconds')}",
                f"Context: {context}",
                f"Selected Day: {selected_day or '-'}",
                f"Dropbox Path: {dropbox_path or '-'}",
                "",
                error_text.strip(),
                "",
            ]
        )
        log_path.write_text(payload, encoding="utf-8")
        return log_path

    def _capture_error(self, *, context: str, error_text: str) -> dict[str, str]:
        try:
            log_path = self._write_error_log(context=context, error_text=error_text)
        except Exception:
            return {
                "message": (
                    f"{error_text.strip()}\n\n"
                    "[error-log-write-failed]\n"
                    f"{traceback.format_exc().strip()}"
                )
            }
        return {
            "message": error_text.strip(),
            "log_path": str(log_path),
        }

    def _format_error_status(self, payload: object) -> str:
        if isinstance(payload, dict):
            message = str(payload.get("message", "")).strip()
            log_path = str(payload.get("log_path", "")).strip()
            if log_path:
                return f"{message}\n\nSaved error log: {log_path}"
            return message
        return str(payload).strip()

    def _selected_day(self) -> str:
        path_value = self.dropbox_path_var.get().strip().strip("/")
        if not path_value:
            return ""
        return Path(path_value).name

    def _update_action_button_state(self) -> None:
        report_path, inbox_dir = self._saved_run_paths()
        if report_path.exists() and inbox_dir.exists():
            self.load_button.state(["!disabled"])
        else:
            self.load_button.state(["disabled"])
        if inbox_dir.exists():
            self.apply_threshold_button.state(["!disabled"])
        else:
            self.apply_threshold_button.state(["disabled"])

    def _on_inputs_changed(self, *_args: object) -> None:
        self._update_action_button_state()

    def _on_visual_threshold_changed(self, *_args: object) -> None:
        self._redraw_score_plot()

    def load_existing(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.status_var.set("Detection is already running.")
            return

        day = self._selected_day()
        report_path, inbox_dir = self._saved_run_paths()
        if not report_path.exists() or not inbox_dir.exists():
            self.status_var.set("No saved download/report found for this date.")
            self._update_action_button_state()
            return

        try:
            run_result = load_saved_run(day, settings=self.settings)
            self._apply_saved_classification_settings(run_result.classification_settings)
            self._clear_results()
            self._apply_run_result(run_result)
        except KeyboardInterrupt:
            self.status_var.set("Loading saved results was interrupted.")
            self.progress_label_var.set("Interrupted")
            self.run_button.state(["!disabled"])
            self._update_action_button_state()
            return
        except Exception as exc:
            payload = self._capture_error(
                context="load_existing",
                error_text=traceback.format_exc(),
            )
            self.status_var.set(self._format_error_status(payload))
            self.progress_label_var.set("Failed")
            self.run_button.state(["!disabled"])
            self._update_action_button_state()
            return

        self.status_var.set(f"Loaded saved results for {day}.")
        self._update_action_button_state()

    def _apply_saved_classification_settings(self, classification_settings: dict[str, float]) -> None:
        if "score_threshold" in classification_settings:
            self.score_threshold_var.set(float(classification_settings["score_threshold"]))
        if "roi_line_offset_ratio" in classification_settings:
            self.roi_line_offset_ratio_var.set(float(classification_settings["roi_line_offset_ratio"]))

    def apply_threshold_to_local(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            self.status_var.set("Detection is already running.")
            return

        try:
            day = self._selected_day()
            if not day:
                raise ValueError("Dropbox Path must end with a date-like folder name.")
            classify_workers = max(1, int(self.classify_workers_var.get().strip()))
            score_threshold = float(self.score_threshold_var.get())
            roi_line_offset_ratio = float(self.roi_line_offset_ratio_var.get())
        except (ValueError, tk.TclError) as exc:
            self.status_var.set(str(exc))
            return

        _, inbox_dir = self._saved_run_paths()
        if not inbox_dir.exists():
            self.status_var.set("No downloaded images found for this date.")
            self._update_action_button_state()
            return

        remote_path_override = self.dropbox_path_var.get().strip() or None
        self.status_var.set("Applying threshold to local images...")
        self.progress_label_var.set("Preparing...")
        self.progress_bar.configure(maximum=1, value=0)
        self.remote_path_var.set("-")
        self.report_path_var.set("-")
        self._clear_results()
        self.run_button.state(["disabled"])
        self.apply_threshold_button.state(["disabled"])
        self.load_button.state(["disabled"])

        self.worker_thread = threading.Thread(
            target=self._run_local_reclassify_worker,
            args=(
                day,
                remote_path_override,
                classify_workers,
                score_threshold,
                roi_line_offset_ratio,
            ),
            daemon=True,
        )
        self.worker_thread.start()

    def _run_detection_worker(
        self,
        day: str,
        remote_path_override: str | None,
        download_workers: int,
        classify_workers: int,
        score_threshold: float,
        roi_line_offset_ratio: float,
    ) -> None:
        try:
            settings = apply_runtime_overrides(
                load_settings(),
                download_workers=download_workers,
                classify_workers=classify_workers,
                score_threshold=score_threshold,
                roi_line_offset_ratio=roi_line_offset_ratio,
            )
            pipeline = build_pipeline(settings=settings)
            result = pipeline.run_day_with_details(
                day,
                remote_day_path=remote_path_override,
                progress_callback=self._emit_progress,
            )
            self.result_queue.put(("success", result))
        except Exception:
            self.result_queue.put(
                (
                    "error",
                    self._capture_error(
                        context="_run_detection_worker",
                        error_text=traceback.format_exc(),
                    ),
                )
            )

    def _run_local_reclassify_worker(
        self,
        day: str,
        remote_path_override: str | None,
        classify_workers: int,
        score_threshold: float,
        roi_line_offset_ratio: float,
    ) -> None:
        try:
            settings = apply_runtime_overrides(
                load_settings(),
                download_workers=self.settings.download_workers,
                classify_workers=classify_workers,
                score_threshold=score_threshold,
                roi_line_offset_ratio=roi_line_offset_ratio,
            )
            pipeline = build_pipeline(settings=settings)
            result = pipeline.run_local_day_with_details(
                day,
                remote_day_path=remote_path_override,
                progress_callback=self._emit_progress,
            )
            self.result_queue.put(("success", result))
        except Exception:
            self.result_queue.put(
                (
                    "error",
                    self._capture_error(
                        context="_run_local_reclassify_worker",
                        error_text=traceback.format_exc(),
                    ),
                )
            )

    def _handle_tk_exception(
        self,
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_traceback: object,
    ) -> None:
        payload = self._capture_error(
            context="tk_callback",
            error_text="".join(traceback.format_exception(exc_type, exc_value, exc_traceback)),
        )
        self.status_var.set(self._format_error_status(payload))
        self.progress_label_var.set("Failed")
        self.run_button.state(["!disabled"])
        self._update_action_button_state()

    def _emit_progress(self, stage: str, current: int, total: int) -> None:
        self.result_queue.put(("progress", (stage, current, total)))

    def _poll_result_queue(self) -> None:
        try:
            while True:
                message_type, payload = self.result_queue.get_nowait()
                if message_type == "success":
                    self._apply_run_result(payload)  # type: ignore[arg-type]
                elif message_type == "progress":
                    stage, current, total = payload  # type: ignore[misc]
                    self._apply_progress(stage, current, total)
                else:
                    self.status_var.set(self._format_error_status(payload))
                    self.progress_label_var.set("Failed")
                    self.run_button.state(["!disabled"])
                    self._update_action_button_state()
        except Empty:
            pass
        finally:
            try:
                self.root.after(100, self._poll_result_queue)
            except tk.TclError:
                return

    def _apply_run_result(self, run_result: PipelineRunResult) -> None:
        blocked = blocked_results(run_result.results)
        summary = run_result.summary

        self.summary_vars["processed"].set(str(summary.processed_count))
        self.summary_vars["blocked"].set(str(summary.abnormal_count))
        self.summary_vars["normal"].set(str(summary.normal_count))
        self.summary_vars["unknown"].set(str(summary.unknown_count))
        self.summary_vars["ratio"].set(format_ratio(summary.abnormal_count, summary.processed_count))

        self.remote_path_var.set(run_result.remote_day_path)
        self.report_path_var.set(str(run_result.report_path))
        self.status_var.set(f"Completed. Loaded {summary.processed_count} images.")
        self.progress_bar.configure(maximum=max(1, summary.processed_count), value=max(1, summary.processed_count))
        self.progress_label_var.set(f"Completed {summary.processed_count}/{summary.processed_count}")
        self.latest_results = run_result.results
        self.applied_roi_line_offset_ratio = self._safe_roi_line_offset_ratio()

        self._populate_results_tree(run_result.results)
        self._populate_blocked_gallery(blocked)
        self._redraw_score_plot()
        self.run_button.state(["!disabled"])
        self._update_action_button_state()

    def _apply_progress(self, stage: str, current: int, total: int) -> None:
        stage_title = "Downloading" if stage == "download" else "Analyzing"
        bar_total = max(1, total)
        self.progress_bar.configure(maximum=bar_total, value=min(current, bar_total))
        self.progress_label_var.set(f"{stage_title} {current}/{total}")
        if total > 0:
            self.status_var.set(f"{stage_title} images... ({current}/{total})")
        else:
            self.status_var.set(f"{stage_title} images...")

    def _clear_results(self) -> None:
        if self.gallery_after_id is not None:
            try:
                self.root.after_cancel(self.gallery_after_id)
            except tk.TclError:
                pass
            self.gallery_after_id = None
        self.pending_blocked_results = []
        self.thumbnail_refs.clear()
        self.selected_preview_ref = None
        self.results_by_item_id.clear()

        for item_id in self.results_tree.get_children():
            self.results_tree.delete(item_id)

        for child in self.blocked_frame.winfo_children():
            child.destroy()

        self.summary_vars["processed"].set("0")
        self.summary_vars["blocked"].set("0")
        self.summary_vars["normal"].set("0")
        self.summary_vars["unknown"].set("0")
        self.summary_vars["ratio"].set("0.0% (0/0)")
        self.progress_bar.configure(maximum=1, value=0)
        self.preview_image_label.configure(image="", text="Select a row to preview.")
        self.preview_name_var.set("-")
        self.latest_results = []
        self._redraw_score_plot()

    def _populate_results_tree(self, results: list[ClassificationResult]) -> None:
        for result in results:
            label = "blocked" if is_blocked_result(result) else result.label
            item_id = self.results_tree.insert(
                "",
                "end",
                values=(
                    result.image_path.name,
                    label,
                    f"{result.score:.4f}",
                ),
            )
            self.results_by_item_id[item_id] = result

        children = self.results_tree.get_children()
        if children:
            first_item = children[0]
            self.results_tree.selection_set(first_item)
            self.results_tree.focus(first_item)
            self._show_result_preview(first_item)

    def _populate_blocked_gallery(self, results: list[ClassificationResult]) -> None:
        if not results:
            ttk.Label(
                self.blocked_frame,
                text="No blocked images found for this run.",
                padding=12,
            ).grid(row=0, column=0, sticky="w")
            return
        ttk.Label(
            self.blocked_frame,
            text=f"Loading {len(results)} blocked thumbnails...",
            padding=12,
        ).grid(row=0, column=0, sticky="w")
        self.pending_blocked_results = list(results)
        self.gallery_after_id = self.root.after(1, self._render_blocked_gallery_batch)

    def _render_blocked_gallery_batch(self) -> None:
        if not self.pending_blocked_results:
            self.gallery_after_id = None
            return

        existing_cards = len(
            [child for child in self.blocked_frame.winfo_children() if isinstance(child, ttk.Frame)]
        )
        if existing_cards == 0:
            for child in self.blocked_frame.winfo_children():
                child.destroy()

        batch = self.pending_blocked_results[:GALLERY_BATCH_SIZE]
        self.pending_blocked_results = self.pending_blocked_results[GALLERY_BATCH_SIZE:]

        for offset, result in enumerate(batch):
            index = existing_cards + offset
            card = ttk.Frame(self.blocked_frame, padding=8, relief="ridge")
            card.grid(row=0, column=index, sticky="ns", padx=6, pady=6)

            image_label = ttk.Label(card)
            image_label.grid(row=0, column=0, sticky="nsew")
            image = self._load_gallery_thumbnail(result.image_path)
            if image is not None:
                image_label.configure(image=image)
                self.thumbnail_refs.append(image)
            else:
                image_label.configure(text="Preview unavailable", width=32)

            ttk.Label(card, text=result.image_path.name).grid(row=1, column=0, sticky="w", pady=(8, 0))
            ttk.Label(card, text=f"Score: {result.score:.4f}").grid(row=2, column=0, sticky="w")

        self.gallery_after_id = None
        if self.pending_blocked_results:
            self.gallery_after_id = self.root.after(1, self._render_blocked_gallery_batch)

    def _load_thumbnail(
        self,
        image_path: Path,
        size: tuple[int, int] = THUMBNAIL_SIZE,
        show_roi_overlay: bool = False,
    ) -> ImageTk.PhotoImage | None:
        try:
            with Image.open(image_path) as image:
                preview = render_preview_image(
                    image,
                    size,
                    show_roi_overlay=show_roi_overlay,
                    roi_line_offset_ratio=self.applied_roi_line_offset_ratio,
                )
            return ImageTk.PhotoImage(preview)
        except Exception:
            return None

    def _thumbnail_cache_path(self, image_path: Path, size: tuple[int, int]) -> Path:
        cache_key = thumbnail_cache_key(image_path, size)
        return self.settings.thumbnail_cache_dir / image_path.parent.name / f"{cache_key}.jpg"

    def _load_gallery_thumbnail(self, image_path: Path) -> ImageTk.PhotoImage | None:
        cache_path = self._thumbnail_cache_path(image_path, THUMBNAIL_SIZE)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            if not cache_path.exists():
                with Image.open(image_path) as image:
                    preview = render_preview_image(image, THUMBNAIL_SIZE, show_roi_overlay=False)
                preview.save(cache_path, format="JPEG", quality=85)
            with Image.open(cache_path) as cached_image:
                return ImageTk.PhotoImage(cached_image.convert("RGB"))
        except Exception:
            return self._load_thumbnail(image_path, size=THUMBNAIL_SIZE, show_roi_overlay=False)

    def _on_result_selected(self, _event: tk.Event) -> None:
        selection = self.results_tree.selection()
        if not selection:
            return
        self._show_result_preview(selection[0])

    def _show_result_preview(self, item_id: str) -> None:
        result = self.results_by_item_id.get(item_id)
        if result is None:
            return

        self.preview_name_var.set(result.image_path.name)

        image = self._load_thumbnail(
            result.image_path,
            size=SELECTED_PREVIEW_SIZE,
            show_roi_overlay=True,
        )
        if image is None:
            self.selected_preview_ref = None
            self.preview_image_label.configure(image="", text="Preview unavailable")
            return

        self.selected_preview_ref = image
        self.preview_image_label.configure(image=image, text="")

    def _redraw_score_plot(self, _event: tk.Event | None = None) -> None:
        canvas = self.score_plot_canvas
        canvas.delete("all")

        width = max(canvas.winfo_width(), 520)
        height = max(canvas.winfo_height(), 220)
        left = 40
        top = 14
        right = width - 14
        bottom = height - 28

        threshold = self._safe_score_threshold()

        if not self.latest_results:
            canvas.create_rectangle(left, top, right, bottom, outline="#c9ced6")
            canvas.create_text(left - 10, top, text="1.0", anchor="e", fill="#55606f")
            canvas.create_text(left - 10, (top + bottom) / 2, text="0.5", anchor="e", fill="#55606f")
            canvas.create_text(left - 10, bottom, text="0.0", anchor="e", fill="#55606f")
            canvas.create_text((left + right) / 2, height - 6, text="Image Order", anchor="s", fill="#55606f")
            canvas.create_text(14, (top + bottom) / 2, text="Score", angle=90, fill="#55606f")
            canvas.create_text(
                (left + right) / 2,
                (top + bottom) / 2,
                text="Run, load, or reclassify to plot scores.",
                fill="#66717f",
            )
            return

        scores = [max(0.0, min(1.0, result.score)) for result in self.latest_results]
        min_value = min(scores + [threshold])
        max_value = max(scores + [threshold])
        if math.isclose(min_value, max_value, abs_tol=1e-9):
            min_value = max(0.0, min_value - 0.1)
            max_value = min(1.0, max_value + 0.1)
        padding = max(0.02, (max_value - min_value) * 0.08)
        view_min = max(0.0, math.floor((min_value - padding) * 10) / 10)
        view_max = min(1.0, math.ceil((max_value + padding) * 10) / 10)
        if math.isclose(view_min, view_max, abs_tol=1e-9):
            view_max = min(1.0, view_min + 0.1)
        scale = bottom - top

        def y_for(value: float) -> float:
            if math.isclose(view_min, view_max, abs_tol=1e-9):
                return bottom
            normalized = (value - view_min) / (view_max - view_min)
            return bottom - normalized * scale

        canvas.create_rectangle(left, top, right, bottom, outline="#c9ced6")
        minor_grid_value = math.ceil(view_min * 100) / 100
        while minor_grid_value <= view_max + 1e-9:
            if not math.isclose(minor_grid_value * 10, round(minor_grid_value * 10), abs_tol=1e-9):
                y = y_for(minor_grid_value)
                canvas.create_line(left, y, right, y, fill="#f1f3f6", dash=(1, 5))
            minor_grid_value += 0.01

        grid_value = math.ceil(view_min * 10) / 10
        while grid_value <= view_max + 1e-9:
            y = y_for(grid_value)
            canvas.create_line(left, y, right, y, fill="#e3e7ec", dash=(2, 4))
            canvas.create_text(left - 10, y, text=f"{grid_value:.1f}", anchor="e", fill="#55606f")
            grid_value += 0.1
        canvas.create_text(14, (top + bottom) / 2, text="Score", angle=90, fill="#55606f")

        threshold_y = y_for(threshold)
        canvas.create_line(left, threshold_y, right, threshold_y, fill="#ff8c00", width=2, dash=(6, 3))
        canvas.create_text(
            right,
            max(top + 4, threshold_y - 6),
            text=f"threshold {threshold:.3f}",
            anchor="se",
            fill="#ff8c00",
        )

        count = len(self.latest_results)
        capture_seconds = [extract_capture_seconds(result.image_path) for result in self.latest_results]
        use_time_axis = all(value is not None for value in capture_seconds)
        if use_time_axis:
            time_values = [int(value) for value in capture_seconds if value is not None]
            min_time = min(time_values)
            max_time = max(time_values)
            if min_time == max_time:
                x_positions = [(left + right) / 2 for _ in time_values]
            else:
                x_positions = [
                    left + ((value - min_time) / (max_time - min_time)) * (right - left)
                    for value in time_values
                ]

            start_hour = min_time // 3600
            end_hour = math.ceil(max_time / 3600)
            label_step = calculate_hour_tick_step(right - left, max(1, end_hour - start_hour))
            for hour in range(start_hour, end_hour + 1):
                tick_seconds = hour * 3600
                if tick_seconds < min_time or tick_seconds > max_time:
                    continue
                x = left + ((tick_seconds - min_time) / (max_time - min_time)) * (right - left)
                canvas.create_line(x, top, x, bottom, fill="#e3e7ec", dash=(2, 4))
                if (hour - start_hour) % label_step == 0 or hour == end_hour:
                    canvas.create_text(x, height - 6, text=str(hour), anchor="s", fill="#55606f")
            canvas.create_text((left + right) / 2, height - 22, text="Time", anchor="s", fill="#55606f")
        elif count == 1:
            x_positions = [(left + right) / 2]
            canvas.create_text((left + right) / 2, height - 22, text="Image Order", anchor="s", fill="#55606f")
        else:
            x_positions = [
                left + (index / (count - 1)) * (right - left)
                for index in range(count)
            ]
            canvas.create_text((left + right) / 2, height - 22, text="Image Order", anchor="s", fill="#55606f")

        for index, result in enumerate(self.latest_results):
            y = y_for(scores[index])
            fill = "#d94141" if result.score >= threshold else "#2d7dd2"
            canvas.create_oval(
                x_positions[index] - 3,
                y - 3,
                x_positions[index] + 3,
                y + 3,
                fill=fill,
                outline="",
            )

        if not use_time_axis:
            canvas.create_text(left, height - 6, text="1", anchor="sw", fill="#55606f")
            canvas.create_text(right, height - 6, text=str(count), anchor="se", fill="#55606f")

    def _sync_blocked_scroll_region(self, _event: tk.Event) -> None:
        self.blocked_canvas.configure(scrollregion=self.blocked_canvas.bbox("all"))

    def _resize_blocked_window(self, event: tk.Event) -> None:
        self.blocked_canvas.configure(scrollregion=self.blocked_canvas.bbox("all"))

    def _close(self) -> None:
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    DetectionGui(root)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        try:
            root.destroy()
        except tk.TclError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
