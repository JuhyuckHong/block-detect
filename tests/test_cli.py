import io
import os
import shutil
import sys
import tempfile
import threading
import time
import types
import unittest
import uuid
from contextlib import contextmanager
from contextlib import redirect_stdout
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
TEST_TMP_ROOT = PROJECT_ROOT / ".tmp-tests"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

if "dropbox" not in sys.modules:
    sys.modules["dropbox"] = types.SimpleNamespace(Dropbox=mock.Mock(name="Dropbox"))

from block_detect import cli  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from block_detect.classifier import BlackPixelClassifier, ClassificationResult  # noqa: E402
from block_detect.config import ensure_workspace_dirs, load_dotenv_file, load_settings  # noqa: E402
from block_detect.dropbox_client import (  # noqa: E402
    DropboxClient,
    RemoteImageMetadata,
    load_dropbox_credentials,
)
from block_detect.gui import (  # noqa: E402
    apply_runtime_overrides,
    blocked_results,
    format_ratio,
    render_preview_image,
    thumbnail_cache_key,
)
from block_detect.pipeline import (  # noqa: E402
    DetectionPipeline,
    PipelineRunResult,
    PipelineSummary,
    load_saved_run,
)


@contextmanager
def workspace_tempdir():
    TEST_TMP_ROOT.mkdir(parents=True, exist_ok=True)
    tmpdir = TEST_TMP_ROOT / f"case-{uuid.uuid4().hex}"
    tmpdir.mkdir(parents=True, exist_ok=False)
    try:
        yield str(tmpdir)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


class FakeDropboxClient:
    def __init__(self, file_map: dict[str, Path]):
        self.file_map = file_map
        self.listed_paths: list[str] = []
        self.downloaded_paths: list[str] = []

    def list_day_images(self, remote_day_path: str) -> list[RemoteImageMetadata]:
        self.listed_paths.append(remote_day_path)
        return [
            RemoteImageMetadata(path_display=remote_path, name=Path(remote_path).name)
            for remote_path in sorted(self.file_map)
        ]

    def download_images(self, remote_paths: list[str], target_dir: Path, progress_callback=None) -> list[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        total = len(remote_paths)
        if progress_callback is not None:
            progress_callback(0, total)
        for index, remote_path in enumerate(remote_paths, start=1):
            self.downloaded_paths.append(remote_path.path_display)
            source = self.file_map[remote_path.path_display]
            target = target_dir / source.name
            shutil.copy2(source, target)
            downloaded.append(target)
            if progress_callback is not None:
                progress_callback(index, total)
        return downloaded


class FakePipeline:
    def __init__(self):
        self.prepare_called = False
        self.run_calls: list[tuple[str, str | None]] = []

    def prepare(self) -> None:
        self.prepare_called = True

    def run_day(self, day: str, remote_day_path: str | None = None) -> PipelineSummary:
        self.run_calls.append((day, remote_day_path))
        return PipelineSummary(
            processed_count=6,
            abnormal_count=5,
            normal_count=1,
            unknown_count=0,
        )


class RecordingClassifier:
    def __init__(self):
        self.thread_ids: list[int] = []
        self._lock = threading.Lock()

    def classify(self, image_path: Path) -> ClassificationResult:
        time.sleep(0.05)
        with self._lock:
            self.thread_ids.append(threading.get_ident())
        return ClassificationResult(
            image_path=image_path,
            label="normal",
            score=0.0,
            reason="recorded",
        )


class RecordingDownloader:
    def __init__(self, file_map: dict[str, Path]):
        self.file_map = file_map
        self.thread_ids: list[int] = []
        self._lock = threading.Lock()

    def files_download_to_file(self, local_path: str, remote_path: str) -> None:
        time.sleep(0.05)
        shutil.copy2(self.file_map[remote_path], local_path)
        with self._lock:
            self.thread_ids.append(threading.get_ident())


class SequenceClassifier:
    def classify(self, image_path: Path) -> ClassificationResult:
        return ClassificationResult(
            image_path=image_path,
            label="normal",
            score=0.0,
            reason="sequence",
        )


def create_roi_normal_image(image_path: Path) -> None:
    image = Image.new("L", (200, 120), color=90)
    draw = ImageDraw.Draw(image)
    draw.rectangle((110, 0, 199, 69), fill=0)
    image.save(image_path)


def create_border_occlusion_image(image_path: Path) -> None:
    image = Image.new("L", (200, 120), color=80)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, 109, 119), fill=0)
    image.save(image_path)


class CliTest(unittest.TestCase):
    def test_prepare_creates_workspace_dirs(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            settings = load_settings(workspace)
            ensure_workspace_dirs(settings)

            self.assertTrue((workspace / "data").exists())
            self.assertTrue((workspace / "data" / "inbox").exists())
            self.assertTrue((workspace / "data" / "working" / "thumbnail_cache").exists())
            self.assertTrue((workspace / "reports").exists())

    def test_parser_accepts_prepare_only(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--prepare-only"])
        self.assertTrue(args.prepare_only)

    def test_load_settings_exposes_detection_thresholds(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)

            original_dropbox_root = os.environ.get("BLOCK_DETECT_DROPBOX_ROOT")
            original_day_template = os.environ.get("BLOCK_DETECT_DROPBOX_DAY_TEMPLATE")
            original_download_workers = os.environ.get("BLOCK_DETECT_DOWNLOAD_WORKERS")
            original_classify_workers = os.environ.get("BLOCK_DETECT_CLASSIFY_WORKERS")
            os.environ["BLOCK_DETECT_DROPBOX_ROOT"] = "/captures"
            os.environ["BLOCK_DETECT_DROPBOX_DAY_TEMPLATE"] = "{date}/camera-a"
            os.environ["BLOCK_DETECT_DOWNLOAD_WORKERS"] = "6"
            os.environ["BLOCK_DETECT_CLASSIFY_WORKERS"] = "3"
            try:
                settings = load_settings(workspace)
            finally:
                if original_dropbox_root is None:
                    os.environ.pop("BLOCK_DETECT_DROPBOX_ROOT", None)
                else:
                    os.environ["BLOCK_DETECT_DROPBOX_ROOT"] = original_dropbox_root

                if original_day_template is None:
                    os.environ.pop("BLOCK_DETECT_DROPBOX_DAY_TEMPLATE", None)
                else:
                    os.environ["BLOCK_DETECT_DROPBOX_DAY_TEMPLATE"] = original_day_template

                if original_download_workers is None:
                    os.environ.pop("BLOCK_DETECT_DOWNLOAD_WORKERS", None)
                else:
                    os.environ["BLOCK_DETECT_DOWNLOAD_WORKERS"] = original_download_workers

                if original_classify_workers is None:
                    os.environ.pop("BLOCK_DETECT_CLASSIFY_WORKERS", None)
                else:
                    os.environ["BLOCK_DETECT_CLASSIFY_WORKERS"] = original_classify_workers

        self.assertEqual(settings.dropbox_root, "/captures")
        self.assertEqual(settings.dropbox_day_template, "{date}/camera-a")
        self.assertEqual(settings.dark_threshold, 32)
        self.assertEqual(settings.score_threshold, 0.80)
        self.assertEqual(settings.roi_line_offset_ratio, 0.12)
        self.assertEqual(settings.dark_ratio_threshold, 0.58)
        self.assertEqual(settings.mean_brightness_threshold, 50.0)
        self.assertEqual(settings.border_dark_region_threshold, 0.48)
        self.assertEqual(settings.border_dark_ratio_min, 0.54)
        self.assertEqual(settings.download_workers, 6)
        self.assertEqual(settings.classify_workers, 3)

    def test_load_settings_reads_dropbox_credentials_from_dotenv(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            env_path = workspace / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "DROPBOX_APP_KEY=test-app-key",
                        "DROPBOX_APP_SECRET=test-app-secret",
                        "DROPBOX_REFRESH_TOKEN=test-refresh-token",
                        "BLOCK_DETECT_DROPBOX_ROOT=/dotenv-root",
                    ]
                ),
                encoding="utf-8",
            )
            original_env = {
                "DROPBOX_APP_KEY": os.environ.pop("DROPBOX_APP_KEY", None),
                "DROPBOX_APP_SECRET": os.environ.pop("DROPBOX_APP_SECRET", None),
                "DROPBOX_REFRESH_TOKEN": os.environ.pop("DROPBOX_REFRESH_TOKEN", None),
                "BLOCK_DETECT_DROPBOX_ROOT": os.environ.pop("BLOCK_DETECT_DROPBOX_ROOT", None),
            }
            try:
                settings = load_settings(workspace)
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        self.assertEqual(settings.dropbox_app_key, "test-app-key")
        self.assertEqual(settings.dropbox_app_secret, "test-app-secret")
        self.assertEqual(settings.dropbox_refresh_token, "test-refresh-token")
        self.assertEqual(settings.dropbox_root, "/dotenv-root")

    def test_load_dotenv_file_does_not_override_existing_environment(self):
        with workspace_tempdir() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text("DROPBOX_APP_KEY=from-dotenv", encoding="utf-8")
            original_value = os.environ.get("DROPBOX_APP_KEY")
            os.environ["DROPBOX_APP_KEY"] = "from-env"
            try:
                load_dotenv_file(env_path)
                loaded_value = os.environ["DROPBOX_APP_KEY"]
            finally:
                if original_value is None:
                    os.environ.pop("DROPBOX_APP_KEY", None)
                else:
                    os.environ["DROPBOX_APP_KEY"] = original_value

        self.assertEqual(loaded_value, "from-env")

    def test_classifier_marks_ab_samples_abnormal(self):
        classifier = BlackPixelClassifier()
        sample_paths = sorted(PROJECT_ROOT.glob("tests/ab-*.jpg"))

        self.assertTrue(sample_paths)

        for sample_path in sample_paths:
            result = classifier.classify(sample_path)
            self.assertEqual(result.label, "abnormal", sample_path.name)
            self.assertGreater(result.score, 0.0, sample_path.name)

    def test_classifier_marks_normal_sample_normal(self):
        classifier = BlackPixelClassifier()

        with workspace_tempdir() as tmpdir:
            image_path = Path(tmpdir) / "roi-normal.png"
            create_roi_normal_image(image_path)
            result = classifier.classify(image_path)

        self.assertEqual(result.label, "normal")
        self.assertGreaterEqual(result.score, 0.0)

    def test_classifier_marks_border_occlusion_as_abnormal(self):
        classifier = BlackPixelClassifier()

        with workspace_tempdir() as tmpdir:
            image_path = Path(tmpdir) / "border-occlusion.png"
            image = Image.new("L", (200, 120), color=80)
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 109, 119), fill=0)
            image.save(image_path)

            result = classifier.classify(image_path)

        self.assertEqual(result.label, "abnormal")
        self.assertIn("mean_brightness=", result.reason)

    def test_classifier_keeps_center_dark_region_normal(self):
        classifier = BlackPixelClassifier()

        with workspace_tempdir() as tmpdir:
            image_path = Path(tmpdir) / "center-dark.png"
            image = Image.new("L", (200, 120), color=80)
            draw = ImageDraw.Draw(image)
            draw.rectangle((70, 30, 129, 89), fill=0)
            image.save(image_path)

            result = classifier.classify(image_path)

        self.assertEqual(result.label, "normal")

    def test_classifier_ignores_upper_right_dark_region(self):
        classifier = BlackPixelClassifier()

        with workspace_tempdir() as tmpdir:
            image_path = Path(tmpdir) / "upper-right-dark.png"
            image = Image.new("L", (200, 120), color=80)
            draw = ImageDraw.Draw(image)
            draw.rectangle((110, 0, 199, 69), fill=0)
            image.save(image_path)

            result = classifier.classify(image_path)

        self.assertEqual(result.label, "normal")
        self.assertIn("roi=shifted_lower_left_triangle", result.reason)

    def test_pipeline_runs_day_batch(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            settings = load_settings(workspace)
            ensure_workspace_dirs(settings)
            fixtures_dir = workspace / "fixtures"
            fixtures_dir.mkdir(parents=True, exist_ok=True)
            normal_path = fixtures_dir / "normal.png"
            abnormal_1 = fixtures_dir / "ab-1.png"
            abnormal_2 = fixtures_dir / "ab-2.png"
            create_roi_normal_image(normal_path)
            create_border_occlusion_image(abnormal_1)
            create_border_occlusion_image(abnormal_2)
            file_map = {
                f"/captures/2026-04-13/{path.name}": path
                for path in [normal_path, abnormal_1, abnormal_2]
            }
            pipeline = DetectionPipeline(
                settings=settings,
                classifier=BlackPixelClassifier(),
                dropbox_client=FakeDropboxClient(file_map),
            )

            summary = pipeline.run_day("2026-04-13", remote_day_path="/captures/2026-04-13")

        self.assertEqual(summary.processed_count, 3)
        self.assertEqual(summary.abnormal_count, 2)
        self.assertEqual(summary.normal_count, 1)
        self.assertEqual(summary.unknown_count, 0)

    def test_pipeline_returns_run_details(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            settings = load_settings(workspace)
            ensure_workspace_dirs(settings)
            file_map = {
                f"/captures/2026-04-13/{path.name}": path
                for path in sorted(PROJECT_ROOT.glob("tests/*.jpg"))
            }
            pipeline = DetectionPipeline(
                settings=settings,
                classifier=BlackPixelClassifier(),
                dropbox_client=FakeDropboxClient(file_map),
            )

            run_result = pipeline.run_day_with_details(
                "2026-04-13",
                remote_day_path="/captures/2026-04-13",
            )

            self.assertIsInstance(run_result, PipelineRunResult)
            self.assertEqual(run_result.summary.processed_count, 6)
            self.assertEqual(run_result.remote_day_path, "/captures/2026-04-13")
            self.assertTrue(run_result.report_path.exists())
            self.assertEqual(len(run_result.results), 6)

    def test_pipeline_can_reclassify_existing_local_day_without_download(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            settings = replace(load_settings(workspace), classify_workers=1)
            ensure_workspace_dirs(settings)
            day = "2026-04-13"
            local_day_dir = settings.inbox_dir / day
            local_day_dir.mkdir(parents=True, exist_ok=True)

            normal_path = local_day_dir / "normal.png"
            blocked_path = local_day_dir / "blocked.png"
            create_roi_normal_image(normal_path)
            create_border_occlusion_image(blocked_path)

            pipeline = DetectionPipeline(
                settings=settings,
                classifier=BlackPixelClassifier(),
                dropbox_client=FakeDropboxClient({}),
            )

            run_result = pipeline.run_local_day_with_details(
                day,
                remote_day_path="/captures/2026-04-13",
            )

            self.assertEqual(run_result.summary.processed_count, 2)
            self.assertEqual(run_result.summary.abnormal_count, 1)
            self.assertEqual(run_result.summary.normal_count, 1)
            self.assertTrue(run_result.report_path.exists())

    def test_load_saved_run_reconstructs_results_from_report(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            settings = load_settings(workspace)
            ensure_workspace_dirs(settings)
            file_map = {
                f"/captures/2026-04-13/{path.name}": path
                for path in sorted(PROJECT_ROOT.glob("tests/*.jpg"))
            }
            pipeline = DetectionPipeline(
                settings=settings,
                classifier=BlackPixelClassifier(),
                dropbox_client=FakeDropboxClient(file_map),
            )
            saved = pipeline.run_day_with_details(
                "2026-04-13",
                remote_day_path="/captures/2026-04-13",
            )

            loaded = load_saved_run("2026-04-13", settings=settings)

        self.assertEqual(loaded.day, saved.day)
        self.assertEqual(loaded.remote_day_path, saved.remote_day_path)
        self.assertEqual(loaded.summary.processed_count, saved.summary.processed_count)
        self.assertEqual(
            [item.image_path for item in loaded.results],
            [item.image_path for item in saved.results],
        )

    def test_pipeline_classifies_with_multiple_workers(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            settings = replace(load_settings(workspace), classify_workers=4)
            classifier = RecordingClassifier()
            pipeline = DetectionPipeline(settings=settings, classifier=classifier, dropbox_client=FakeDropboxClient({}))
            image_paths = sorted(PROJECT_ROOT.glob("tests/*.jpg"))

            results = pipeline.classify_local_images(image_paths)

        self.assertEqual([result.image_path for result in results], image_paths)
        self.assertGreater(len(set(classifier.thread_ids)), 1)

    def test_pipeline_reports_progress_for_download_and_classify(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            settings = replace(load_settings(workspace), classify_workers=1, download_workers=1)
            file_map = {
                f"/captures/2026-04-13/{path.name}": path
                for path in sorted(PROJECT_ROOT.glob("tests/*.jpg"))[:3]
            }
            pipeline = DetectionPipeline(
                settings=settings,
                classifier=SequenceClassifier(),
                dropbox_client=FakeDropboxClient(file_map),
            )
            progress_events: list[tuple[str, int, int]] = []

            run_result = pipeline.run_day_with_details(
                "2026-04-13",
                remote_day_path="/captures/2026-04-13",
                progress_callback=lambda stage, current, total: progress_events.append(
                    (stage, current, total)
                ),
            )

        self.assertEqual(run_result.summary.processed_count, 3)
        self.assertIn(("download", 0, 3), progress_events)
        self.assertIn(("download", 3, 3), progress_events)
        self.assertIn(("classify", 0, 3), progress_events)
        self.assertIn(("classify", 3, 3), progress_events)

    def test_dropbox_client_downloads_with_multiple_workers(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            settings = replace(load_settings(workspace), download_workers=4)
            client = DropboxClient(settings)
            remote_paths = [
                RemoteImageMetadata(
                    path_display=f"/captures/2026-04-13/{path.name}",
                    name=path.name,
                )
                for path in sorted(PROJECT_ROOT.glob("tests/*.jpg"))
            ]
            file_map = {remote_path.path_display: PROJECT_ROOT / "tests" / remote_path.name for remote_path in remote_paths}
            downloader = RecordingDownloader(file_map)
            target_dir = workspace / "downloads"

            with mock.patch.object(client, "get_client", return_value=downloader):
                downloaded = client.download_images(remote_paths, target_dir)

        self.assertEqual([path.name for path in downloaded], [remote_path.name for remote_path in remote_paths])
        self.assertGreater(len(set(downloader.thread_ids)), 1)

    def test_dropbox_client_skips_download_when_local_file_matches_remote_metadata(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            settings = replace(load_settings(workspace), download_workers=1)
            client = DropboxClient(settings)
            target_dir = workspace / "downloads"
            target_dir.mkdir(parents=True, exist_ok=True)
            local_path = target_dir / "normal.jpg"
            shutil.copy2(PROJECT_ROOT / "tests" / "normal.jpg", local_path)

            content_hash = client._compute_dropbox_content_hash(local_path)
            modified_time = datetime.fromtimestamp(local_path.stat().st_mtime, tz=timezone.utc)
            remote_paths = [
                RemoteImageMetadata(
                    path_display="/captures/2026-04-13/normal.jpg",
                    name="normal.jpg",
                    size=local_path.stat().st_size,
                    content_hash=content_hash,
                    server_modified=modified_time,
                )
            ]
            downloader = RecordingDownloader({})

            with mock.patch.object(client, "get_client", return_value=downloader):
                downloaded = client.download_images(remote_paths, target_dir)

        self.assertEqual(downloaded, [local_path])
        self.assertEqual(downloader.thread_ids, [])

    def test_gui_helpers_format_blocked_ratio(self):
        self.assertEqual(format_ratio(5, 6), "83.3% (5/6)")
        self.assertEqual(format_ratio(0, 0), "0.0% (0/0)")

    def test_gui_helpers_filter_blocked_results(self):
        results = [
            ClassificationResult(PROJECT_ROOT / "tests/ab-1.jpg", "abnormal", 0.9, "blocked"),
            ClassificationResult(PROJECT_ROOT / "tests/normal.jpg", "normal", 0.1, "normal"),
            ClassificationResult(PROJECT_ROOT / "tests/ab-2.jpg", "blocked", 0.95, "blocked alias"),
        ]

        filtered = blocked_results(results)

        self.assertEqual([item.image_path.name for item in filtered], ["ab-1.jpg", "ab-2.jpg"])

    def test_render_preview_image_draws_roi_overlay(self):
        image = Image.new("RGB", (200, 120), color=(90, 90, 90))

        plain = render_preview_image(image, (100, 60), show_roi_overlay=False)
        overlay = render_preview_image(image, (100, 60), show_roi_overlay=True)

        self.assertEqual(plain.size, (100, 60))
        self.assertEqual(overlay.size, (100, 60))
        self.assertNotEqual(plain.getpixel((80, 10)), overlay.getpixel((80, 10)))
        self.assertNotEqual(plain.getpixel((50, 30)), overlay.getpixel((50, 30)))

    def test_apply_runtime_overrides_updates_thresholds(self):
        settings = load_settings(PROJECT_ROOT)

        updated = apply_runtime_overrides(
            settings,
            download_workers=3,
            classify_workers=5,
            score_threshold=0.62,
            roi_line_offset_ratio=0.18,
        )

        self.assertEqual(updated.download_workers, 3)
        self.assertEqual(updated.classify_workers, 5)
        self.assertEqual(updated.score_threshold, 0.62)
        self.assertEqual(updated.roi_line_offset_ratio, 0.18)

    def test_thumbnail_cache_key_changes_when_file_changes(self):
        with workspace_tempdir() as tmpdir:
            image_path = Path(tmpdir) / "thumb.png"
            create_roi_normal_image(image_path)
            first = thumbnail_cache_key(image_path, (240, 180))
            time.sleep(0.01)
            create_border_occlusion_image(image_path)
            second = thumbnail_cache_key(image_path, (240, 180))

        self.assertNotEqual(first, second)

    def test_pipeline_summary_counts_blocked_alias_as_abnormal(self):
        settings = load_settings(PROJECT_ROOT)
        pipeline = DetectionPipeline(settings=settings, dropbox_client=FakeDropboxClient({}))
        results = [
            ClassificationResult(PROJECT_ROOT / "tests/ab-1.jpg", "abnormal", 0.9, "abnormal"),
            ClassificationResult(PROJECT_ROOT / "tests/ab-2.jpg", "blocked", 0.95, "blocked alias"),
            ClassificationResult(PROJECT_ROOT / "tests/normal.jpg", "normal", 0.1, "normal"),
        ]

        summary = pipeline.summarize(results)

        self.assertEqual(summary.processed_count, 3)
        self.assertEqual(summary.abnormal_count, 2)
        self.assertEqual(summary.normal_count, 1)
        self.assertEqual(summary.unknown_count, 0)

    def test_main_runs_batch_for_date(self):
        fake_pipeline = FakePipeline()
        stdout = io.StringIO()

        with mock.patch("block_detect.cli.build_pipeline", return_value=fake_pipeline):
            with redirect_stdout(stdout):
                exit_code = cli.main(["--date", "2026-04-13"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(fake_pipeline.run_calls, [("2026-04-13", None)])
        self.assertIn("processed=6", stdout.getvalue())

    def test_main_applies_worker_overrides(self):
        captured_settings = {}
        stdout = io.StringIO()

        def fake_build_pipeline(settings=None):
            captured_settings["settings"] = settings
            return FakePipeline()

        with mock.patch("block_detect.cli.build_pipeline", side_effect=fake_build_pipeline):
            with redirect_stdout(stdout):
                exit_code = cli.main(
                    [
                        "--date",
                        "2026-04-13",
                        "--download-workers",
                        "7",
                        "--classify-workers",
                        "5",
                    ]
                )

        self.assertEqual(exit_code, 0)
        self.assertEqual(captured_settings["settings"].download_workers, 7)
        self.assertEqual(captured_settings["settings"].classify_workers, 5)
        self.assertIn("processed=6", stdout.getvalue())

    def test_dropbox_client_prefers_refresh_token_flow_when_both_exist(self):
        with workspace_tempdir() as tmpdir:
            workspace = Path(tmpdir)
            original_env = {
                "DROPBOX_APP_KEY": os.environ.get("DROPBOX_APP_KEY"),
                "DROPBOX_APP_SECRET": os.environ.get("DROPBOX_APP_SECRET"),
                "DROPBOX_REFRESH_TOKEN": os.environ.get("DROPBOX_REFRESH_TOKEN"),
                "DROPBOX_ACCESS_TOKEN": os.environ.get("DROPBOX_ACCESS_TOKEN"),
            }
            os.environ["DROPBOX_APP_KEY"] = "app-key"
            os.environ["DROPBOX_APP_SECRET"] = "app-secret"
            os.environ["DROPBOX_REFRESH_TOKEN"] = "refresh-token"
            os.environ["DROPBOX_ACCESS_TOKEN"] = "expired-access-token"
            try:
                settings = load_settings(workspace)
            finally:
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

        client = DropboxClient(settings)

        with mock.patch("block_detect.dropbox_client.dropbox.Dropbox") as patched_dropbox:
            client.get_client()

        _, kwargs = patched_dropbox.call_args
        self.assertEqual(kwargs["app_key"], "app-key")
        self.assertEqual(kwargs["app_secret"], "app-secret")
        self.assertEqual(kwargs["oauth2_refresh_token"], "refresh-token")
        self.assertNotIn("oauth2_access_token", kwargs)


if __name__ == "__main__":
    unittest.main()
