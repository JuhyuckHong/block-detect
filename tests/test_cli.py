import os
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from block_detect import cli  # noqa: E402
from block_detect.classifier import PlaceholderClassifier  # noqa: E402
from block_detect.config import load_settings, ensure_workspace_dirs  # noqa: E402


class CliTest(unittest.TestCase):
    def test_prepare_creates_workspace_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace = Path(tmpdir)
            settings = load_settings(workspace)
            ensure_workspace_dirs(settings)

            self.assertTrue((workspace / "data").exists())
            self.assertTrue((workspace / "data" / "inbox").exists())
            self.assertTrue((workspace / "reports").exists())

    def test_parser_accepts_prepare_only(self):
        parser = cli.build_parser()
        args = parser.parse_args(["--prepare-only"])
        self.assertTrue(args.prepare_only)

    def test_classifier_marks_ab_samples_abnormal(self):
        classifier = PlaceholderClassifier()
        sample_paths = sorted(PROJECT_ROOT.glob("tests/ab-*.jpg"))

        self.assertTrue(sample_paths)

        for sample_path in sample_paths:
            result = classifier.classify(sample_path)
            self.assertEqual(result.label, "abnormal", sample_path.name)
            self.assertGreater(result.score, 0.0, sample_path.name)

    def test_classifier_marks_normal_sample_normal(self):
        classifier = PlaceholderClassifier()

        result = classifier.classify(PROJECT_ROOT / "tests/normal.jpg")

        self.assertEqual(result.label, "normal")
        self.assertGreaterEqual(result.score, 0.0)


if __name__ == "__main__":
    unittest.main()
