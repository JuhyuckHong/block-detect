from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


def load_dotenv_file(env_file: Path) -> None:
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def _load_positive_int_env(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    return max(1, value)


@dataclass(frozen=True)
class Settings:
    workspace_dir: Path
    data_dir: Path
    inbox_dir: Path
    working_dir: Path
    thumbnail_cache_dir: Path
    reports_dir: Path
    samples_dir: Path
    dropbox_root: str
    dropbox_app_key: str
    dropbox_app_secret: str
    dropbox_refresh_token: str
    dropbox_access_token: str
    dropbox_account_info_file: str
    dropbox_day_template: str
    dark_threshold: int
    score_threshold: float
    roi_left_y_ratio: float
    roi_bottom_x_ratio: float
    dark_ratio_threshold: float
    mean_brightness_threshold: float
    border_dark_region_threshold: float
    border_dark_ratio_min: float
    download_workers: int
    classify_workers: int


def load_settings(workspace_dir: Path | None = None) -> Settings:
    root = workspace_dir or Path(__file__).resolve().parents[2]
    load_dotenv_file(root / ".env")
    data_dir = root / "data"
    cpu_count = os.cpu_count() or 4
    return Settings(
        workspace_dir=root,
        data_dir=data_dir,
        inbox_dir=data_dir / "inbox",
        working_dir=data_dir / "working",
        thumbnail_cache_dir=data_dir / "working" / "thumbnail_cache",
        reports_dir=root / "reports",
        samples_dir=data_dir / "samples",
        dropbox_root=os.getenv("BLOCK_DETECT_DROPBOX_ROOT", "/"),
        dropbox_app_key=os.getenv("DROPBOX_APP_KEY", "").strip(),
        dropbox_app_secret=os.getenv("DROPBOX_APP_SECRET", "").strip(),
        dropbox_refresh_token=os.getenv("DROPBOX_REFRESH_TOKEN", "").strip(),
        dropbox_access_token=os.getenv("DROPBOX_ACCESS_TOKEN", "").strip(),
        dropbox_account_info_file=os.getenv("DROPBOX_ACCOUNT_INFO_FILE", "").strip(),
        dropbox_day_template=os.getenv("BLOCK_DETECT_DROPBOX_DAY_TEMPLATE", "{date}").strip(),
        dark_threshold=int(os.getenv("BLOCK_DETECT_DARK_THRESHOLD", "32")),
        score_threshold=float(
            os.getenv(
                "BLOCK_DETECT_SCORE_THRESHOLD",
                "0.80",
            )
        ),
        roi_left_y_ratio=float(os.getenv("BLOCK_DETECT_ROI_LEFT_Y_RATIO", "0.30")),
        roi_bottom_x_ratio=float(os.getenv("BLOCK_DETECT_ROI_BOTTOM_X_RATIO", "0.40")),
        dark_ratio_threshold=float(os.getenv("BLOCK_DETECT_DARK_RATIO_THRESHOLD", "0.58")),
        mean_brightness_threshold=float(
            os.getenv("BLOCK_DETECT_MEAN_BRIGHTNESS_THRESHOLD", "50.0")
        ),
        border_dark_region_threshold=float(
            os.getenv("BLOCK_DETECT_BORDER_DARK_REGION_THRESHOLD", "0.48")
        ),
        border_dark_ratio_min=float(os.getenv("BLOCK_DETECT_BORDER_DARK_RATIO_MIN", "0.54")),
        download_workers=_load_positive_int_env("BLOCK_DETECT_DOWNLOAD_WORKERS", 8),
        classify_workers=_load_positive_int_env("BLOCK_DETECT_CLASSIFY_WORKERS", cpu_count),
    )


def ensure_workspace_dirs(settings: Settings) -> None:
    for path in (
        settings.data_dir,
        settings.inbox_dir,
        settings.working_dir,
        settings.thumbnail_cache_dir,
        settings.reports_dir,
        settings.samples_dir,
        settings.workspace_dir / "secrets",
    ):
        path.mkdir(parents=True, exist_ok=True)
