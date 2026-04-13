from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    workspace_dir: Path
    data_dir: Path
    inbox_dir: Path
    working_dir: Path
    reports_dir: Path
    samples_dir: Path
    dropbox_root: str
    dropbox_app_key: str
    dropbox_app_secret: str
    dropbox_refresh_token: str
    dropbox_access_token: str
    dropbox_account_info_file: str


def load_settings(workspace_dir: Path | None = None) -> Settings:
    root = workspace_dir or Path(__file__).resolve().parents[2]
    data_dir = root / "data"
    return Settings(
        workspace_dir=root,
        data_dir=data_dir,
        inbox_dir=data_dir / "inbox",
        working_dir=data_dir / "working",
        reports_dir=root / "reports",
        samples_dir=data_dir / "samples",
        dropbox_root=os.getenv("BLOCK_DETECT_DROPBOX_ROOT", "/"),
        dropbox_app_key=os.getenv("DROPBOX_APP_KEY", "").strip(),
        dropbox_app_secret=os.getenv("DROPBOX_APP_SECRET", "").strip(),
        dropbox_refresh_token=os.getenv("DROPBOX_REFRESH_TOKEN", "").strip(),
        dropbox_access_token=os.getenv("DROPBOX_ACCESS_TOKEN", "").strip(),
        dropbox_account_info_file=os.getenv("DROPBOX_ACCOUNT_INFO_FILE", "").strip(),
    )


def ensure_workspace_dirs(settings: Settings) -> None:
    for path in (
        settings.data_dir,
        settings.inbox_dir,
        settings.working_dir,
        settings.reports_dir,
        settings.samples_dir,
        settings.workspace_dir / "secrets",
    ):
        path.mkdir(parents=True, exist_ok=True)

