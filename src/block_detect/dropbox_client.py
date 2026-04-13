from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import Settings


@dataclass
class DropboxCredentials:
    app_key: str = ""
    app_secret: str = ""
    refresh_token: str = ""
    access_token: str = ""


def load_dropbox_credentials(settings: Settings) -> DropboxCredentials:
    creds = DropboxCredentials(
        app_key=settings.dropbox_app_key,
        app_secret=settings.dropbox_app_secret,
        refresh_token=settings.dropbox_refresh_token,
        access_token=settings.dropbox_access_token,
    )

    if not settings.dropbox_account_info_file:
        return creds

    info_file = Path(settings.dropbox_account_info_file)
    if not info_file.exists():
        return creds

    with info_file.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return DropboxCredentials(
        app_key=creds.app_key or str(data.get("app_key", "")).strip(),
        app_secret=creds.app_secret or str(data.get("app_secret", "")).strip(),
        refresh_token=creds.refresh_token or str(
            data.get("oauth2_refresh_token", data.get("refresh_token", ""))
        ).strip(),
        access_token=creds.access_token or str(
            data.get("oauth2_access_token", data.get("access_token", ""))
        ).strip(),
    )


class DropboxClient:
    """Thin wrapper reserved for Dropbox listing/download/upload work."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.credentials = load_dropbox_credentials(settings)

    def validate_credentials(self) -> bool:
        if self.credentials.app_key and self.credentials.app_secret and self.credentials.refresh_token:
            return True
        if self.credentials.access_token:
            return True
        return False

    def list_day_images(self, remote_day_path: str) -> list[str]:
        raise NotImplementedError("Implement Dropbox folder listing here.")

    def download_images(self, remote_paths: list[str], target_dir: Path) -> list[Path]:
        raise NotImplementedError("Implement Dropbox file download here.")

    def upload_report(self, local_report: Path, remote_path: str) -> None:
        raise NotImplementedError("Implement Dropbox report upload here.")

