from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import dropbox

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
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}

    def __init__(self, settings: Settings):
        self.settings = settings
        self.credentials = load_dropbox_credentials(settings)
        self._client: dropbox.Dropbox | None = None

    def validate_credentials(self) -> bool:
        if self.credentials.app_key and self.credentials.app_secret and self.credentials.refresh_token:
            return True
        if self.credentials.access_token:
            return True
        return False

    def get_client(self) -> dropbox.Dropbox:
        if self._client is not None:
            return self._client

        if self.credentials.app_key and self.credentials.app_secret and self.credentials.refresh_token:
            self._client = dropbox.Dropbox(
                app_key=self.credentials.app_key,
                app_secret=self.credentials.app_secret,
                oauth2_refresh_token=self.credentials.refresh_token,
            )
            return self._client

        if self.credentials.access_token:
            self._client = dropbox.Dropbox(oauth2_access_token=self.credentials.access_token)
            return self._client

        raise RuntimeError("Dropbox credentials are not configured.")

    def list_day_images(self, remote_day_path: str) -> list[str]:
        entries = self.get_client().files_list_folder(remote_day_path).entries
        image_paths: list[str] = []
        for entry in entries:
            path_display = getattr(entry, "path_display", None)
            if not path_display:
                continue
            if Path(path_display).suffix.lower() in self.IMAGE_EXTENSIONS:
                image_paths.append(path_display)
        return sorted(image_paths)

    def download_images(self, remote_paths: list[str], target_dir: Path) -> list[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        client = self.get_client()
        downloaded_paths: list[Path] = []
        for remote_path in remote_paths:
            local_path = target_dir / Path(remote_path).name
            client.files_download_to_file(str(local_path), remote_path)
            downloaded_paths.append(local_path)
        return downloaded_paths

    def upload_report(self, local_report: Path, remote_path: str) -> None:
        raise NotImplementedError("Implement Dropbox report upload here.")
