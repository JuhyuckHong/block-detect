from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from dataclasses import dataclass
import os
from pathlib import Path
import threading
from typing import Callable

import dropbox

from .config import Settings


@dataclass
class DropboxCredentials:
    app_key: str = ""
    app_secret: str = ""
    refresh_token: str = ""
    access_token: str = ""


@dataclass(frozen=True)
class RemoteImageMetadata:
    path_display: str
    name: str
    size: int | None = None
    content_hash: str = ""
    server_modified: datetime | None = None


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
    HASH_BLOCK_SIZE = 4 * 1024 * 1024

    def __init__(self, settings: Settings):
        self.settings = settings
        self.credentials = load_dropbox_credentials(settings)
        self._thread_local = threading.local()

    def validate_credentials(self) -> bool:
        if self.credentials.app_key and self.credentials.app_secret and self.credentials.refresh_token:
            return True
        if self.credentials.access_token:
            return True
        return False

    def get_client(self) -> dropbox.Dropbox:
        client = getattr(self._thread_local, "client", None)
        if client is not None:
            return client

        if self.credentials.app_key and self.credentials.app_secret and self.credentials.refresh_token:
            client = dropbox.Dropbox(
                app_key=self.credentials.app_key,
                app_secret=self.credentials.app_secret,
                oauth2_refresh_token=self.credentials.refresh_token,
            )
            self._thread_local.client = client
            return client

        if self.credentials.access_token:
            client = dropbox.Dropbox(oauth2_access_token=self.credentials.access_token)
            self._thread_local.client = client
            return client

        raise RuntimeError("Dropbox credentials are not configured.")

    def list_day_images(self, remote_day_path: str) -> list[RemoteImageMetadata]:
        entries = self.get_client().files_list_folder(remote_day_path).entries
        image_paths: list[RemoteImageMetadata] = []
        for entry in entries:
            path_display = getattr(entry, "path_display", None)
            if not path_display:
                continue
            if Path(path_display).suffix.lower() in self.IMAGE_EXTENSIONS:
                image_paths.append(
                    RemoteImageMetadata(
                        path_display=path_display,
                        name=getattr(entry, "name", Path(path_display).name),
                        size=getattr(entry, "size", None),
                        content_hash=str(getattr(entry, "content_hash", "") or ""),
                        server_modified=getattr(entry, "server_modified", None),
                    )
                )
        return sorted(image_paths, key=lambda item: item.path_display)

    def _compute_dropbox_content_hash(self, file_path: Path) -> str:
        overall_hasher = hashlib.sha256()
        with file_path.open("rb") as handle:
            while True:
                block = handle.read(self.HASH_BLOCK_SIZE)
                if not block:
                    break
                overall_hasher.update(hashlib.sha256(block).digest())
        return overall_hasher.hexdigest()

    def _remote_timestamp(self, remote_image: RemoteImageMetadata) -> float | None:
        if remote_image.server_modified is None:
            return None
        if remote_image.server_modified.tzinfo is None:
            return remote_image.server_modified.replace(tzinfo=timezone.utc).timestamp()
        return remote_image.server_modified.astimezone(timezone.utc).timestamp()

    def _timestamps_match(self, local_path: Path, remote_image: RemoteImageMetadata) -> bool:
        remote_timestamp = self._remote_timestamp(remote_image)
        if remote_timestamp is None:
            return False
        return abs(local_path.stat().st_mtime - remote_timestamp) < 1.0

    def _preserve_remote_timestamp(self, local_path: Path, remote_image: RemoteImageMetadata) -> None:
        remote_timestamp = self._remote_timestamp(remote_image)
        if remote_timestamp is None:
            return
        os.utime(local_path, (remote_timestamp, remote_timestamp))

    def _is_local_copy_current(self, local_path: Path, remote_image: RemoteImageMetadata) -> bool:
        if not local_path.exists() or not local_path.is_file():
            return False

        if remote_image.size is not None and local_path.stat().st_size != remote_image.size:
            return False

        if remote_image.content_hash:
            if self._compute_dropbox_content_hash(local_path) != remote_image.content_hash:
                return False

        if remote_image.server_modified is not None and not self._timestamps_match(local_path, remote_image):
            return False

        return True

    def _download_image(self, remote_image: RemoteImageMetadata, target_dir: Path) -> Path:
        local_path = target_dir / remote_image.name
        if self._is_local_copy_current(local_path, remote_image):
            return local_path
        self.get_client().files_download_to_file(str(local_path), remote_image.path_display)
        self._preserve_remote_timestamp(local_path, remote_image)
        return local_path

    def download_images(
        self,
        remote_paths: list[RemoteImageMetadata],
        target_dir: Path,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        if not remote_paths:
            if progress_callback is not None:
                progress_callback(0, 0)
            return []

        max_workers = min(self.settings.download_workers, len(remote_paths))
        if progress_callback is not None:
            progress_callback(0, len(remote_paths))

        if max_workers == 1:
            downloaded_paths: list[Path] = []
            total = len(remote_paths)
            for index, remote_path in enumerate(remote_paths, start=1):
                downloaded_paths.append(self._download_image(remote_path, target_dir))
                if progress_callback is not None:
                    progress_callback(index, total)
            return downloaded_paths

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_index = {
                executor.submit(self._download_image, remote_path, target_dir): index
                for index, remote_path in enumerate(remote_paths)
            }
            downloaded_paths: list[Path | None] = [None] * len(remote_paths)
            completed = 0
            total = len(remote_paths)
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                downloaded_paths[index] = future.result()
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)
            return [path for path in downloaded_paths if path is not None]

    def upload_report(self, local_report: Path, remote_path: str) -> None:
        raise NotImplementedError("Implement Dropbox report upload here.")
