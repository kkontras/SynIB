import hashlib
import os
import pickle
import re
import shutil
import tarfile
import zipfile
from html import unescape
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin
from urllib.request import urlretrieve

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def sha256sum(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _extract_confirm_token(response) -> Optional[str]:
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value
    return None


def _stream_to_file(response, output_path: Path, chunk_size: int = 1024 * 1024) -> None:
    ensure_parent_dir(output_path)
    with output_path.open("wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def _looks_like_html(output_path: Path) -> bool:
    with output_path.open("rb") as f:
        head = f.read(1024).lstrip().lower()
    return head.startswith(b"<!doctype html") or head.startswith(b"<html")


def _parse_confirm_url_from_html(html_text: str) -> Optional[str]:
    patterns = [
        r'href="([^"]*confirm=[^"]+)"',
        r'action="([^"]*\/uc\?[^"]+)"',
    ]
    for pattern in patterns:
        match = re.search(pattern, html_text)
        if match:
            return urljoin("https://drive.google.com", unescape(match.group(1)))
    return None


def _download_confirmed(
    session,
    response,
    output_path: Path,
    timeout_seconds: int,
    file_id: str,
) -> None:
    token = _extract_confirm_token(response)
    if token:
        retry = session.get(
            "https://drive.google.com/uc",
            params={"export": "download", "id": file_id, "confirm": token},
            stream=True,
            timeout=timeout_seconds,
        )
        retry.raise_for_status()
        _stream_to_file(retry, output_path)
        if not _looks_like_html(output_path):
            return

    confirm_url = _parse_confirm_url_from_html(response.text)
    if not confirm_url:
        raise RuntimeError("Google Drive returned HTML and no confirm URL was found.")

    retry = session.get(confirm_url, stream=True, timeout=timeout_seconds)
    retry.raise_for_status()
    _stream_to_file(retry, output_path)
    if _looks_like_html(output_path):
        raise RuntimeError("Google Drive download still produced an HTML page, not the dataset file.")


def download_google_drive_file(
    file_id: str,
    output_path: Path,
    overwrite: bool = False,
    expected_sha256: Optional[str] = None,
    timeout_seconds: int = 60,
) -> Path:
    import requests

    if output_path.exists() and not overwrite:
        if expected_sha256:
            actual = sha256sum(output_path)
            if actual.lower() != expected_sha256.lower():
                raise ValueError(
                    f"Existing file hash mismatch for {output_path}: expected {expected_sha256}, got {actual}"
                )
        return output_path

    base_url = "https://drive.google.com/uc"
    params = {"export": "download", "id": file_id}

    with requests.Session() as session:
        response = session.get(base_url, params=params, stream=True, timeout=timeout_seconds)
        response.raise_for_status()
        _stream_to_file(response, output_path)
        if _looks_like_html(output_path):
            _download_confirmed(
                session=session,
                response=response,
                output_path=output_path,
                timeout_seconds=timeout_seconds,
                file_id=file_id,
            )

    if expected_sha256:
        actual = sha256sum(output_path)
        if actual.lower() != expected_sha256.lower():
            raise ValueError(
                f"Downloaded file hash mismatch for {output_path}: expected {expected_sha256}, got {actual}"
            )

    return output_path


def default_output_root() -> Path:
    env_root = os.getenv("SYNIB_FACTORCL_DATA_ROOT", "").strip()
    if not env_root:
        env_root = os.getenv("SYNERGY_FACTORCL_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path("src/synib/mydatasets/Factor_CL_Datasets/prepared").resolve()


def default_raw_output_root() -> Path:
    env_root = os.getenv("SYNIB_FACTORCL_RAW_ROOT", "").strip()
    if not env_root:
        env_root = os.getenv("SYNIB_FACTORCL_DATA_ROOT", "").strip()
    if not env_root:
        env_root = os.getenv("SYNERGY_FACTORCL_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path("src/synib/mydatasets/Factor_CL_Datasets/raw").resolve()


def download_url_file(
    url: str,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    output_path = output_path.expanduser().resolve()
    ensure_parent_dir(output_path)
    if output_path.exists() and not overwrite:
        return output_path
    urlretrieve(url, str(output_path))
    return output_path


def extract_archive(
    archive_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> Path:
    archive_path = archive_path.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        return output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    suffixes = [s.lower() for s in archive_path.suffixes]
    if archive_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(output_dir)
        return output_dir

    if suffixes[-2:] == [".tar", ".gz"] or suffixes[-2:] == [".tar", ".bz2"] or archive_path.suffix.lower() in {".tar", ".tgz"}:
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(output_dir)
        return output_dir

    raise ValueError(f"Unsupported archive format: {archive_path}")


def materialize_local_file(
    source_path: Path,
    output_path: Path,
    overwrite: bool = False,
    symlink: bool = False,
) -> Path:
    source_path = source_path.expanduser().resolve()
    output_path = output_path.expanduser().resolve()

    if not source_path.exists():
        raise FileNotFoundError(f"Local source file not found: {source_path}")

    if source_path.suffix.lower() == ".pkl":
        try:
            with source_path.open("rb") as f:
                pickle.load(f)
        except Exception as exc:
            raise ValueError(
                f"Local source is not a valid pickle file: {source_path}. "
                f"Original error: {exc}"
            ) from exc

    ensure_parent_dir(output_path)

    if output_path.exists():
        if not overwrite:
            return output_path
        if output_path.is_symlink() or output_path.is_file():
            output_path.unlink()
        else:
            raise IsADirectoryError(f"Expected file path, got directory: {output_path}")

    if symlink:
        output_path.symlink_to(source_path)
    else:
        shutil.copy2(source_path, output_path)
    return output_path
