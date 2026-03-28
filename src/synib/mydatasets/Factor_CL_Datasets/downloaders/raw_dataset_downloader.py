#!/usr/bin/env python3
"""
Helpers for materializing canonical raw FactorCL dataset roots.

Canonical layout:
    <output_root>/<dataset>_raw/
        metadata.jsonl
        source.json
        media/              # optional
        folds.json          # optional

The metadata.jsonl schema is intentionally simple and shared by the cache builder:
    {
      "id": str,
      "text": str,
      "label": int,
      "video_path": "media/foo.mp4",   # optional, relative to dataset root
      "context": [str],                # optional
      "folds": {"0": "train", "1": "val", "2": "test"}   # optional
    }
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Iterable, Optional

try:
    from .common import (
        default_raw_output_root,
        download_url_file,
        ensure_parent_dir,
        extract_archive,
    )
except ImportError:  # direct script execution fallback
    from common import (  # type: ignore
        default_raw_output_root,
        download_url_file,
        ensure_parent_dir,
        extract_archive,
    )


def canonical_dataset_root(output_root: Path, dataset_name: str) -> Path:
    return output_root.expanduser().resolve() / f"{dataset_name}_raw"


def _copy_or_symlink_tree(source: Path, dest: Path, *, symlink: bool) -> None:
    source = source.expanduser().resolve()
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = dest / item.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if symlink:
            target.symlink_to(item.resolve(), target_is_directory=item.is_dir())
        elif item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _write_json(path: Path, payload: dict) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _normalize_media_root(raw_root: Path, media_root: Optional[Path], *, symlink: bool) -> None:
    if media_root is None:
        return
    media_root = media_root.expanduser().resolve()
    target_media = raw_root / "media"
    if target_media.exists() or target_media.is_symlink():
        if target_media.is_dir() and not target_media.is_symlink():
            shutil.rmtree(target_media)
        else:
            target_media.unlink()
    if symlink:
        target_media.symlink_to(media_root, target_is_directory=True)
    else:
        shutil.copytree(media_root, target_media)


def materialize_canonical_raw_root(
    *,
    dataset_name: str,
    output_root: Path,
    source_url: Optional[str] = None,
    local_archive: Optional[Path] = None,
    local_root: Optional[Path] = None,
    metadata_jsonl: Optional[Path] = None,
    folds_json: Optional[Path] = None,
    media_root: Optional[Path] = None,
    symlink: bool = False,
    force: bool = False,
) -> Path:
    raw_root = canonical_dataset_root(output_root, dataset_name)
    raw_root.mkdir(parents=True, exist_ok=True)

    existing_meta = raw_root / "metadata.jsonl"
    legacy_meta = raw_root / "metadata.json"
    if (existing_meta.exists() or legacy_meta.exists()) and not force and metadata_jsonl is None and local_root is None and local_archive is None and source_url is None:
        return raw_root

    if metadata_jsonl is None and local_root is None and local_archive is None and not source_url:
        raise ValueError(
            f"No source provided for {dataset_name}. Pass one of --local-root, --local-archive, "
            f"--metadata-jsonl, or --source-url."
        )

    if force and raw_root.exists():
        for child in list(raw_root.iterdir()):
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        raw_root.mkdir(parents=True, exist_ok=True)

    source_payload = {
        "dataset": dataset_name,
        "materialized_at": str(raw_root),
        "mode": None,
    }

    if local_root is not None:
        _copy_or_symlink_tree(local_root, raw_root, symlink=symlink)
        source_payload["mode"] = "local_root"
        source_payload["local_root"] = str(local_root.expanduser().resolve())
    elif local_archive is not None or source_url:
        archive_path = local_archive
        if archive_path is None:
            archive_name = Path(source_url).name or f"{dataset_name}_raw_archive"
            archive_path = raw_root / "_downloads" / archive_name
            download_url_file(source_url, archive_path, overwrite=force)
            source_payload["source_url"] = source_url
        extract_archive(archive_path, raw_root, overwrite=force)
        source_payload["mode"] = "archive"
        source_payload["archive"] = str(Path(archive_path).expanduser().resolve())
    else:
        source_payload["mode"] = "metadata_only"

    if metadata_jsonl is not None:
        metadata_target = raw_root / "metadata.jsonl"
        ensure_parent_dir(metadata_target)
        if symlink:
            if metadata_target.exists() or metadata_target.is_symlink():
                metadata_target.unlink()
            metadata_target.symlink_to(metadata_jsonl.expanduser().resolve())
        else:
            shutil.copy2(metadata_jsonl.expanduser().resolve(), metadata_target)

    if folds_json is not None:
        folds_target = raw_root / "folds.json"
        ensure_parent_dir(folds_target)
        if symlink:
            if folds_target.exists() or folds_target.is_symlink():
                folds_target.unlink()
            folds_target.symlink_to(folds_json.expanduser().resolve())
        else:
            shutil.copy2(folds_json.expanduser().resolve(), folds_target)

    _normalize_media_root(raw_root, media_root, symlink=symlink)
    _write_json(raw_root / "source.json", source_payload)
    return raw_root


def build_common_arg_parser(parser, dataset_name: str, default_url: str = ""):
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_raw_output_root(),
        help="Root directory for raw FactorCL datasets.",
    )
    parser.add_argument(
        "--source-url",
        type=str,
        default=default_url,
        help="Optional remote archive URL for the raw dataset.",
    )
    parser.add_argument(
        "--local-archive",
        type=Path,
        default=None,
        help="Use a local raw archive instead of downloading.",
    )
    parser.add_argument(
        "--local-root",
        type=Path,
        default=None,
        help="Use a local canonical raw root and copy/symlink it into place.",
    )
    parser.add_argument(
        "--metadata-jsonl",
        type=Path,
        default=None,
        help="Canonical metadata.jsonl file to install into the raw root.",
    )
    parser.add_argument(
        "--folds-json",
        type=Path,
        default=None,
        help="Optional folds.json file describing fold-specific train/val/test assignments.",
    )
    parser.add_argument(
        "--media-root",
        type=Path,
        default=None,
        help="Optional directory containing media files referenced by metadata.jsonl.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink local sources instead of copying them.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing raw root contents.",
    )
    return parser


def finalize_raw_download(raw_root: Path, *, dataset_name: str) -> None:
    print(f"[{dataset_name}_raw] ready: {raw_root}")
    print(f"[{dataset_name}_raw] metadata={raw_root / 'metadata.jsonl'}")
    if (raw_root / "folds.json").exists():
        print(f"[{dataset_name}_raw] folds={raw_root / 'folds.json'}")
