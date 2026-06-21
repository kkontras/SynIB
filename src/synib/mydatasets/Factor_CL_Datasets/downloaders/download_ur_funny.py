#!/usr/bin/env python3
"""
Download UR-Funny prepared artifact used by MultiBench/FactorCL affect loader.
"""

import argparse
from pathlib import Path

try:
    from .common import default_output_root, download_google_drive_file, materialize_local_file
except ImportError:  # direct script execution fallback
    from common import default_output_root, download_google_drive_file, materialize_local_file


DATASET_NAME = "ur_funny"
DEFAULT_GDRIVE_FILE_ID = "1L5slPmYyhEVtwGyM1kgcFMjeBpXLZGT0"
DEFAULT_FILENAME = "ur_funny_data.pkl"


def download_ur_funny(
    output_root: Path,
    source_id: str,
    filename: str,
    force: bool = False,
    local_file: Path = None,
    symlink: bool = False,
) -> Path:
    output_dir = output_root / DATASET_NAME
    output_path = output_dir / filename
    if local_file is not None:
        return materialize_local_file(source_path=local_file, output_path=output_path, overwrite=force, symlink=symlink)
    return download_google_drive_file(file_id=source_id, output_path=output_path, overwrite=force)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download UR-Funny prepared artifact for FactorCL dataloader.")
    parser.add_argument("--output-root", type=Path, default=default_output_root(), help="Root directory for downloads.")
    parser.add_argument("--source-id", type=str, default=DEFAULT_GDRIVE_FILE_ID, help="Google Drive file id.")
    parser.add_argument("--filename", type=str, default=DEFAULT_FILENAME, help="Output filename.")
    parser.add_argument("--local-file", type=Path, default=None, help="Use local source file instead of Google Drive.")
    parser.add_argument("--symlink", action="store_true", help="When --local-file is used, symlink instead of copy.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output file.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_path = download_ur_funny(
        output_root=args.output_root.resolve(),
        source_id=args.source_id.strip(),
        filename=args.filename.strip(),
        force=bool(args.force),
        local_file=args.local_file,
        symlink=bool(args.symlink),
    )
    print(f"[ur_funny] ready: {output_path}")
    print(f"[ur_funny] set config.dataset.data_roots={output_path}")
    print("[ur_funny] use config.dataset.data_type=humor for current get_data.py logic.")


if __name__ == "__main__":
    main()
