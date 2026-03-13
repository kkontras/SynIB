#!/usr/bin/env python3
"""Download all FactorCL affect dataset prepared artifacts."""

import argparse
from pathlib import Path

try:
    from .download_mosi import DEFAULT_FILENAME as MOSI_FILENAME, DEFAULT_GDRIVE_FILE_ID as MOSI_ID, download_mosi
    from .download_mosei import DEFAULT_FILENAME as MOSEI_FILENAME, DEFAULT_GDRIVE_FILE_ID as MOSEI_ID, download_mosei
    from .download_mustard import (
        DEFAULT_FILENAME as MUSTARD_FILENAME,
        DEFAULT_GDRIVE_FILE_ID as MUSTARD_ID,
        download_mustard,
    )
    from .download_ur_funny import (
        DEFAULT_FILENAME as UR_FUNNY_FILENAME,
        DEFAULT_GDRIVE_FILE_ID as UR_FUNNY_ID,
        download_ur_funny,
    )
except ImportError:  # direct script execution fallback
    from download_mosi import DEFAULT_FILENAME as MOSI_FILENAME, DEFAULT_GDRIVE_FILE_ID as MOSI_ID, download_mosi
    from download_mosei import DEFAULT_FILENAME as MOSEI_FILENAME, DEFAULT_GDRIVE_FILE_ID as MOSEI_ID, download_mosei
    from download_mustard import DEFAULT_FILENAME as MUSTARD_FILENAME, DEFAULT_GDRIVE_FILE_ID as MUSTARD_ID, download_mustard
    from download_ur_funny import (
        DEFAULT_FILENAME as UR_FUNNY_FILENAME,
        DEFAULT_GDRIVE_FILE_ID as UR_FUNNY_ID,
        download_ur_funny,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download MOSI/MOSEI/UR-Funny/MUStARD prepared artifacts.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("src/synib/mydatasets/Factor_CL_Datasets/prepared").resolve(),
        help="Root directory for downloads.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output files.")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    root = args.output_root.resolve()
    force = bool(args.force)

    outputs = {
        "mosi": download_mosi(root, MOSI_ID, MOSI_FILENAME, force=force),
        "mosei": download_mosei(root, MOSEI_ID, MOSEI_FILENAME, force=force),
        "ur_funny": download_ur_funny(root, UR_FUNNY_ID, UR_FUNNY_FILENAME, force=force),
        "mustard": download_mustard(root, MUSTARD_ID, MUSTARD_FILENAME, force=force),
    }

    for name, path in outputs.items():
        print(f"[{name}] {path}")


if __name__ == "__main__":
    main()
