#!/usr/bin/env python3
"""Materialize a canonical raw CMU-MOSI root for downstream cache building."""

import argparse

try:
    from .raw_dataset_downloader import (
        build_common_arg_parser,
        finalize_raw_download,
        materialize_canonical_raw_root,
    )
except ImportError:  # direct script execution fallback
    from raw_dataset_downloader import (  # type: ignore
        build_common_arg_parser,
        finalize_raw_download,
        materialize_canonical_raw_root,
    )


DATASET_NAME = "mosi"
DEFAULT_SOURCE_URL = ""


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or materialize canonical raw CMU-MOSI data for cache building."
    )
    return build_common_arg_parser(parser, DATASET_NAME, default_url=DEFAULT_SOURCE_URL)


def main() -> None:
    args = build_arg_parser().parse_args()
    raw_root = materialize_canonical_raw_root(
        dataset_name=DATASET_NAME,
        output_root=args.output_root.resolve(),
        source_url=args.source_url.strip() or None,
        local_archive=args.local_archive,
        local_root=args.local_root,
        metadata_jsonl=args.metadata_jsonl,
        folds_json=args.folds_json,
        media_root=args.media_root,
        symlink=bool(args.symlink),
        force=bool(args.force),
    )
    finalize_raw_download(raw_root, dataset_name=DATASET_NAME)


if __name__ == "__main__":
    main()
