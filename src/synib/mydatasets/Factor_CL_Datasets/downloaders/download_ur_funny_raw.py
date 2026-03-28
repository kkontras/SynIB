#!/usr/bin/env python3
"""Materialize a canonical raw UR-Funny root for downstream cache building."""

import argparse
import json
import pickle
from pathlib import Path

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


DATASET_NAME = "ur_funny"
DEFAULT_SOURCE_URL = ""


def _default_local_source_root(output_root: Path) -> Path:
    candidate = output_root.expanduser().resolve() / "raw_sources" / DATASET_NAME
    return candidate


def _find_video_dir(raw_root: Path) -> Path:
    candidates = [
        raw_root / "media",
        raw_root / "videos",
        raw_root / "urfunny2_videos",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate UR-FUNNY video directory under {raw_root}")


def _build_canonical_metadata(raw_root: Path, local_source_root: Path) -> None:
    humor_pkl = local_source_root / "humor.pkl"
    humor_text_pkl = local_source_root / "humor_raw_text.pkl"
    if not humor_pkl.exists() or not humor_text_pkl.exists():
        raise FileNotFoundError(
            f"Need both {humor_pkl} and {humor_text_pkl} to create UR-FUNNY metadata."
        )

    with humor_pkl.open("rb") as f:
        humor = pickle.load(f)
    with humor_text_pkl.open("rb") as f:
        humor_text = pickle.load(f)

    video_dir = _find_video_dir(raw_root)
    rel_video_dir = video_dir.relative_to(raw_root)

    split_map = {"train": "train", "valid": "val", "test": "test"}
    row_by_id = {}
    fold_payload = {"0": {"train": [], "val": [], "test": []}}

    for split_name, out_split in split_map.items():
        split = humor[split_name]
        ids = split["id"].reshape(-1).tolist()
        labels = split["labels"].reshape(-1).tolist()
        for sample_id_raw, label_raw in zip(ids, labels):
            sample_id = str(int(sample_id_raw))
            text_tokens = humor_text.get(int(sample_id_raw), [])
            text = " ".join(str(tok) for tok in text_tokens).strip()
            video_rel = str(rel_video_dir / f"{sample_id}.mp4")
            if not (raw_root / video_rel).exists():
                continue
            row_by_id[sample_id] = {
                "id": sample_id,
                "text": text,
                "label": int(label_raw),
                "video_path": video_rel,
                "folds": {"0": out_split},
            }
            fold_payload["0"][out_split].append(sample_id)

    metadata_path = raw_root / "metadata.jsonl"
    with metadata_path.open("w", encoding="utf-8") as f:
        for sample_id in sorted(row_by_id, key=lambda x: int(x)):
            f.write(json.dumps(row_by_id[sample_id]) + "\n")

    with (raw_root / "folds.json").open("w", encoding="utf-8") as f:
        json.dump(fold_payload, f, indent=2, sort_keys=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download or materialize canonical raw UR-Funny data for cache building."
    )
    parser = build_common_arg_parser(parser, DATASET_NAME, default_url=DEFAULT_SOURCE_URL)
    parser.add_argument(
        "--local-source-root",
        type=Path,
        default=None,
        help="Directory containing humor.pkl and humor_raw_text.pkl.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_root = args.output_root.resolve()
    raw_root = materialize_canonical_raw_root(
        dataset_name=DATASET_NAME,
        output_root=output_root,
        source_url=args.source_url.strip() or None,
        local_archive=args.local_archive,
        local_root=args.local_root,
        metadata_jsonl=args.metadata_jsonl,
        folds_json=args.folds_json,
        media_root=args.media_root,
        symlink=bool(args.symlink),
        force=bool(args.force),
    )
    if not (raw_root / "metadata.jsonl").exists():
        local_source_root = args.local_source_root or _default_local_source_root(output_root)
        if local_source_root.exists():
            _build_canonical_metadata(raw_root, local_source_root.resolve())
    finalize_raw_download(raw_root, dataset_name=DATASET_NAME)


if __name__ == "__main__":
    main()
