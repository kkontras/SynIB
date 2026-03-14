#!/usr/bin/env python3
"""
Download raw MUStARD video clips and metadata from the soujanyaporia/MUStARD GitHub repo.

The git repo contains only metadata (data/sarcasm_data.json).
Video clips must be downloaded separately from HuggingFace:
    https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip

Output layout:
    <output_root>/mustard_raw/
        videos/          # .mp4 clips (copies or symlinks)
        metadata.json    # [{id, utterance, speaker, sarcasm, context, context_speakers, show}, ...]

Usage:
    python -m synib.mydatasets.MUStARD.download_mustard_raw [options]
    ./run/mustard/download.sh [options]

Options:
    --output-root PATH   Root directory for downloaded data
    --force              Re-download and overwrite existing files
    --local-repo PATH    Path to an already-cloned MUStARD repository (skips git clone)
    --video-zip PATH     Path to an already-downloaded mmsd_raw_data.zip (skips HuggingFace download)
    --symlink            Symlink video files instead of copying (requires --local-repo or --video-zip)
    --verbose            Enable debug logging
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from pathlib import Path

log = logging.getLogger(__name__)

REPO_URL = "https://github.com/soujanyaporia/MUStARD"
VIDEO_ZIP_URL = "https://huggingface.co/datasets/MichiganNLP/MUStARD/resolve/main/mmsd_raw_data.zip"
DATASET_NAME = "mustard_raw"

REQUIRED_FIELDS = {"utterance", "speaker", "sarcasm", "context", "show"}


def default_output_root() -> Path:
    env_root = os.getenv("SYNIB_MUSTARD_DATA_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path("src/synib/mydatasets/MUStARD/prepared").resolve()


def _clone_repo(dest: Path) -> None:
    log.info("Cloning MUStARD repo from %s into %s", REPO_URL, dest)
    subprocess.run(
        ["git", "clone", "--depth=1", REPO_URL, str(dest)],
        check=True,
    )


def _find_sarcasm_data(repo_root: Path) -> Path:
    """Locate sarcasm_data.json — it lives at data/sarcasm_data.json in the repo."""
    candidates = [
        repo_root / "data" / "sarcasm_data.json",
        repo_root / "sarcasm_data.json",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"sarcasm_data.json not found in repo under {repo_root}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def _find_utterances_dir(search_roots: list[Path]) -> Path | None:
    """Locate utterances_final/ directory, searching multiple roots."""
    candidates = []
    for root in search_roots:
        candidates.extend([
            root / "utterances_final",
            root / "data" / "utterances_final",
        ])
    for p in candidates:
        if p.is_dir():
            return p
    return None


def _download_video_zip(dest_path: Path) -> None:
    log.info("Downloading video zip from %s", VIDEO_ZIP_URL)
    log.info("  → saving to %s (this may take a few minutes, ~1.2 GB)", dest_path)

    def _progress(block_num, block_size, total_size):
        if total_size > 0:
            pct = min(100, block_num * block_size * 100 // total_size)
            print(f"\r  {pct}% ", end="", flush=True)

    urllib.request.urlretrieve(VIDEO_ZIP_URL, str(dest_path), reporthook=_progress)
    print()  # newline after progress
    log.info("Download complete: %s", dest_path)


def _extract_utterances_from_zip(zip_path: Path, extract_to: Path) -> Path | None:
    """Extract utterances_final/ from the video zip. Returns the extracted dir or None."""
    log.info("Inspecting zip %s …", zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()

    # Find the utterances_final/ prefix inside the zip
    utterances_prefix = None
    for name in names:
        if "utterances_final" in name and name.endswith(".mp4"):
            # e.g. "mmsd_raw_data/utterances_final/1_60.mp4"
            idx = name.find("utterances_final")
            utterances_prefix = name[: idx + len("utterances_final")]
            break

    if utterances_prefix is None:
        log.warning("No utterances_final/*.mp4 found in zip — zip entries: %s", names[:10])
        return None

    log.info("Extracting '%s' from zip to %s …", utterances_prefix, extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.startswith(utterances_prefix) and member.endswith(".mp4"):
                # Flatten into extract_to/utterances_final/<filename>
                filename = Path(member).name
                target = extract_to / "utterances_final" / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(member) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

    result = extract_to / "utterances_final"
    n = len(list(result.glob("*.mp4")))
    log.info("Extracted %d mp4 files to %s", n, result)
    return result


def _parse_metadata(sarcasm_data_path: Path) -> list[dict]:
    with sarcasm_data_path.open() as f:
        raw = json.load(f)

    records = []
    for clip_id, entry in raw.items():
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            log.warning("Clip %s missing fields %s — skipping.", clip_id, missing)
            continue
        records.append(
            {
                "id": clip_id,
                "utterance": entry["utterance"],
                "speaker": entry["speaker"],
                "sarcasm": bool(entry["sarcasm"]),
                "context": list(entry["context"]),
                "context_speakers": list(entry.get("context_speakers", [])),
                "show": entry["show"],
            }
        )
    return records


def download_mustard_raw(
    output_root: Path,
    force: bool = False,
    local_repo: Path = None,
    video_zip: Path = None,
    symlink: bool = False,
) -> Path:
    out_dir = output_root / DATASET_NAME
    videos_dir = out_dir / "videos"
    metadata_path = out_dir / "metadata.json"

    if metadata_path.exists() and videos_dir.exists() and not force:
        log.info("Output already exists at %s — skipping (use --force to re-download).", out_dir)
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        # ── Step 1: get repo (for metadata) ──────────────────────────────────
        if local_repo is not None:
            repo_root = Path(local_repo).expanduser().resolve()
            log.info("Using local repo at %s", repo_root)
        else:
            repo_root = tmp_path / "MUStARD"
            _clone_repo(repo_root)

        sarcasm_data_path = _find_sarcasm_data(repo_root)
        log.info("Found sarcasm_data.json at %s", sarcasm_data_path)

        # ── Step 2: get videos ────────────────────────────────────────────────
        # First check if utterances_final/ is already present (local repo or zip extract)
        utterances_dir = _find_utterances_dir([repo_root])

        if utterances_dir is None:
            # Videos are not in the git repo — need the zip
            if video_zip is not None:
                zip_path = Path(video_zip).expanduser().resolve()
                if not zip_path.exists():
                    raise FileNotFoundError(f"--video-zip path does not exist: {zip_path}")
            else:
                zip_path = tmp_path / "mmsd_raw_data.zip"
                _download_video_zip(zip_path)

            extract_root = tmp_path / "videos_extracted"
            utterances_dir = _extract_utterances_from_zip(zip_path, extract_root)

        if utterances_dir is None:
            raise FileNotFoundError(
                "utterances_final/ directory not found in repo or zip. "
                "Cannot locate .mp4 files."
            )

        log.info("Using utterances directory: %s", utterances_dir)

        # ── Step 3: parse metadata and copy/symlink videos ───────────────────
        records = _parse_metadata(sarcasm_data_path)
        log.info("Parsed %d clips from sarcasm_data.json.", len(records))

        kept = []
        skipped = 0
        for rec in records:
            src_video = utterances_dir / f"{rec['id']}.mp4"
            if not src_video.exists():
                log.warning("Video not found for clip %s — skipping.", rec["id"])
                skipped += 1
                continue

            dst_video = videos_dir / src_video.name
            if dst_video.exists() and not force:
                pass  # already there
            elif symlink:
                if dst_video.is_symlink() or dst_video.exists():
                    dst_video.unlink()
                dst_video.symlink_to(src_video.resolve())
            else:
                shutil.copy2(src_video, dst_video)

            kept.append(rec)

        if skipped:
            log.warning("%d clips had missing video files and were skipped.", skipped)

        log.info("Writing metadata.json with %d clips.", len(kept))
        with metadata_path.open("w") as f:
            json.dump(kept, f, indent=2)

    return out_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download raw MUStARD video clips and metadata."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Root directory for downloaded data. "
             "Set SYNIB_MUSTARD_DATA_ROOT to override the default.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and overwrite existing files.",
    )
    parser.add_argument(
        "--local-repo",
        type=Path,
        default=None,
        help="Path to an already-cloned MUStARD repository. "
             "Skips the git clone step.",
    )
    parser.add_argument(
        "--video-zip",
        type=Path,
        default=None,
        help="Path to an already-downloaded mmsd_raw_data.zip. "
             "Skips the HuggingFace download step.",
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Symlink video files instead of copying them.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    out_dir = download_mustard_raw(
        output_root=args.output_root.resolve(),
        force=args.force,
        local_repo=args.local_repo,
        video_zip=args.video_zip,
        symlink=args.symlink,
    )

    print(f"[mustard_raw] ready: {out_dir}")
    print(f"[mustard_raw] set config.dataset.data_roots={out_dir / 'metadata.json'}")
    print(f"[mustard_raw] videos at: {out_dir / 'videos'}")


if __name__ == "__main__":
    main()
