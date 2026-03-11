from typing import Any, Dict, List
from torch.utils.data import Dataset, DataLoader
import os
import glob
import zipfile
import urllib.request
import multiprocessing
import numpy as np
import random

import torch
from torchvision import transforms
from PIL import Image

import pandas as pd
from datasets import Dataset as HFDataset  # create a HF-style Dataset from a pandas df
import logging

LABEL2IDX = {"entailment": 0, "neutral": 1, "contradiction": 2}
IDX2LABEL = {v: k for k, v in LABEL2IDX.items()}


# -------------------------
# Flickr30k (HF zip) helper
# -------------------------
from huggingface_hub import hf_hub_download


def ensure_flickr30k(
    flickr_images_root: str,
    hf_cache_root: str = "./hf_cache",
    repo_id: str = "nlphuji/flickr30k",
    zip_name: str = "flickr30k-images.zip",
    logger=logging.getLogger("Flickr30k")
):
    """
    Downloads Flickr30k images zip from the HF dataset repo and extracts it.

    After extraction, images are typically under:
        <flickr_images_root>/flickr30k-images/
    and named like:
        1000092795.jpg, ...

    This avoids datasets.load_dataset(...) entirely.
    """
    os.makedirs(flickr_images_root, exist_ok=True)

    extracted_dir = os.path.join(flickr_images_root, "flickr30k-images")
    if os.path.isdir(extracted_dir) and len(os.listdir(extracted_dir)) > 0:
        logger.info(f" Found extracted images at: {extracted_dir}")
        return extracted_dir

    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=zip_name,
        repo_type="dataset",
        cache_dir=hf_cache_root,
    )
    logger.info(f" Downloaded zip to: {zip_path}")

    logger.info(f" Extracting to: {flickr_images_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(flickr_images_root)

    if os.path.isdir(extracted_dir):
        logger.info(f" Extracted images at: {extracted_dir}")
        return extracted_dir

    logger.info(f" WARNING: expected {extracted_dir} not found; using root: {flickr_images_root}")
    return flickr_images_root


def find_flickr30k_image_path(flickr_id, flickr_images_root):
    if flickr_id is None:
        return None
    fid = str(flickr_id).strip()
    if fid.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        p = os.path.join(flickr_images_root, fid)
        return p if os.path.isfile(p) else None
    p = os.path.join(flickr_images_root, fid + ".jpg")
    return p if os.path.isfile(p) else None


# -------------------------
# e-SNLI-VE (GitHub) helper
# -------------------------
def _download_url(url: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.isfile(out_path) and os.path.getsize(out_path) > 0:
        return out_path
    self.logger.info(f"[GitHub] Downloading: {url}")
    urllib.request.urlretrieve(url, out_path)
    return out_path


def ensure_esnli_ve_repo(
    cache_root: str,
    source: str = "evil",  # "evil" (recommended) or "virginie"
    logger=None,
):
    """
    Downloads & extracts a repo zip that contains e-SNLI-VE splits.

    source="evil":     https://github.com/maximek3/e-ViL (recommended)
    source="virginie": https://github.com/virginie-do/e-SNLI-VE (deprecated)
    """
    os.makedirs(cache_root, exist_ok=True)

    if source == "evil":
        zip_url = "https://github.com/maximek3/e-ViL/archive/refs/heads/main.zip"
        zip_name = "e-ViL-main.zip"
        extracted_folder_name = "e-ViL-main"
    elif source == "virginie":
        zip_url = "https://github.com/virginie-do/e-SNLI-VE/archive/refs/heads/master.zip"
        zip_name = "e-SNLI-VE-master.zip"
        extracted_folder_name = "e-SNLI-VE-master"
    else:
        raise ValueError(f"Unknown source={source!r}. Use 'evil' or 'virginie'.")

    zip_path = os.path.join(cache_root, zip_name)
    extract_root = os.path.join(cache_root, extracted_folder_name)

    if os.path.isdir(extract_root) and len(os.listdir(extract_root)) > 0:
        return extract_root

    _download_url(zip_url, zip_path)

    # self.logger.info(f"[GitHub] Extracting: {zip_path} -> {cache_root}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(cache_root)

    if not os.path.isdir(extract_root):
        candidates = [d for d in glob.glob(os.path.join(cache_root, "*")) if os.path.isdir(d)]
        raise RuntimeError(f"Could not find extracted repo folder. Candidates: {candidates[:10]}")

    return extract_root


def find_esnli_ve_split_files(repo_root: str):
    """
    Finds train/dev/test split files inside repo_root.
    Supports CSV/TSV. Handles train split being split into multiple files.
    """
    data_dir = os.path.join(repo_root, "data")
    if not os.path.isdir(data_dir):
        hits = glob.glob(os.path.join(repo_root, "**", "data"), recursive=True)
        if hits:
            data_dir = hits[0]

    exts = ("*.csv", "*.tsv")
    all_files = []
    for ext in exts:
        all_files.extend(glob.glob(os.path.join(data_dir, "**", ext), recursive=True))

    if not all_files:
        raise RuntimeError(f"No CSV/TSV files found under {data_dir}")

    def pick(split_keywords):
        return [f for f in all_files if any(k in os.path.basename(f).lower() for k in split_keywords)]

    train_files = sorted(pick(["train"]))
    dev_files = pick(["dev", "valid", "val"])
    test_files = pick(["test"])

    def choose_one(files):
        if not files:
            return None
        return sorted(files, key=lambda p: (len(os.path.basename(p)), os.path.basename(p)))[0]

    dev_one = choose_one(dev_files)
    test_one = choose_one(test_files)

    return {
        "train": train_files,
        "dev": [dev_one] if dev_one else [],
        "test": [test_one] if test_one else [],
    }


def load_esnli_ve_from_repo(cache_root: str, split: str, source: str = "evil", logger=None):
    """
    Returns a HF Dataset for the requested split.
    Loads CSV/TSV from the extracted repo, merging multi-part train splits if needed.
    """
    repo_root = ensure_esnli_ve_repo(cache_root=cache_root, source=source, logger=logger)
    split_files = find_esnli_ve_split_files(repo_root)

    if split not in split_files or not split_files[split]:
        raise RuntimeError(f"Could not find files for split={split!r}. Found: {split_files}")

    files = split_files[split]
    logger.info(f" Using {len(files)} file(s) for split={split}:")

    dfs = []
    for f in files:
        sep = "\t" if f.lower().endswith(".tsv") else ","
        df = pd.read_csv(f, sep=sep, dtype=str, keep_default_na=False)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    return HFDataset.from_pandas(df, preserve_index=False)


def pick_first(ex: Dict[str, Any], keys: List[str], default=""):
    for k in keys:
        if k in ex and str(ex.get(k, "")).strip() != "":
            return ex.get(k)
    return default


# -------------------------
# Label helper
# -------------------------
def gold_label_to_index(gold):
    s = str(gold).strip().lower()
    if s in LABEL2IDX:
        return LABEL2IDX[s]
    raise ValueError(f"Invalid gold_label: {gold!r}")


# ==========================================================
# KEEPING YOUR CLASS NAMES THE SAME (as you requested)
# ==========================================================
class ESNLI_VE_ClassificationDataset(Dataset):
    """
    Returns:
      {
        "id": str,
        "text": str,           # hypothesis (sentence2)
        "image": FloatTensor,  # [3,H,W]
        "label": LongTensor,   # {0,1,2}
      }
    """

    def __init__(
        self,
        config,  # expects config.dataset.data_roots
        split: str = "train",
        image_size: int = 224,
        drop_invalid_labels: bool = True,
        require_explanation: bool = False,
        # new: choose repo source ("evil" recommended)
        esnli_source: str = "evil",
    ):
        super().__init__()
        self.logger = logging.getLogger('ESNLI Dataset')
        self.split = split
        self.hf_cache_root = config.dataset.data_roots

        # 1) Ensure Flickr30k images are downloaded
        self.flickr_images_dir = ensure_flickr30k(
            flickr_images_root=self.hf_cache_root,
            hf_cache_root=self.hf_cache_root,
        )

        # 2) Load e-SNLI-VE from GitHub repo zip (v2-ish packaging)
        #    We map split names to repo conventions: train/dev/test
        repo_split = split
        if split == "validation":
            repo_split = "dev"
        self.ds = load_esnli_ve_from_repo(
            cache_root=self.hf_cache_root,
            split=repo_split,
            source=esnli_source,
            logger=self.logger,
        )

        # 3) Filter indices (labels valid, optional explanation)
        self.keep_indices = []
        for i in range(len(self.ds)):
            ex = self.ds[i]

            if drop_invalid_labels:
                gold = pick_first(ex, ["gold_label", "label", "Gold_Label"], default=None)
                try:
                    _ = gold_label_to_index(gold)
                except Exception:
                    continue

            if require_explanation:
                expl = pick_first(ex, ["Explanation_1", "explanation", "explanation_1"], default="")
                if not str(expl).strip():
                    continue

            # Optional image existence check (commented out to avoid slow scans)
            # fid = str(pick_first(ex, ["Flikr30kID","Flickr30kID","flickr30k_id","image_id"], default="")).strip()
            # img_path = find_flickr30k_image_path(fid, self.flickr_images_dir)
            # if img_path is None:
            #     continue

            self.keep_indices.append(i)

        self.tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

        self.logger.info(f" split={split} kept {len(self.keep_indices)} / {len(self.ds)}")

    def __len__(self):
        return len(self.keep_indices)
        # return 8

    def __getitem__(self, idx: int):
        real_idx = self.keep_indices[idx]
        ex = self.ds[real_idx]

        text = str(
            pick_first(ex, ["sentence2", "hypothesis", "Sentence2"], default="")
        ).strip()

        gold = pick_first(ex, ["gold_label", "label", "Gold_Label"], default=None)
        label = torch.tensor(gold_label_to_index(gold), dtype=torch.long)

        fid = str(
            pick_first(ex, ["Flikr30kID", "Flickr30kID", "flickr30k_id", "image_id"], default="")
        ).strip()
        img_path = find_flickr30k_image_path(fid, self.flickr_images_dir)
        if img_path is None:
            raise FileNotFoundError(
                f"Could not find Flickr30k image for id={fid!r} under {self.flickr_images_dir}"
            )

        with Image.open(img_path) as im:
            im = im.convert("RGB")
            image = self.tf(im)

        return {
            "id": f"{self.split}_{real_idx}",
            "text": text,
            "image": image,
            "label": label,
        }


def collate_esnli_ve(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "data":{
            0 : [b["text"] for b in batch],
            1 : torch.stack([b["image"] for b in batch], dim=0),
        },
        "id": [b["id"] for b in batch],
        "label": torch.stack([b["label"] for b in batch], dim=0),
    }


class ESNLI_VE_Dataloader:

    def __init__(self, config):
        batch_size = config.training_params.batch_size
        self.logger = logging.getLogger('ESNLI DataLoader')
        g = torch.Generator()
        g.manual_seed(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.collate_fn = collate_esnli_ve

        total_cpus = multiprocessing.cpu_count()
        workers = max(1, min(24, total_cpus - 1))
        workers = 0

        self.logger.info(f" CPUs: {total_cpus} | Workers: {workers}")

        self.train_loader = DataLoader(
            ESNLI_VE_ClassificationDataset(config=config, split="train"),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
            collate_fn=self.collate_fn,
            num_workers=workers,
            pin_memory=True,
            drop_last=True,
        )

        self.valid_loader = DataLoader(
            ESNLI_VE_ClassificationDataset(config=config, split="dev"),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        self.test_loader = DataLoader(
            ESNLI_VE_ClassificationDataset(config=config, split="test"),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )


# ---- simple main for exploration (EasyDict config, no argparse) ----
def main():
    from easydict import EasyDict as edict
    from torch.utils.data import DataLoader

    cfg = edict(
        {
            "dataset": {
                "data_roots": "/esat/smcdata/users/kkontras/Image_Dataset/no_backup/ESNLI",
            },
            "training_params": {"batch_size": 8},
        }
    )

    split = "train"
    image_size = 224
    n_print = 3

    dl = ESNLI_VE_Dataloader(config=cfg).valid_loader

    batch = next(iter(dl))
    print(batch.keys())
    print(batch["data"].keys())
    print(batch["data"][0])
    print(f"Batch image: {tuple(batch['data'][0].shape)}  labels: {batch['data'][1].shape}")

    for i in range(min(n_print, len(batch["id"]))):
        y = int(batch["label"][i].item())
        print("-" * 80)
        print("id:", batch["id"][i])
        print("label:", y, f"({IDX2LABEL.get(y, '?')})")
        print("text:", batch["text"][i][:400] + ("..." if len(batch["text"][i]) > 400 else ""))


if __name__ == "__main__":
    main()
