import os
import json
import random
import pynvml
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from datasets import load_dataset
from collections import Counter
import multiprocessing
from accelerate import Accelerator
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

LETTERS_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def load_scienceqa_filtered(
    data_root: str,
    split: str,
    require_image: bool = True,
    require_outside_knowledge: bool = True,
    drop_near_blank: bool = True,
    blank_std_thresh: float = 0.01,
):
    ds = load_dataset("derek-thomas/ScienceQA", split=split, cache_dir=data_root)

    keep_indices: List[int] = []

    for i, ex in enumerate(ds):
        pil_img = ex.get("image", None)
        has_img = pil_img is not None

        lecture = (ex.get("lecture") or "").strip()
        hint = (ex.get("hint") or "").strip()
        has_outside = (len(lecture) > 0) or (len(hint) > 0)

        if require_image and not has_img:
            continue
        if require_outside_knowledge and not has_outside:
            continue

        if drop_near_blank and has_img:
            t = to_tensor(pil_img)
            if t.std().item() < blank_std_thresh:
                continue

        keep_indices.append(i)

    print(
        f"[ScienceQA] Split={split}: kept {len(keep_indices)} / {len(ds)} examples "
        f"(image={require_image}, outside_k={require_outside_knowledge}, "
        f"drop_near_blank={drop_near_blank})"
    )

    return ds, keep_indices


# ================================================================
#   TEXT BUILDERS
# ================================================================

def build_scienceqa_hint_text(example: Dict[str, Any]) -> str:
    lecture = (example.get("lecture") or "").strip()
    hint = (example.get("hint") or "").strip()

    parts = []
    if lecture:
        parts.append("Science passage:\n" + lecture)
    if hint:
        parts.append("Hint:\n" + hint)

    return "\n\n".join(parts)


def build_question_text_with_letters(
    question: str,
    choices: List[str],
    letters: List[str],
) -> str:
    parts = []
    question = (question or "").strip()
    if question:
        parts.append("Question:\n" + question)

    assert len(choices) == len(letters)
    choice_lines = [f"({L}) {c}" for L, c in zip(letters, choices)]
    parts.append("Choices:\n" + "\n".join(choice_lines))

    return "\n\n".join(parts)

def compute_lengths_and_filter(
    raw_ds,
    keep_indices: List[int],
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    *,
    mode: str = "chars",   # "chars" or "words"
    include_hint: bool = True,
    include_qa: bool = True,
    build_hint_fn=None,
    build_qa_fn=None,
) -> Tuple[List[int], List[int]]:
    if build_hint_fn is None or build_qa_fn is None:
        raise ValueError("You must pass build_hint_fn and build_qa_fn.")
    if mode not in {"chars", "words"}:
        raise ValueError("mode must be 'chars' or 'words'")

    do_min = (min_len is not None) and (int(min_len) > 0)
    do_max = (max_len is not None) and (int(max_len) > 0)
    min_len = int(min_len) if do_min else None
    max_len = int(max_len) if do_max else None

    lengths: List[int] = []
    filtered_keep: List[int] = []

    LETTERS_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    for ridx in keep_indices:
        ex = raw_ds[ridx]

        hint_text = build_hint_fn(ex) if include_hint else ""
        choices = list(ex.get("choices", []))
        n = len(choices)
        letters = list(LETTERS_POOL[:n])
        qa_text = build_qa_fn(ex.get("question"), choices, letters) if include_qa else ""

        text = ""
        if include_hint and hint_text:
            text += str(hint_text)
        if include_qa and qa_text:
            if text:
                text += "\n\n"
            text += str(qa_text)

        L = len(text) if mode == "chars" else len(text.split())
        lengths.append(L)

        if do_min and L < min_len:
            continue
        if do_max and L > max_len:
            continue

        filtered_keep.append(ridx)

    return lengths, filtered_keep



def suggest_max_len_from_text_lens(
    text_lens: List[int],
    *,
    drop_top_frac: float = 0.01,
    round_to: int = 64,
    min_len: Optional[int] = None,
    max_len: Optional[int] = None,
    verbose: bool = True,
) -> Tuple[int, List[int], Dict[str, Any]]:
    """
    Suggest a max length cutoff from a list of proxy text lengths (chars or words),
    while respecting user-provided min/max bounds.

    Args:
        text_lens: list of per-sample lengths
        drop_top_frac: fraction of longest samples to drop (e.g. 0.01 = top 1%)
        round_to: round suggested cutoff to a nice multiple (64/128)
        min_len: lower bound (samples shorter than this are already filtered elsewhere)
        max_len: hard upper bound (never suggest above this)
        verbose: print stats

    Returns:
        suggested_max_len: int
        outlier_positions: positions in text_lens that exceed suggested_max_len
        stats: distribution + decision info
    """
    arr = np.asarray(text_lens, dtype=np.int64)
    n = arr.size
    if n == 0:
        raise ValueError("text_lens is empty")

    # ---- distribution stats ----
    stats = {
        "n": int(n),
        "min": int(arr.min()),
        "p50": int(np.percentile(arr, 50)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p97": int(np.percentile(arr, 97)),
        "p99": int(np.percentile(arr, 99)),
        "p995": int(np.percentile(arr, 99.5)),
        "max": int(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }

    # ---- percentile-based cutoff ----
    drop_top_frac = float(np.clip(drop_top_frac, 0.0, 0.5))
    keep_frac = 1.0 - drop_top_frac
    keep_k = max(1, int(round(keep_frac * n)))

    kth = np.partition(arr, keep_k - 1)[keep_k - 1]
    suggested = int(kth)

    # ---- clamp to user bounds ----
    if min_len is not None:
        suggested = max(int(min_len), suggested)
    if max_len is not None:
        suggested = min(int(max_len), suggested)

    # ---- round nicely ----
    if round_to and round_to > 1:
        suggested = int(np.ceil(suggested / round_to) * round_to)

    # ---- identify outliers ----
    outlier_positions = np.where(arr > suggested)[0].tolist()

    stats.update({
        "drop_top_frac": drop_top_frac,
        "suggested_max_len": int(suggested),
        "would_drop": int(len(outlier_positions)),
        "would_drop_frac": float(len(outlier_positions) / n),
        "user_min_len": min_len,
        "user_max_len": max_len,
    })

    if verbose:
        print(
            f"[lens] n={stats['n']} min={stats['min']} "
            f"p95={stats['p95']} p99={stats['p99']} max={stats['max']}"
        )
        print(
            f"[lens] suggested max_len={stats['suggested_max_len']} "
            f"(drop_top_frac={drop_top_frac}, "
            f"would_drop={stats['would_drop']} = {stats['would_drop_frac']*100:.2f}%)"
        )
        print(
            f"[lens] candidates: p95={stats['p95']} "
            f"p97={stats['p97']} p99={stats['p99']} p99.5={stats['p995']}"
        )

    return suggested, outlier_positions, stats


class ScienceQA_Dataset(Dataset):
    """
    item = {
        "id": sample_id,
        "data": {
            0: hint_text,
            1: qa_text,          # question + lettered choices
            2: img_tensor,
            3: choices,          # permuted choice texts
            4: letters,          # permuted letters
        },
        "label": correct_index  # LongTensor scalar
    }
    """

    def __init__(
        self,
        config: Dict[str, Any],
        split: str = "train",
        require_image: bool = True,
        require_outside_knowledge: bool = True,
        image_size: int = 224,
        drop_near_blank: bool = True,
        blank_std_thresh: float = 0.01,
    ):
        super().__init__()
        self.split = split.lower()
        self.image_size = image_size

        self.raw_ds, self.keep_indices = load_scienceqa_filtered(
            data_root=config.dataset.data_roots,
            split=self.split,
            require_image=require_image,
            require_outside_knowledge=require_outside_knowledge,
            drop_near_blank=drop_near_blank,
            blank_std_thresh=blank_std_thresh,
        )

        # self.text_lens, self.keep_indices = compute_lengths_and_filter(
        #     raw_ds=self.raw_ds,
        #     keep_indices=self.keep_indices,
        #     min_len=getattr(config.dataset, "min_chars", 200),
        #     max_len=getattr(config.dataset, "max_chars", 1800),
        #     mode=getattr(config.dataset, "length_mode", "chars"),
        #     build_hint_fn=build_scienceqa_hint_text,
        #     build_qa_fn=build_question_text_with_letters,
        # )

        # print( f"[ScienceQA] length filter: kept {len(self.keep_indices)} / {len(self.text_lens)} ")

        self.train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.eval_tf = self.train_tf

        if split=="train":
            stats = compute_label_stats_and_weights(self.raw_ds, self.keep_indices, weight_mode="inv_freq", normalize="mean1")
            self.weights = stats["weights"].clone().detach()

    def __len__(self):
        # return len(self.keep_indices)
        return 10

    def _load_image(self, pil_img: Image.Image):
        pil_img = pil_img.convert("RGB")
        if self.split == "train":
            return self.train_tf(pil_img)
        else:
            return self.eval_tf(pil_img)


    def __getitem__(self, idx: int):
        real_idx = self.keep_indices[idx]
        ex = self.raw_ds[real_idx]

        pil_img = ex["image"]
        img_tensor = self._load_image(pil_img)

        hint_text = build_scienceqa_hint_text(ex)
        choices = list(ex.get("choices", []))
        question = ex.get("question")
        orig_correct_idx = ex["answer"]
        n = len(choices)

        letters_for_question = LETTERS_POOL[:n]
        correct_idx = orig_correct_idx

        qa_text = build_question_text_with_letters(question=question,choices=choices,letters=letters_for_question)

        label = torch.tensor(correct_idx, dtype=torch.long)
        sample_id = f"{self.split}_{real_idx}"

        return {
            "id": sample_id,
            "data": {
                0: hint_text,
                1: qa_text,
                2: img_tensor,
                3: choices,
                4: letters_for_question,
            },
            "label": label,
        }

def scienceqa_collate_qwen(batch):
    hint_texts   = [b["data"][0] for b in batch]
    qa_texts     = [b["data"][1] for b in batch]
    images       = [b["data"][2] for b in batch]
    choices_list = [b["data"][3] for b in batch]
    letter_lists = [b["data"][4] for b in batch]
    labels       = [b["label"] for b in batch]
    ids          = [b["id"] for b in batch]

    image_batch = torch.stack(images, dim=0)
    label_batch = torch.stack(labels, dim=0)

    return {
        "data": {
            0: hint_texts,
            1: qa_texts,
            2: image_batch,
            3: choices_list,
            4: letter_lists,
        },
        "ids": ids,
        "label": label_batch,
    }


# ================================================================
#   DATALOADER WRAPPER
# ================================================================

class ScienceQA_Dataloader:
    def __init__(self, config):
        batch_size = config.training_params.batch_size

        g = torch.Generator()
        g.manual_seed(0)

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        self.collate_fn = scienceqa_collate_qwen

        def get_physical_gpu_count():
            try:
                pynvml.nvmlInit()
                count = pynvml.nvmlDeviceGetCount()
                pynvml.nvmlShutdown()
                return count
            except Exception as e:
                return f"Could not query NVML: {e}"

        total_cpus = multiprocessing.cpu_count()
        num_gpus = get_physical_gpu_count()
        workers_per_gpu = max(1, (total_cpus - 1) // num_gpus)
        # workers_per_gpu = 0

        print(
            f"[ScienceQA] GPUs: {torch.cuda.device_count()} (Phys: {num_gpus}) | SLURM: {os.environ.get('CUDA_VISIBLE_DEVICES', 'N/A')} | CPUs: {total_cpus} | Workers: {torch.cuda.device_count()}x{workers_per_gpu}={torch.cuda.device_count() * workers_per_gpu}")

        self.train_loader = DataLoader(
            ScienceQA_Dataset(
                config=config,
                split="train"
            ),
            batch_size=batch_size,
            shuffle=True,
            generator=g,
            worker_init_fn=seed_worker,
            collate_fn=self.collate_fn,
            # --- ADD THESE FOR H100 PERFORMANCE ---
            num_workers=workers_per_gpu,  # Start with 8-12 per GPU (e.g., 48 total if on one node)
            pin_memory=True,  # Speeds up CPU-to-GPU transfer
            prefetch_factor=2,  # Ensures workers stay ahead of the GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )

        self.valid_loader = DataLoader(
            ScienceQA_Dataset(
                config=config,
                split="validation"
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers_per_gpu,  # Start with 8-12 per GPU (e.g., 48 total if on one node)
            pin_memory=True,  # Speeds up CPU-to-GPU transfer
            prefetch_factor=2,  # Ensures workers stay ahead of the GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )

        self.test_loader = DataLoader(
            ScienceQA_Dataset(
                config=config,
                split="test"
            ),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
            num_workers=workers_per_gpu,  # Start with 8-12 per GPU (e.g., 48 total if on one node)
            pin_memory=True,  # Speeds up CPU-to-GPU transfer
            prefetch_factor=2,  # Ensures workers stay ahead of the GPU
            persistent_workers=True  # Keeps workers alive between epochs
        )

def compute_label_stats_and_weights(
    ds,
    keep_indices,
    weight_mode="inv_freq",   # "inv_freq" or "cui"
    beta=0.9999,              # only used for "cui"
    normalize="mean1",        # "none", "sumK", "mean1"
):

    counts = Counter()
    K = 0
    for i in keep_indices:
        ex = ds[i]
        ans = int(ex["answer"])
        counts[ans] += 1
        K = max(K, len(ex.get("choices", [])))

    # Build count vector of length K (e.g., 5 => A..E)
    count_vec = torch.zeros(K, dtype=torch.long)
    for k, v in counts.items():
        if 0 <= k < K:
            count_vec[k] = v

    total = int(count_vec.sum().item())
    freq = count_vec.float() / max(total, 1)

    # Convert counts -> weights
    if weight_mode == "inv_freq":
        # w_k = total / (K * count_k)
        w = total / (K * torch.clamp(count_vec.float(), min=1.0))
    elif weight_mode == "cui":
        # "Class-Balanced Loss" (Cui et al.): w_k ∝ (1 - beta) / (1 - beta^n_k)
        n = torch.clamp(count_vec.float(), min=1.0)
        w = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), n))
    else:
        raise ValueError(f"Unknown weight_mode: {weight_mode}")

    # Optional normalization
    if normalize == "sumK":
        w = w * (K / w.sum())
    elif normalize == "mean1":
        w = w / w.mean()
    elif normalize == "none":
        pass
    else:
        raise ValueError(f"Unknown normalize: {normalize}")

    return {
        "K": K,
        "counts": count_vec,
        "freq": freq,
        "weights": w,
        "keep_n": len(keep_indices),
        "total": total,
    }



if __name__ == "__main__":
    import types
    import matplotlib
    import matplotlib.pyplot as plt
    import torchvision.transforms.functional as F
    from collections import Counter
    matplotlib.use("Agg")

    config = types.SimpleNamespace()
    config.dataset = types.SimpleNamespace()
    config.training_params = types.SimpleNamespace()
    config.training_params.batch_size = 16

    stats = compute_label_stats_and_weights(config=config, split="train", weight_mode="inv_freq", normalize="mean1")
    print("Max choices K =", stats["K"])
    print("Counts:", stats["counts"].tolist())
    print("Freq  :", [round(x, 6) for x in stats["freq"].tolist()])
    print("Wts   :", [round(x, 6) for x in stats["weights"].tolist()])


    loader = ScienceQA_Dataloader(config)
    batch = next(iter(loader.train_loader))

    label_counts = Counter(batch["label"].tolist())
    print("Label distribution in batch:", label_counts)


    print("IDs:", batch["ids"])
    print("\n--- Hint text sample ---\n", batch["data"][0][0][:400], "…")
    print("\n--- Q/A text sample ---\n", batch["data"][1][0][:400], "…")
    print("Image batch shape:", batch["data"][2].shape)
    print("Label indices:", batch["label"])
    print("Choices for sample 0:", batch["data"][3][0])
    print("Letters for sample 0:", batch["data"][4][0])

    img_tensor = batch["data"][2][0]
    img = F.to_pil_image(img_tensor)

    plt.imshow(img)
    plt.title(f"Sample ID: {batch['ids'][0]}")
    plt.axis("off")
    plt.savefig("scienceqa_sample.png")
    print("Saved scienceqa_sample.png")

    print("Choices for sample 0:", batch["data"][3][0])
    print("Letters for sample 0:", batch["data"][4][0])

    img_tensor = batch["data"][2][0]
    img = F.to_pil_image(img_tensor)

    plt.imshow(img)
    plt.title(f"Sample ID: {batch['ids'][0]}")
    plt.axis("off")
    plt.savefig("scienceqa_sample.png")
    print("Saved scienceqa_sample.png")
