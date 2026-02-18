
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ScienceQA → Qwen3-VL offline token cache builder.

What this script does (per example):
  - Builds the exact prompt you use (image token + "\n" + [hint] + "\n\n" + [QA] + "\n\n" + [instruction] + "\n\n" + "<CLS>")
  - Runs AutoProcessor once to get:
      input_ids, attention_mask, pixel_values, image_grid_thw
  - Builds per-token masks for:
      image / hint / question / choices / instruction / cls / other
    and a token_type_ids uint8 array encoding the same categories.
  - (Optional) also caches "vision tokens" as float embeddings by running the model's visual module:
      model.model.visual(pixel_values, grid_thw=image_grid_thw)
    NOTE: these are NOT integer IDs; they are embeddings (tensor or tuple/dict of tensors).

Outputs:
  out_dir/<split>/
    meta.json
    manifest.jsonl
    shard_00000.pt
    shard_00001.pt
    ...

Each shard is a list[dict] with keys:
  id, label,
  input_ids, attention_mask,
  token_type_ids, masks (dict of bool tensors),
  pixel_values, image_grid_thw (if present),
  image_embeds (optional; tensor OR dict[str,tensor] OR list[tensor])

Example:
  python cache_scienceqa_qwen3vl.py \
    --data_root /path/to/hf_cache \
    --out_dir /path/to/cache \
    --model_name Qwen/Qwen3-VL-2B-Instruct \
    --split train \
    --batch_size 16 \
    --num_workers 8 \
    --cache_image_embeds
"""
from torchvision import transforms

import os
import json
import argparse
from typing import List, Dict, Any, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import to_tensor
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, AutoConfig

# Try importing Qwen3VL model class (preferred). If your transformers doesn't have it,
# fall back to AutoModelForCausalLM and access .model.visual if present.
try:
    from transformers import Qwen3VLForConditionalGeneration  # type: ignore
    HAVE_QWEN3VL_CLASS = True
except Exception:
    from transformers import AutoModelForCausalLM
    HAVE_QWEN3VL_CLASS = False


LETTERS_POOL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# =========================
# Filtering / loading
# =========================
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
        f"[ScienceQA] Split={split}: kept {len(keep_indices)} / {len(ds)} "
        f"(image={require_image}, outside_k={require_outside_knowledge}, "
        f"drop_near_blank={drop_near_blank})"
    )
    return ds, keep_indices


# =========================
# Text builders
# =========================
def build_scienceqa_hint_text(example: Dict[str, Any]) -> str:
    lecture = (example.get("lecture") or "").strip()
    hint = (example.get("hint") or "").strip()

    parts = []
    if lecture:
        parts.append("Science passage:\n" + lecture)
    if hint:
        parts.append("Hint:\n" + hint)
    return "\n\n".join(parts)


def build_question_text(question: str) -> str:
    q = (question or "").strip()
    return ("Question:\n" + q) if q else ""


def build_choices_text(choices: List[str], letters: List[str]) -> str:
    assert len(choices) == len(letters)
    lines = [f"({L}) {c}" for L, c in zip(letters, choices)]
    return "Choices:\n" + "\n".join(lines)


def build_instruction_text(letters: List[str]) -> str:
    if not letters:
        return ""
    letters_str = ", ".join(f"({L})" for L in letters)
    return f"Answer with one of: {letters_str}. Describe the image and give an explanation."

def _build_prompts_with_choices(hint_texts, qa_texts, choises_text, instruction_text):
    prompts = []
    for hint, qa, ch, instr in zip(hint_texts, qa_texts, choises_text, instruction_text):
        parts = []
        if hint is not None and hint.strip():
            parts.append(hint.strip())
        parts.append("\n")
        if qa is not None and qa.strip():
            parts.append(qa.strip())
        parts.append("\n")
        if ch is not None and ch.strip():
            parts.append(ch.strip())
        parts.append("\n")
        if instr is not None and instr.strip():
            parts.append(instr.strip())
        # Put CLS token at the END so it can attend to all previous tokens (causal LM)
        parts.append("<CLS>")
        prompts.append("\n\n".join(parts))
    return prompts


def build_full_prompt(
    *,
    processor,
    image_token_str: str,
    hint_text: str,
    question_text: str,
    choices_text: str,
    instr_text: str,
    cls_text: str = "<CLS>",
) -> str:
    """
    Mirrors your model logic:
      parts = [hint?, qa?, instr?, "<CLS>"]
      prompts = "\n\n".join(parts)
      prompts_with_image = image_token_str + "\n" + prompts
    Where qa is question + "\n\n" + choices (if both exist).
    """
    texts = _build_prompts_with_choices(hint_text, question_text, choices_text, instr_text)

    messages_batch = [
        [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": t},
        ]}]
        for t in texts
    ]
    prompts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in messages_batch
    ]
    return prompts


# =========================
# Token types / masks
# =========================
TT_PAD = 0
TT_IMAGE = 1
TT_HINT = 2
TT_QUESTION = 3
TT_CHOICES = 4
TT_INSTR = 5
TT_CLS = 6
TT_OTHER = 7


def _tokenize_no_special(tok, s: str) -> List[int]:
    if not s:
        return []
    return tok(s, add_special_tokens=False).input_ids

def _normalize_pos_to_B_3_1_T(position_ids: torch.Tensor, B: int) -> torch.Tensor:
    """
    Returns position_ids as (B,3,1,T) on CPU.

    Accepts common variants seen in practice:
      - (3,1,T)                  -> broadcast to (B,3,1,T) (only valid if B==1; else error)
      - (B,3,1,T)
      - (3,B,T) or (3,B,1,T)     -> permute to (B,3,1,T)
      - (B,3,T)                  -> view to (B,3,1,T)
    """
    pos = position_ids

    if pos.dim() == 3 and tuple(pos.shape[:2]) == (3, 1):
        # (3,1,T)
        if B != 1:
            raise RuntimeError(f"Got position_ids (3,1,T) but B={B}. Expected batched position_ids.")
        return pos.unsqueeze(0)  # (1,3,1,T)

    if pos.dim() == 4 and int(pos.shape[0]) == B and tuple(pos.shape[1:3]) == (3, 1):
        return pos  # (B,3,1,T)

    if pos.dim() == 3 and int(pos.shape[0]) == B and int(pos.shape[1]) == 3:
        # (B,3,T)
        return pos.unsqueeze(2)  # (B,3,1,T)

    if pos.dim() == 3 and int(pos.shape[0]) == 3 and int(pos.shape[1]) == B:
        # (3,B,T)
        return pos.permute(1, 0, 2).unsqueeze(2)  # (B,3,1,T)

    if pos.dim() == 4 and int(pos.shape[0]) == 3 and int(pos.shape[1]) == B:
        # (3,B,1,T)
        return pos.permute(1, 0, 2, 3)  # (B,3,1,T)

    raise RuntimeError(f"Unrecognized position_ids shape={tuple(pos.shape)} for B={B}")

def _stack_deep_levels_per_sample(deep_stack_viz_list: Any, B: int) -> List[torch.Tensor]:
    """
    Returns list length B, each element (K,64,2048).

    Supports deep level tensor shapes:
      - (64*B, 2048)   <- your case (packed batch into token axis)
      - (B, 64, 2048)
      - (64, 2048)     (only if B==1)
    """
    # Default empty
    if not isinstance(deep_stack_viz_list, (list, tuple)):
        return [torch.empty((0, 64, 2048), dtype=torch.float16) for _ in range(B)]

    levels = [t for t in deep_stack_viz_list if torch.is_tensor(t)]
    if len(levels) == 0:
        return [torch.empty((0, 64, 2048), dtype=torch.float16) for _ in range(B)]

    per_sample_levels: List[List[torch.Tensor]] = [[] for _ in range(B)]

    for lvl in levels:
        t = lvl.detach().cpu()

        # Case 1: packed: (64*B,2048)
        if t.dim() == 2 and int(t.shape[1]) == 2048 and int(t.shape[0]) == 64 * B:
            tb = t.view(B, 64, 2048)  # (B,64,2048)
            for i in range(B):
                per_sample_levels[i].append(tb[i])
            continue

        # Case 2: already batched: (B,64,2048)
        if t.dim() == 3 and int(t.shape[0]) == B and tuple(t.shape[1:]) == (64, 2048):
            for i in range(B):
                per_sample_levels[i].append(t[i])
            continue

        # Case 3: single sample: (64,2048)
        if t.dim() == 2 and tuple(t.shape) == (64, 2048):
            if B != 1:
                raise RuntimeError("deep_stack_viz level is (64,2048) but B>1; expected (64*B,2048) or (B,64,2048).")
            per_sample_levels[0].append(t)
            continue

        raise RuntimeError(
            f"Unexpected deep level shape {tuple(t.shape)}. "
            f"Expected (64*B,2048) or (B,64,2048) or (64,2048 for B==1). B={B}"
        )

    out: List[torch.Tensor] = []
    for i in range(B):
        if len(per_sample_levels[i]) == 0:
            out.append(torch.empty((0, 64, 2048), dtype=torch.float16))
        else:
            out.append(torch.cat([x.unsqueeze(0) for x in per_sample_levels[i]], dim=0))  # (K,64,2048)
    return out

def _tokenize_no_special(tok, text: str):
    # must return a list[int]
    return tok.encode(text, add_special_tokens=False)

def _find_subseq(haystack, needle, start=0, end=None):
    """
    Return first index i such that haystack[i:i+len(needle)] == needle, else -1.
    """
    if end is None:
        end = len(haystack)
    n = len(needle)
    if n == 0:
        return -1
    last = end - n
    for i in range(start, last + 1):
        if haystack[i:i+n] == needle:
            return i
    return -1

def build_token_type_ids(
    *,
    tok,
    input_ids_1d: torch.Tensor,
    image_token_id: int,
    cls_token_id: int,
    hint_text: str,
    question_text: str,
    choices_text: str,
    instr_text: str,
    search_after_images: bool = True,
    verbose: bool = False,
) -> torch.Tensor:
    """
    Debuggable span alignment:
      - mark image/CLS tokens by ID (hard rules)
      - locate each text block by subsequence search in input_ids
      - label matched spans

    Much easier to debug than 'pos += len(tokenize(text))' logic.
    """
    ids = input_ids_1d.tolist()[0]
    L = len(ids)

    ttid = torch.full((L,), TT_OTHER, dtype=torch.uint8)

    # 1) hard-mark image tokens
    img_id = int(image_token_id)
    for i, tid in enumerate(ids):
        if tid == img_id:
            ttid[i] = TT_IMAGE

    # define search window start
    start = 0
    if search_after_images:
        while start < L and ids[start] == img_id:
            start += 1

    # helper: label a found block
    def label_block(start, tt, text, name):
        text = (text or "").strip()
        if not text:
            return start

        needle = _tokenize_no_special(tok, text+"\n")
        i = _find_subseq(ids, needle[:-1], start=start, end=L)
        if i < 0:
            if verbose:
                print(f"[MISS] {name}: couldn't find tokens for text={text[:80]!r}")
                print(f"  needle_len={len(needle)} start={start} L={L}")
            return start

        # label span, but don't overwrite images/cls
        for k in range(i, min(i + len(needle)-1, L)):
            if ttid[k] not in (TT_IMAGE, TT_CLS):
                ttid[k] = tt

        if verbose:
            print(f"[HIT] {name}: span=({i}, {i+len(needle)}) len={len(needle)}")
        # move start forward so later blocks tend to match after this one
        start = i + len(needle)
        return start

    # 2) label blocks by searching real ids
    start = label_block(start, TT_HINT, hint_text, "hint")
    start = label_block(start, TT_QUESTION, question_text, "question")
    start = label_block(start, TT_CHOICES, choices_text, "choices")
    start = label_block(start, TT_INSTR, instr_text, "instr")

    # 3) hard-mark CLS tokens last (wins)
    cls_id = int(cls_token_id)
    for i, tid in enumerate(ids):
        if tid == cls_id:
            ttid[i] = TT_CLS

    return ttid


import einops
# =========================
# Dataset returning raw pieces (PIL + strings)
# =========================
class ScienceQA_Raw(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        require_image: bool = True,
        require_outside_knowledge: bool = True,
        drop_near_blank: bool = True,
        blank_std_thresh: float = 0.01,
        image_size: int = 224,
        max_samples: int = 0,
    ):
        super().__init__()
        self.split = split
        self.raw_ds, self.keep_indices = load_scienceqa_filtered(
            data_root=data_root,
            split=split,
            require_image=require_image,
            require_outside_knowledge=require_outside_knowledge,
            drop_near_blank=drop_near_blank,
            blank_std_thresh=blank_std_thresh,
        )

        if max_samples and max_samples > 0:
            self.keep_indices = self.keep_indices[:max_samples]

        self.train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        self.eval_tf = self.train_tf

    def __len__(self) -> int:
        return len(self.keep_indices)

    def _load_image(self, pil_img: Image.Image):
        pil_img = pil_img.convert("RGB")
        if self.split == "train":
            return self.train_tf(pil_img)
        else:
            return self.eval_tf(pil_img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ridx = self.keep_indices[idx]
        ex = self.raw_ds[ridx]

        img: Image.Image = self._load_image(ex["image"])

        hint_text = build_scienceqa_hint_text(ex)

        choices = list(ex.get("choices", []))
        n = len(choices)
        letters = list(LETTERS_POOL[:n])

        question_text = build_question_text(ex.get("question"))
        choices_text = build_choices_text(choices, letters)
        instr_text = build_instruction_text(letters)

        label = int(ex["answer"])
        sample_id = f"{self.split}_{ridx}"

        return {
            "id": sample_id,
            "image": img,
            "hint_text": hint_text,
            "question_text": question_text,
            "choices_text": choices_text,
            "instr_text": instr_text,
            "letters": letters,
            "choices": choices,
            "label": label,
        }


def raw_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:

    image_batch = torch.stack([b["image"] for b in batch], dim=0)

    return {
        "ids": [b["id"] for b in batch],
        "images": image_batch,
        "hint_texts": [b["hint_text"] for b in batch],
        "question_texts": [b["question_text"] for b in batch],
        "choices_texts": [b["choices_text"] for b in batch],
        "instr_texts": [b["instr_text"] for b in batch],
        "letters": [b["letters"] for b in batch],
        "choices": [b["choices"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


# =========================
# Optional: vision embeddings ("vision tokens") caching
# =========================
def _to_cpu_serializable(obj: Any) -> Any:
    """Convert tensors to CPU tensors recursively for saving."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().contiguous()
    if isinstance(obj, dict):
        return {k: _to_cpu_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_cpu_serializable(v) for v in obj]
    return obj

def _tensor_stats_line(name: str, t: Any) -> str:
    if not torch.is_tensor(t):
        return f"{name:22s}: {type(t)}"
    if t.numel() == 0:
        return f"{name:22s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} numel=0"
    x = t.detach()
    xf = x.float()
    mn = xf.min().item()
    mx = xf.max().item()
    mean = xf.mean().item()
    std = xf.std(unbiased=False).item()
    nonzero = (x != 0).float().mean().item() * 100.0
    return (f"{name:22s}: shape={tuple(t.shape)} dtype={t.dtype} device={t.device} "
            f"min={mn:.6g} max={mx:.6g} mean={mean:.6g} std={std:.6g} nonzero={nonzero:.2f}%")


def verify_shard(shard_path: str, n_show: int = 1) -> None:
    print(f"\n[verify] loading shard: {shard_path}")
    items = torch.load(shard_path, map_location="cpu")
    if not isinstance(items, (list, tuple)):
        raise TypeError(f"Shard is not list/tuple: {type(items)}")
    print(f"[verify] items: {len(items)}")

    n_show = min(int(n_show), len(items))
    for i in range(n_show):
        ex = items[i]
        print(f"\n[verify] example {i} id={ex.get('id', None)}")
        keys = [
            "input_ids",
            "attention_mask",
            "position_ids",
            "input_embeds",
            "visual_pos_masks",
            "deepstack_visual_embeds",
        ]
        for k in keys:
            print(_tensor_stats_line(k, ex.get(k, None)))


@torch.no_grad()
def compute_qwen3vl_visual_outputs(
    model,
    pixel_values: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor],
) -> Any:
    """
    Attempts to run the vision module to produce "vision tokens" (embeddings).

    Expected API in many Qwen-VL implementations:
      model.model.visual(pixel_values, grid_thw=image_grid_thw)

    Some versions may use different kwarg name; we try a couple.
    Return can be:
      - Tensor (B, N, D) or similar
      - Tuple/List of tensors
      - Dict of tensors (e.g., deepstack/base)
    """
    model.eval()

    if not hasattr(model, "model") or not hasattr(model.model, "visual"):
        raise RuntimeError("Model does not expose model.visual; cannot compute vision embeddings.")

    visual = model.model.visual

    # Try common signatures
    try:
        out = visual(pixel_values, grid_thw=image_grid_thw)
        return out
    except TypeError:
        pass

    try:
        out = visual(pixel_values=pixel_values, grid_thw=image_grid_thw)
        return out
    except TypeError:
        pass

    try:
        out = visual(pixel_values=pixel_values, image_grid_thw=image_grid_thw)
        return out
    except TypeError as e:
        raise RuntimeError(f"Could not call model.model.visual with known signatures: {e}")

@torch.no_grad()
def generate_answer(
        backbone,
        processor,
        proc,  # dict from self.processor(...), already includes images tensors if provided
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        min_new_tokens=20,
        strip_prompt=True,
        debug=False,
):
    backbone.eval()

    device = backbone.device

    # Move ONLY tensor entries to model device (keeps lists/strings untouched)
    gen_kwargs = {k: v.to(device) for k, v in proc.items() if torch.is_tensor(v)}

    if "input_ids" not in gen_kwargs or "attention_mask" not in gen_kwargs:
        raise ValueError("proc must contain at least input_ids and attention_mask")

    input_ids = gen_kwargs["input_ids"]
    attention_mask = gen_kwargs["attention_mask"]

    tok = processor.tokenizer
    eos_token_id = tok.eos_token_id
    pad_token_id = tok.pad_token_id

    # Avoid immediate stop if prompt ends with EOS (common with some chat templates)
    if eos_token_id is not None and input_ids.shape[1] > 1:
        if (input_ids[:, -1] == eos_token_id).all():
            input_ids = input_ids[:, :-1]
            attention_mask = attention_mask[:, :-1]
            gen_kwargs["input_ids"] = input_ids
            gen_kwargs["attention_mask"] = attention_mask

    gen_ids = backbone.generate(
        **gen_kwargs,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
        pad_token_id=pad_token_id,
        eos_token_id=eos_token_id,
        return_dict_in_generate=False,
    )

    # # Debug shapes
    # if debug:
    #     print("input_ids:", input_ids.shape)
    #     print("gen_ids:", gen_ids.shape)
    #     print("new tokens:", gen_ids.shape[1] - input_ids.shape[1])

    # Decode only generated continuation (recommended)
    if strip_prompt:
        prompt_len = input_ids.shape[1]
        gen_part = gen_ids[:, prompt_len:]
    else:
        gen_part = gen_ids

    texts = tok.batch_decode(
        gen_part,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    return texts

# =========================
# Cache builder
# =========================
@torch.no_grad()
def build_and_save_cache(
    *,
    data_root: str,
    out_dir: str,
    model_name: str,
    split: str,
    batch_size: int,
    num_workers: int,
    shard_size: int,
    max_samples: int,
    require_image: bool,
    require_outside_knowledge: bool,
    drop_near_blank: bool,
    blank_std_thresh: float,
    cache_image_embeds: bool,
    device: str,
    dtype: str,
):
    os.makedirs(out_dir, exist_ok=True)
    split_out = os.path.join(out_dir, split)
    os.makedirs(split_out, exist_ok=True)

    processor = AutoProcessor.from_pretrained(model_name, cache_dir=data_root)
    tok = processor.tokenizer
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Ensure CLS exists
    tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
    cls_token_id = int(tok.convert_tokens_to_ids("<CLS>"))

    # Get image token id/str from config
    cfg = AutoConfig.from_pretrained(model_name, cache_dir=data_root)
    if not hasattr(cfg, "image_token_id"):
        raise RuntimeError("Config has no image_token_id; cannot build image token masks.")
    image_token_id = int(getattr(cfg, "image_token_id"))
    image_token_str = tok.convert_ids_to_tokens(image_token_id)

    # Optional model for visual outputs caching
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        trust_remote_code=True,
    ).cuda()
    model.eval()

    ds = ScienceQA_Raw(
        data_root=data_root,
        split=split,
        require_image=require_image,
        require_outside_knowledge=require_outside_knowledge,
        drop_near_blank=drop_near_blank,
        blank_std_thresh=blank_std_thresh,
        max_samples=max_samples,
    )

    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=raw_collate,
        pin_memory=False,
    )

    manifest_path = os.path.join(split_out, "manifest.jsonl")
    if os.path.exists(manifest_path):
        os.remove(manifest_path)

    meta_path = os.path.join(split_out, "meta.json")

    shard_idx = 0
    shard_items: List[Dict[str, Any]] = []

    def flush_shard():
        nonlocal shard_idx, shard_items
        if not shard_items:
            return
        shard_file = os.path.join(split_out, f"shard_{shard_idx:05d}.pt")
        torch.save(shard_items, shard_file)
        with open(manifest_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"shard": os.path.basename(shard_file), "num_items": len(shard_items)}) + "\n")
        shard_items = []
        shard_idx += 1
        return shard_file

    pbar = tqdm(dl, desc=f"[cache] {split}", total=len(dl))

    for batch in pbar:
        ids = batch["ids"]
        images = batch["images"]
        hint_texts = batch["hint_texts"]
        question_texts = batch["question_texts"]
        choices_texts = batch["choices_texts"]
        letters_texts = batch["letters"]
        instr_texts = batch["instr_texts"]
        labels = batch["labels"]

        imgs255 = (images.clamp(0, 1) * 255.0).round().to(torch.uint8)

        full_texts = build_full_prompt(
            processor=processor,
            image_token_str=image_token_str,
            hint_text=hint_texts,
            question_text=question_texts,
            choices_text=choices_texts,
            instr_text=instr_texts,
            cls_text="<CLS>",
        )

        # Processor outputs (language tokens + image inputs)
        proc = processor(
            text=full_texts,
            images=imgs255,
            padding=True,      # pad for batching; we will store trimmed per-sample
            truncation=True,
            return_tensors="pt",
        )

        input_ids = proc["input_ids"].to(device)           # (B, T)
        attention_mask = proc["attention_mask"].to(device)     # (B, T)
        pixel_values = proc["pixel_values"].to(device)         # (B, C, H, W)
        image_grid_thw = proc["image_grid_thw"].to(device)  # (B, 3) typically


        try:
            with torch.no_grad():

                token_embeds = model.model.get_input_embeddings()(input_ids)  # (B,T,2048)

                image_embeds_list, deep_stack_viz_list = model.get_image_features(pixel_values, image_grid_thw)

                image_embeds_cat = torch.cat(image_embeds_list, dim=0).to(token_embeds.device, token_embeds.dtype)

                placeholder_mask, _ = model.model.get_placeholder_mask(
                    input_ids,
                    inputs_embeds=token_embeds,
                    image_features=image_embeds_cat,
                )
                placeholder_mask_2d = placeholder_mask[..., 0] if placeholder_mask.dim() == 3 else placeholder_mask  # (B,T)

                token_embeds = token_embeds.masked_scatter(placeholder_mask, image_embeds_cat)

                # Force fp16 for caching (your requirement)
                # token_embeds = token_embeds.to(torch.float16)

                attention_mask_tensor = (
                    attention_mask if not isinstance(attention_mask, dict) else attention_mask["full_attention"]
                )
                if attention_mask_tensor is not None and attention_mask_tensor.ndim == 4:
                    attention_mask_tensor = torch.diagonal(attention_mask_tensor[:, 0], dim1=1, dim2=2)
                    # Only apply conversion for floating point tensors (inverted masks)
                    if attention_mask_tensor.dtype.is_floating_point:
                        attention_mask_tensor = attention_mask_tensor / torch.finfo(attention_mask_tensor.dtype).min
                        attention_mask_tensor = (1.0 - attention_mask_tensor).int()
                position_ids, _ = model.model.get_rope_index(
                    input_ids,
                    image_grid_thw,
                    None,
                    attention_mask=attention_mask_tensor,
                )

            input_ids_cpu = input_ids.detach().cpu()
            attention_cpu = attention_mask.detach().cpu()
            placeholder_mask_2d_cpu = placeholder_mask_2d.detach().cpu()  # (B,T) bool
            token_embeds_cpu = token_embeds.detach().cpu()  # (B,T,2048) fp16
            position_ids_cpu = position_ids.detach().cpu()
            pos_b = _normalize_pos_to_B_3_1_T(position_ids_cpu, len(labels))  # CPU
            deep_per_sample = _stack_deep_levels_per_sample(deep_stack_viz_list, len(labels))


        except Exception as e:
            raise Exception(e)

        # gen_texts = generate_answer(
        #     backbone=model,
        #     processor=processor,
        #     proc=proc,
        #     max_new_tokens=500,  # labels are short; keep tiny for debugging
        #     do_sample=False,  # deterministic label output
        #     temperature=0.0,
        #     top_p=1.0,
        #     min_new_tokens=1,
        #     strip_prompt=False,
        #     debug=True,
        # )
        #
        # print("###NEW ONE####")
        # for t in gen_texts:
        #     print("-----")
        #     print(t)

        B = input_ids.size(0)
        for i in range(B):

            keep = attention_cpu[i].bool()  # (T,)
            if keep.sum().item() <= 0:
                raise RuntimeError(f"attention_mask had no True tokens for sample {i}")

            input_ids_keep = input_ids_cpu[i][keep].contiguous().unsqueeze(0)                 # (1,L)
            attention_keep = attention_cpu[i][keep].contiguous().unsqueeze(0)                 # (1,L)
            input_embeds_keep = token_embeds_cpu[i][keep].contiguous().unsqueeze(0)           # (1,L,2048)
            visual_pos_masks = placeholder_mask_2d_cpu[i][keep].contiguous().unsqueeze(0).bool()  # (1,L)
            pos_keep = pos_b[i][:, :, keep].contiguous()                                      # (3,1,L)
            deepstack_visual_embeds = deep_per_sample[i]

            ttid = build_token_type_ids(
                tok=tok,
                input_ids_1d=input_ids_keep,
                image_token_id=image_token_id,
                cls_token_id=cls_token_id,
                hint_text=hint_texts[i],
                question_text=question_texts[i],
                choices_text=choices_texts[i],
                instr_text=instr_texts[i]
            )

            masks = {
                "image": (ttid == TT_IMAGE),
                "hint": (ttid == TT_HINT),
                "question": (ttid == TT_QUESTION),
                "choices": (ttid == TT_CHOICES),
                "instr": (ttid == TT_INSTR),
                "cls": (ttid == TT_CLS),
                "other": (ttid == TT_OTHER),
            }

            item: Dict[str, Any] = {
                "id": ids[i],
                "label": labels[i].clone(),
                "token_type_ids": ttid.cpu(),
                "masks": {k: v.cpu() for k, v in masks.items()},

                "input_ids": input_ids_keep,  # (1,L)
                "attention_mask": attention_keep,  # (1,L)
                "position_ids": pos_keep,  # (3,1,L)
                "input_embeds": input_embeds_keep,  # (1,L,2048) fp16
                "visual_pos_masks": visual_pos_masks,  # (1,L) bool
                "deepstack_visual_embeds": deepstack_visual_embeds,  # (K,64,2048)

            }

            shard_items.append(item)

            if len(shard_items) >= shard_size:
                shard_path = flush_shard()
                # shard_path = flush_shard(items, split_out, shard_idx, manifest_path)
                verify_shard(shard_path, n_show=1)

    shard_path = flush_shard()
    verify_shard(shard_path, n_show=1)

    meta = {
        "model_name": model_name,
        "split": split,
        "cls_token_id": cls_token_id,
        "image_token_id": image_token_id,
        "image_token_str": image_token_str,
        "token_type_map": {
            "PAD": TT_PAD,
            "IMAGE": TT_IMAGE,
            "HINT": TT_HINT,
            "QUESTION": TT_QUESTION,
            "CHOICES": TT_CHOICES,
            "INSTR": TT_INSTR,
            "CLS": TT_CLS,
            "OTHER": TT_OTHER,
        },
        "cache_image_embeds": bool(cache_image_embeds),
        "dtype": dtype,
        "padding_side": tok.padding_side,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Wrote cache to: {split_out}")
    print(f"[OK] Manifest: {manifest_path}")
    print(f"[OK] Meta: {meta_path}")


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="HF cache directory")
    ap.add_argument("--out_dir", type=str, required=True, help="Output cache directory")
    ap.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-2B-Instruct")

    ap.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--shard_size", type=int, default=4096)
    ap.add_argument("--max_samples", type=int, default=0, help="0 means all filtered samples")

    ap.add_argument("--require_image", action="store_true", default=True)
    ap.add_argument("--require_outside_knowledge", action="store_true", default=True)
    ap.add_argument("--drop_near_blank", action="store_true", default=True)
    ap.add_argument("--blank_std_thresh", type=float, default=0.01)

    ap.add_argument("--cache_image_embeds", action="store_true",
                    help="Also cache vision 'tokens' (embeddings) via model.model.visual(...)")
    ap.add_argument("--device", type=str, default="cuda:0", help='device_map value, e.g. "cuda:0"')
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])

    args = ap.parse_args()

    build_and_save_cache(
        data_root=args.data_root,
        out_dir=args.out_dir,
        model_name=args.model_name,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shard_size=args.shard_size,
        max_samples=args.max_samples,
        require_image=args.require_image,
        require_outside_knowledge=args.require_outside_knowledge,
        drop_near_blank=args.drop_near_blank,
        blank_std_thresh=args.blank_std_thresh,
        cache_image_embeds=args.cache_image_embeds,
        device=args.device,
        dtype=args.dtype,
    )


if __name__ == "__main__":
    main()
#
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ESNLI" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens8B" --model_name "Qwen/Qwen3-VL-8B-Instruct" --split test
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens8B" --model_name "Qwen/Qwen3-VL-8B-Instruct" --split validation
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens8B" --model_name "Qwen/Qwen3-VL-8B-Instruct"
#
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens32B" --model_name "Qwen/Qwen3-VL-32B-Instruct" --split test
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens32B" --model_name "Qwen/Qwen3-VL-32B-Instruct" --split validation
# CUDA_VISIBLE_DEVICES=4 python ScienceQA_Codebook.py --data_root "/scratch/kkontras/ScienceQA" --out_dir "/scratch/kkontras/ScienceQA/cache_tokens32B" --model_name "Qwen/Qwen3-VL-32B-Instruct"

