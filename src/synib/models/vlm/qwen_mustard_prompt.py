"""
Qwen3-VL prompt models for MUStARD sarcasm detection.

Three classes are provided:
  QwenVL_MUStARD_Prompt          – combined (video + text) baseline
  QwenVL_MUStARD_Prompt_TextOnly – text unimodal baseline (no video frames)
  QwenVL_MUStARD_Prompt_VideoOnly– video unimodal baseline (no utterance/context)

All three share the same __init__ (inherited from _QwenVL_PromptFrozenCLSImpl)
and differ only in their forward() implementation.

Batch schema expected in x (= served_dict["data"]):
    x["text"]                 list[str]           utterance transcripts
    x["speaker"]              list[str]           speaker names
    x["context"]              list[list[str]]     preceding context turns
    x["video_frames"]         FloatTensor (B,F,3,H,W)  frames in [0,1]
    x["attention_mask_video"] LongTensor  (B,F)   1=real frame, 0=padding
"""

import torch
from torchvision.transforms.functional import to_pil_image

from synib.models.vlm.qwen_base_models import _QwenVL_PromptFrozenCLSImpl

__all__ = [
    "QwenVL_MUStARD_Prompt",
    "QwenVL_MUStARD_Prompt_TextOnly",
    "QwenVL_MUStARD_Prompt_VideoOnly",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_optional_vision(backbone, input_ids, attention_mask, pixel_values, image_grid_thw):
    """Call backbone with or without vision tensors depending on availability."""
    kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )
    if pixel_values is not None:
        kwargs["pixel_values"] = pixel_values.contiguous()
    if image_grid_thw is not None:
        kwargs["image_grid_thw"] = image_grid_thw.to(dtype=torch.long).contiguous()
    out = backbone(**kwargs)
    return out.hidden_states[-1]  # (B, T, d)


# ---------------------------------------------------------------------------
# Base MUStARD impl  (inherits all __init__ machinery from the ScienceQA base)
# ---------------------------------------------------------------------------

class _QwenVL_MUStARD_PromptImpl(_QwenVL_PromptFrozenCLSImpl):
    """
    MUStARD-specific extension of _QwenVL_PromptFrozenCLSImpl.

    Overrides forward() to accept the MUStARD batch schema and builds
    sarcasm-detection prompts with interleaved multi-frame image tokens.
    """

    # ------------------------------------------------------------------ #
    #  Prompt construction                                                 #
    # ------------------------------------------------------------------ #

    def _build_sarcasm_messages(self, texts, speakers, contexts, num_frames_per_sample):
        """
        Build a list of chat message dicts suitable for apply_chat_template.

        For each sample, the user message contains:
            - F <image> entries (one per video frame)
            - A text block:  Speaker / Context / Utterance / <CLS>

        Returns:
            messages_batch  list[list[dict]]
        """
        max_ctx = getattr(self.args, "max_context_turns", 3)
        messages_batch = []

        for text, speaker, context, n_frames in zip(
            texts, speakers, contexts, num_frames_per_sample
        ):
            n_frames = int(n_frames)
            ctx_turns = list(context)[-max_ctx:] if context else []

            parts = [f"Speaker: {speaker}"]
            if ctx_turns:
                parts.append("Context:")
                parts.extend(ctx_turns)
            parts.append(f'\nUtterance: "{text}"')
            parts.append("\nIs the speaker being sarcastic? <CLS>")
            prompt_text = "\n".join(parts)

            content = [{"type": "image"} for _ in range(n_frames)]
            content.append({"type": "text", "text": prompt_text})

            messages_batch.append([{"role": "user", "content": content}])

        return messages_batch

    def _frames_to_pil(self, video_frames, num_frames_per_sample):
        """
        Convert (B, F, 3, H, W) → per-sample lists of PIL images.

        Only real frames (not padding) are included. The nesting matches the
        prompt batch structure expected by the processor for multi-image inputs.
        """
        B = video_frames.shape[0]
        pil_images = []
        for b in range(B):
            n = int(num_frames_per_sample[b])
            sample_images = []
            for f in range(n):
                frame = video_frames[b, f].detach().cpu().clamp(0.0, 1.0)
                sample_images.append(to_pil_image(frame))
            pil_images.append(sample_images)
        return pil_images

    # ------------------------------------------------------------------ #
    #  Forward                                                             #
    # ------------------------------------------------------------------ #

    def forward(
        self,
        x,
        *,
        label=None,
        return_features=False,
        current_step=None,
        image_token_mask=None,
        text_token_mask=None,
        **kwargs,
    ):
        texts = x["text"]           # list[str]
        speakers = x["speaker"]     # list[str]
        contexts = x["context"]     # list[list[str]]
        video_frames = x["video_frames"]              # (B, F, 3, H, W)
        attn_mask_video = x.get("attention_mask_video", None)  # (B, F) or None

        model_device = self.backbone.device
        B, F = video_frames.shape[:2]

        # Determine actual frame count per sample
        if attn_mask_video is not None:
            num_frames = attn_mask_video.sum(dim=1).long().tolist()
        else:
            num_frames = [F] * B

        # PIL images (only real frames, in order)
        pil_images = self._frames_to_pil(video_frames, num_frames)

        # Build prompts
        messages_batch = self._build_sarcasm_messages(texts, speakers, contexts, num_frames)
        prompts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        flat_images = [img for sample in pil_images for img in sample]
        proc = self.processor(
            text=prompts,
            images=flat_images if flat_images else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        proc = {k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in proc.items()}

        input_ids = proc["input_ids"]
        attention_mask = proc["attention_mask"]
        pixel_values = proc.get("pixel_values", None)
        image_grid_thw = proc.get("image_grid_thw", None)

        hidden = _encode_optional_vision(
            self.backbone, input_ids, attention_mask, pixel_values, image_grid_thw
        )

        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        head_logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            if torch.is_tensor(label):
                label = label.to(head_logits.device)
            losses["ce_loss_combined"] = self._mc_ce_loss(head_logits, label)

        # Optional generation-based reasoning at eval time
        if (not self.training) and getattr(self.args, "generate_reasoning", False):
            reasoning = self.generate_answer(
                proc,
                max_new_tokens=getattr(self.args, "max_new_tokens", 256),
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                min_new_tokens=10,
                strip_prompt=True,
            )
            preds_out = {"combined": head_logits, "reasoning": reasoning}
        else:
            preds_out = {"combined": head_logits}

        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        return {"preds": preds_out, "features": features, "losses": losses}


# ---------------------------------------------------------------------------
# Text-only unimodal baseline
# ---------------------------------------------------------------------------

class _QwenVL_MUStARD_TextOnlyImpl(_QwenVL_MUStARD_PromptImpl):
    """
    Text unimodal baseline.

    Builds the same sarcasm prompt but without any <image> tokens.
    No visual input is passed to the backbone.
    """

    def forward(
        self,
        x,
        *,
        label=None,
        return_features=False,
        current_step=None,
        **kwargs,
    ):
        texts = x["text"]
        speakers = x["speaker"]
        contexts = x["context"]

        model_device = self.backbone.device
        max_ctx = getattr(self.args, "max_context_turns", 3)

        prompts_text = []
        for text, speaker, context in zip(texts, speakers, contexts):
            ctx_turns = list(context)[-max_ctx:] if context else []
            parts = [f"Speaker: {speaker}"]
            if ctx_turns:
                parts.append("Context:")
                parts.extend(ctx_turns)
            parts.append(f'\nUtterance: "{text}"')
            parts.append("\nIs the speaker being sarcastic? <CLS>")
            prompts_text.append("\n".join(parts))

        messages_batch = [
            [{"role": "user", "content": [{"type": "text", "text": t}]}]
            for t in prompts_text
        ]
        prompts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        proc = self.processor(
            text=prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        proc = {k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in proc.items()}

        input_ids = proc["input_ids"]
        attention_mask = proc["attention_mask"]

        hidden = _encode_optional_vision(
            self.backbone, input_ids, attention_mask, None, None
        )

        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        head_logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            if torch.is_tensor(label):
                label = label.to(head_logits.device)
            losses["ce_loss_combined"] = self._mc_ce_loss(head_logits, label)

        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        return {"preds": {"combined": head_logits}, "features": features, "losses": losses}


# ---------------------------------------------------------------------------
# Video-only unimodal baseline
# ---------------------------------------------------------------------------

class _QwenVL_MUStARD_VideoOnlyImpl(_QwenVL_MUStARD_PromptImpl):
    """
    Video unimodal baseline.

    Builds a prompt with only the speaker name and video frames —
    no utterance text, no context.
    """

    def forward(
        self,
        x,
        *,
        label=None,
        return_features=False,
        current_step=None,
        **kwargs,
    ):
        speakers = x["speaker"]
        video_frames = x["video_frames"]
        attn_mask_video = x.get("attention_mask_video", None)

        model_device = self.backbone.device
        B, F = video_frames.shape[:2]

        if attn_mask_video is not None:
            num_frames = attn_mask_video.sum(dim=1).long().tolist()
        else:
            num_frames = [F] * B

        pil_images = self._frames_to_pil(video_frames, num_frames)

        messages_batch = []
        for speaker, n_frames in zip(speakers, num_frames):
            n_frames = int(n_frames)
            prompt_text = f"Speaker: {speaker}\nIs the speaker being sarcastic? <CLS>"
            content = [{"type": "image"} for _ in range(n_frames)]
            content.append({"type": "text", "text": prompt_text})
            messages_batch.append([{"role": "user", "content": content}])

        prompts = [
            self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
            for m in messages_batch
        ]

        flat_images = [img for sample in pil_images for img in sample]
        proc = self.processor(
            text=prompts,
            images=flat_images if flat_images else None,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        proc = {k: (v.to(model_device) if torch.is_tensor(v) else v) for k, v in proc.items()}

        input_ids = proc["input_ids"]
        attention_mask = proc["attention_mask"]
        pixel_values = proc.get("pixel_values", None)
        image_grid_thw = proc.get("image_grid_thw", None)

        hidden = _encode_optional_vision(
            self.backbone, input_ids, attention_mask, pixel_values, image_grid_thw
        )

        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        head_logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            if torch.is_tensor(label):
                label = label.to(head_logits.device)
            losses["ce_loss_combined"] = self._mc_ce_loss(head_logits, label)

        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        return {"preds": {"combined": head_logits}, "features": features, "losses": losses}


# ---------------------------------------------------------------------------
# Public wrappers (registered names used in configs)
# ---------------------------------------------------------------------------

class QwenVL_MUStARD_Prompt(_QwenVL_MUStARD_PromptImpl):
    """Combined (video + text) prompt model for MUStARD sarcasm detection."""
    pass


class QwenVL_MUStARD_Prompt_TextOnly(_QwenVL_MUStARD_TextOnlyImpl):
    """Text-only unimodal baseline for MUStARD."""
    pass


class QwenVL_MUStARD_Prompt_VideoOnly(_QwenVL_MUStARD_VideoOnlyImpl):
    """Video-only unimodal baseline for MUStARD."""
    pass
