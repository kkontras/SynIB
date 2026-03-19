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
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from synib.models.vlm.qwen_base_models import _QwenVL_PromptFrozenCLSImpl, SynIB_QwenFaster

__all__ = [
    "QwenVL_MUStARD_Prompt",
    "QwenVL_MUStARD_Prompt_TextOnly",
    "QwenVL_MUStARD_Prompt_VideoOnly",
    "QwenVL_MUStARD_Prompt_SynIB",
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


# ---------------------------------------------------------------------------
# SynIB variant
# ---------------------------------------------------------------------------

class QwenVL_MUStARD_Prompt_SynIB(_QwenVL_MUStARD_PromptImpl):
    """
    SynIB regularised MUStARD combined model.

    Runs three forward passes per step at the token-embedding level:
        pass0 – original embeddings (used for the main CE loss)
        pass1 – vision tokens replaced by EMA noise  (text-only view)
        pass2 – text  tokens replaced by EMA noise   (vision-only view)

    The KL-based synergy losses are computed from pass1 / pass2 features
    via SynIB_QwenFaster.compute_training_losses().
    """

    def __init__(self, args, encs=None, **kwargs):
        super().__init__(args, encs=encs, **kwargs)
        self.synergy_weight = float(
            self.args.get("bias_infusion", {}).get("l", 0.0)
        )
        self.synib = SynIB_QwenFaster(args, [], self)

    # ------------------------------------------------------------------
    # Internal helper: run the LM from pre-built inputs_embeds
    # ------------------------------------------------------------------

    def _encode_from_inputs_embeds(self, inputs_embeds, attention_mask, deep_stack_viz=None, visual_pos_masks=None):
        out = self.backbone.model.language_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            deepstack_visual_embeds=deep_stack_viz,
            visual_pos_masks=visual_pos_masks,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        return out.hidden_states[-1]

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x,
        *,
        label=None,
        return_features=False,
        current_step=None,
        current_epoch=None,
        **kwargs,
    ):
        texts        = x["text"]
        speakers     = x["speaker"]
        contexts     = x["context"]
        video_frames = x["video_frames"]
        attn_mask_video = x.get("attention_mask_video", None)

        model_device = self.backbone.device
        B, F = video_frames.shape[:2]

        if attn_mask_video is not None:
            num_frames = attn_mask_video.sum(dim=1).long().tolist()
        else:
            num_frames = [F] * B

        # ── processor ────────────────────────────────────────────────────────
        pil_images     = self._frames_to_pil(video_frames, num_frames)
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

        input_ids      = proc["input_ids"]
        attention_mask = proc["attention_mask"]
        pixel_values   = proc.get("pixel_values", None)
        image_grid_thw = proc.get("image_grid_thw", None)

        # ── visual features ───────────────────────────────────────────────────
        if pixel_values is not None:
            image_embeds_split, deepstack_image_embeds = self.backbone.get_image_features(
                pixel_values.contiguous(),
                image_grid_thw.to(dtype=torch.long).contiguous(),
            )
            # get_image_features returns a tuple of per-image tensors (from torch.split);
            # concatenate into a single flat tensor for masked scatter
            image_embeds = torch.cat(image_embeds_split, dim=0)
            n_dsv = len(deepstack_image_embeds)
            deep_stack_viz = [deepstack_image_embeds[i] for i in range(n_dsv)]
        else:
            image_embeds, deep_stack_viz = None, []

        # ── build full inputs_embeds ──────────────────────────────────────────
        lm = self.backbone.model.language_model
        inputs_embeds = lm.embed_tokens(input_ids)             # (B, T, d)
        image_mask    = (input_ids == self.image_token_id)     # (B, T) bool

        if image_embeds is not None:
            inputs_embeds[image_mask] = image_embeds.to(inputs_embeds.dtype)

        text_mask   = attention_mask.bool() & (~image_mask)    # text positions
        vision_mask = image_mask & attention_mask.bool()       # vision positions

        # ── EMA update ────────────────────────────────────────────────────────
        if inputs_embeds[text_mask].numel() > 0:
            self.synib.z1_stats.ema_update(inputs_embeds[text_mask].detach())
        if inputs_embeds[vision_mask].numel() > 0:
            self.synib.z2_stats.ema_update(inputs_embeds[vision_mask].detach())
        for i in range(len(deep_stack_viz)):
            if deep_stack_viz[i].numel() > 0:
                self.synib.z2_deepstack_stats[i].ema_update(deep_stack_viz[i].detach())

        # ── perturbation config ───────────────────────────────────────────────
        pcfg = (
            self.args.get("perturb", {}) if isinstance(self.args, dict)
            else getattr(self.args, "perturb", {})
        )
        perturb_type = (
            pcfg.get("type", "random") if isinstance(pcfg, dict)
            else getattr(pcfg, "type", "random")
        )

        # ── pass0: original (with optional light random masking) ─────────────
        emb0 = inputs_embeds.clone()
        dsv0 = [deep_stack_viz[di].clone() for di in range(len(deep_stack_viz))]

        # ── pass1 / pass2 ─────────────────────────────────────────────────────
        if perturb_type == "learned" and label is not None and self.training:
            lsparse = float(pcfg.get("lsparse", 0.01) if isinstance(pcfg, dict) else getattr(pcfg, "lsparse", 0.01))
            steps   = int  (pcfg.get("steps",   5)    if isinstance(pcfg, dict) else getattr(pcfg, "steps",   5))
            lr_gate = float(pcfg.get("lr",       0.1)  if isinstance(pcfg, dict) else getattr(pcfg, "lr",       0.1))
            tau     = float(pcfg.get("tau",       1.0)  if isinstance(pcfg, dict) else getattr(pcfg, "tau",       1.0))
            lbl     = label.to(model_device) if torch.is_tensor(label) else label

            req = [p.requires_grad for p in self.parameters()]
            for p in self.parameters():
                p.requires_grad_(False)
            try:
                def _run_logits(ie, dsv):
                    h   = self._encode_from_inputs_embeds(ie, attention_mask, dsv if dsv else None, visual_pos_masks=image_mask if dsv else None)
                    hc  = self._get_cls_token_repr(h, input_ids).to(self.enc_0.linear.weight.dtype)
                    return self.enc_0(hc)

                def _optimize_gate(mask, z_stats, dsv_stats, noise_other, dsv_other, name):
                    n = int(mask.sum().item())
                    if n == 0:
                        return None
                    g   = torch.full((n,), 1.0, device=model_device, dtype=torch.float32, requires_grad=True)
                    opt = torch.optim.Adam([g], lr=lr_gate)
                    for _ in range(steps):
                        gv  = torch.sigmoid(g / tau).clamp(0, 1)
                        ie  = inputs_embeds.clone()
                        ie[mask]  = (ie[mask] * gv.unsqueeze(-1).to(ie.dtype)
                                     + (1 - gv.unsqueeze(-1).to(ie.dtype))
                                     * z_stats.noise_like(ie[mask], 1.0).to(ie.dtype))
                        ie[~mask & attention_mask.bool()] = noise_other
                        dsv_in = [
                            (dsv_stats[di].noise_like(deep_stack_viz[di], 1.0).to(deep_stack_viz[di].dtype)
                             if name == "m2" else dsv_other[di])
                            for di in range(len(deep_stack_viz))
                        ]
                        logits_g = _run_logits(ie, dsv_in)
                        ce       = F.cross_entropy(logits_g, lbl)
                        sp       = (1.0 - torch.sigmoid(g / tau).clamp(0, 1)).mean()
                        (-ce + lsparse * sp).backward()
                        opt.step(); opt.zero_grad(set_to_none=True)
                    return torch.sigmoid(g / tau).detach()

                noise_vis  = self.synib.z2_stats.noise_like(inputs_embeds[vision_mask], 1.0).to(inputs_embeds.dtype) if vision_mask.any() else inputs_embeds[vision_mask]
                noise_txt  = self.synib.z1_stats.noise_like(inputs_embeds[text_mask],   1.0).to(inputs_embeds.dtype) if text_mask.any()   else inputs_embeds[text_mask]
                dsv_noised = [self.synib.z2_deepstack_stats[di].noise_like(deep_stack_viz[di], 1.0).to(deep_stack_viz[di].dtype) for di in range(len(deep_stack_viz))]

                gate1 = _optimize_gate(text_mask,   self.synib.z1_stats, self.synib.z2_deepstack_stats, noise_vis, deep_stack_viz, "m1")
                gate2 = _optimize_gate(vision_mask, self.synib.z2_stats, self.synib.z2_deepstack_stats, noise_txt, deep_stack_viz, "m2")
            finally:
                for p, r in zip(self.parameters(), req):
                    p.requires_grad_(r)

            emb1 = inputs_embeds.clone()
            if gate1 is not None:
                emb1[text_mask]   = (inputs_embeds[text_mask] * gate1.unsqueeze(-1).to(emb1.dtype)
                                     + (1 - gate1.unsqueeze(-1).to(emb1.dtype))
                                     * self.synib.z1_stats.noise_like(inputs_embeds[text_mask], 1.0).to(emb1.dtype))
            if vision_mask.any():
                emb1[vision_mask] = self.synib.z2_stats.noise_like(inputs_embeds[vision_mask], 1.0).to(emb1.dtype)
            dsv1 = [self.synib.z2_deepstack_stats[di].noise_like(deep_stack_viz[di], 1.0).to(deep_stack_viz[di].dtype) for di in range(len(deep_stack_viz))]

            emb2 = inputs_embeds.clone()
            if text_mask.any():
                emb2[text_mask]   = self.synib.z1_stats.noise_like(inputs_embeds[text_mask], 1.0).to(emb2.dtype)
            if gate2 is not None:
                emb2[vision_mask] = (inputs_embeds[vision_mask] * gate2.unsqueeze(-1).to(emb2.dtype)
                                     + (1 - gate2.unsqueeze(-1).to(emb2.dtype))
                                     * self.synib.z2_stats.noise_like(inputs_embeds[vision_mask], 1.0).to(emb2.dtype))
            dsv2 = [
                (deep_stack_viz[di] * gate2.unsqueeze(-1).to(deep_stack_viz[di].dtype)
                 + (1 - gate2.unsqueeze(-1).to(deep_stack_viz[di].dtype))
                 * self.synib.z2_deepstack_stats[di].noise_like(deep_stack_viz[di], 1.0).to(deep_stack_viz[di].dtype))
                if gate2 is not None else deep_stack_viz[di].clone()
                for di in range(len(deep_stack_viz))
            ]
        else:
            # random perturbation (default)
            emb1 = inputs_embeds.clone()
            if vision_mask.any():
                emb1[vision_mask] = self.synib.z2_stats.noise_like(inputs_embeds[vision_mask], 1.0).to(emb1.dtype)
            dsv1 = [self.synib.z2_deepstack_stats[di].noise_like(deep_stack_viz[di], 1.0).to(deep_stack_viz[di].dtype) for di in range(len(deep_stack_viz))]

            emb2 = inputs_embeds.clone()
            if text_mask.any():
                emb2[text_mask] = self.synib.z1_stats.noise_like(inputs_embeds[text_mask], 1.0).to(emb2.dtype)
            dsv2 = [deep_stack_viz[di].clone() for di in range(len(deep_stack_viz))]

        # ── batch 3 passes ────────────────────────────────────────────────────
        attn3 = attention_mask.repeat(3, 1)
        emb3  = torch.cat([emb0, emb1, emb2], dim=0)
        dsv3  = (
            [torch.cat([dsv0[di], dsv1[di], dsv2[di]], dim=0) for di in range(len(dsv0))]
            if dsv0 else None
        )

        mask3    = image_mask.repeat(3, 1) if dsv3 is not None else None
        hidden3  = self._encode_from_inputs_embeds(emb3, attn3, dsv3, visual_pos_masks=mask3)
        ids3     = input_ids.repeat(3, 1)
        h3       = self._get_cls_token_repr(hidden3, ids3).to(self.enc_0.linear.weight.dtype)
        logits3  = self.enc_0(h3)

        logits0, logits1, logits2 = torch.chunk(logits3, 3, dim=0)
        h0,      h1,      h2      = torch.chunk(h3,      3, dim=0)

        losses = {}
        if label is not None:
            if torch.is_tensor(label):
                label = label.to(logits0.device)
            losses["ce_loss_combined"] = self._mc_ce_loss(logits0, label)

        base_output = {
            "preds":    {"combined": logits0, "mask0": logits1, "mask1": logits2},
            "features": {"combined": h0,      "mask0": h1,      "mask1": h2},
        }

        if self.training and self.synergy_weight > 0.0:
            synib_losses = self.synib.compute_training_losses(
                base_output, current_step=current_step
            )
            losses.update(synib_losses)

        features = {"combined": h0}
        if return_features:
            features["hidden"] = hidden3[:B]

        return {"preds": base_output["preds"], "features": features, "losses": losses}
