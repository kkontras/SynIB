import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from .qwen_base_models import LinearHead_Qwen, SynIB_QwenFaster


class _QwenVL_CachedCombinedImpl(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the classifier head.")

        self.args = args
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = int(cfg.text_config.hidden_size)
        else:
            self.d_model = int(cfg.hidden_size)

        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False):
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

    def load_cls_embedding(self, path, strict_dim=True):
        ckpt = torch.load(path, map_location="cpu")
        cls_row = ckpt["cls_row"]

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        current_cls_id = int(self.cls_token_id)

        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}")

        with torch.no_grad():
            emb.weight[current_cls_id].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        if os.path.isfile(cls_path):
            self.load_cls_embedding(cls_path)

    def _apply_lora(self):
        cfg = getattr(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        h = F.layer_norm(h, (h.shape[-1],))
        return h

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(logits, labels, weight=self.args.class_weights.to(logits.device))
        return F.cross_entropy(logits, labels)

    def _build_inputs_embeds_from_cache(
            self,
            input_ids: torch.Tensor,  # (B, T)
            image_mask: torch.Tensor,  # (B, T) bool
            vision_embeds: torch.Tensor,  # (B, N, d) or (N, d)
            *,
            strict: bool = True,  # if True, require N == num_image_positions
    ):
        """
        Build inputs_embeds (B, T, d_model) where positions indicated by image_mask are
        replaced by cached vision_embeds. Does NOT require vision_len.

        If strict=True:
          - requires for each sample: image_mask[b].sum() == vision_embeds[b].shape[0]
        If strict=False:
          - uses min(count_mask, count_embeds) and truncates the longer side.
        """
        lm = self.backbone.model.language_model
        inputs_embeds = lm.embed_tokens(input_ids)  # (B, T, d_model)
        B, T, d_model = inputs_embeds.shape

        for b in range(B):
            pos = image_mask[b].nonzero(as_tuple=False).view(-1)  # indices in [0..T)
            n_mask = int(pos.numel())
            n_vis = int(vision_embeds[b].size(0))

            if (n_mask != n_vis):
                raise ValueError(
                    f"Sample {b}: image_mask has {n_mask} positions but vision_embeds has {n_vis} tokens"
                )
            inputs_embeds[b, pos, :] = vision_embeds[b, :, :].to(inputs_embeds.dtype)

        return inputs_embeds

    @torch.no_grad()
    def generate_answer(
            self,
            proc,  # the same dict you pass as x (processor output)
            max_new_tokens=128,
            min_new_tokens=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
    ):
        self.backbone.eval()

        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)

        # If you used left padding (you did), this is important for many decoders:
        pad_token_id = self.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id

        gen_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=False,
        )

        print("input_ids:", input_ids.shape)
        print("gen_ids:", gen_ids.shape)
        print("new tokens:", gen_ids.shape[1] - input_ids.shape[1])
        prompt_len = input_ids.shape[1]
        new_token_ids = gen_ids[:, prompt_len:]

        texts = self.processor.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return texts

    def _encode_from_inputs_embeds(self, inputs_embeds, attention_mask):
        out = self.backbone.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1]

    def forward(self, x, *, label=None, return_features=False, **kwargs):

        proc = x
        device = self.backbone.device
        tok = self.processor.tokenizer

        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        if "image_mask" in proc:
            image_mask = proc["image_mask"].to(device)
        elif "visual_pos_masks" in proc:
            image_mask = proc["visual_pos_masks"].to(device)
        # vision_embeds = proc["vision_embeds"].to(device)
        input_embeds = proc["input_embeds"].to(device)
        position_ids = proc["position_ids"].to(device)
        deep_stack_viz = proc["deepstack_visual_embeds"].to(device)

        # position_ids = position_ids.permute(1, 0, 2)

        # inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids.to(device))
        # print(vision_embeds.shape)
        # print(inputs_embeds.shape)
        # print(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]).shape)

        # inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]), vision_embeds)
        # position_ids = einops.rearrange(position_ids, "b c i j-> c b (i j)", i=1)
        # deep_stack_viz = einops.rearrange(deep_stack_viz, "b c i j -> c (b i) j")
        deep_stack_viz = [deep_stack_viz[i] for i in range(len(deep_stack_viz))]
        # print(deep_stack_viz.shape)
        # position_ids = position_ids.squeeze(dim=2)

        # inputs_embeds = self._build_inputs_embeds_from_cache(input_ids, image_mask, vision_embeds)


        # print(input_embeds.shape)
        # print(vision_embeds.shape)
        # print(deep_stack_viz.shape)

        def print_lm_input_stats(position_ids, inputs_embeds, attention_mask, image_mask, deep_stack_viz,
                                 name="LM inputs"):
            """
            Short, readable printout of shape + basic stats for each input tensor.
            """
            import torch

            def stats(t):
                if t is None:
                    return "None"
                # handle non-tensors (just in case)
                if not torch.is_tensor(t):
                    return f"{type(t)}"
                tt = t.detach()
                shape = tuple(tt.shape)
                dtype = str(tt.dtype).replace("torch.", "")
                device = str(tt.device)
                numel = tt.numel()

                # min/max/mean only for numeric tensors
                if numel == 0:
                    return f"shape={shape} dtype={dtype} device={device} numel=0"

                # Use float() for stable stats even in fp16/bf16
                if tt.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    x = tt.float()
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item():.5g} max={x.max().item():.5g} "
                            f"mean={x.mean().item():.5g} std={x.std(unbiased=False).item():.5g}")
                elif tt.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                    # for masks / ids: show min/max + %nonzero
                    x = tt
                    nz = (x != 0).float().mean().item() * 100.0
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item()} max={x.max().item()} nonzero={nz:.2f}%")
                else:
                    return f"shape={shape} dtype={dtype} device={device} numel={numel}"

            print(f"\n=== {name} ===")
            print(f"position_ids            : {stats(position_ids)}")
            print(f"inputs_embeds           : {stats(inputs_embeds)}")
            print(f"attention_mask          : {stats(attention_mask)}")
            print(f"visual_pos_masks        : {stats(image_mask)}")
            print(f"deepstack_visual_embeds : {stats(deep_stack_viz)}")

        # call it right before language_model(...)
        # print_lm_input_stats(position_ids, input_embeds, attention_mask, image_mask, deep_stack_viz)


        out = self.backbone.model.language_model(
            input_ids=None,
            position_ids = position_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deep_stack_viz,
            output_hidden_states=True,
            return_dict=True,
            cache_position = None,
            use_cache= False
        )
        hidden = out.hidden_states[-1]


        # hidden = self._encode_from_inputs_embeds(inputs_embeds, attention_mask)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_loss_combined"] = self._mc_ce_loss(logits, label)

        preds = {"combined": logits}
        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        # ============================================================
        # GENERATION (uses cached vision if available)
        # ============================================================
        gen_texts = False
        do_generate = kwargs.get("do_generate", False)  # set True when you want it
        if do_generate:
            # For debugging labels, deterministic decode is usually best
            max_new_tokens = int(kwargs.get("gen_max_new_tokens", 128))
            min_new_tokens = int(kwargs.get("gen_min_new_tokens", 10))
            do_sample = bool(kwargs.get("gen_do_sample", False))
            temperature = float(kwargs.get("gen_temperature", 0.0))
            top_p = float(kwargs.get("gen_top_p", 1.0))

            eos_token_id = tok.eos_token_id
            pad_token_id = self.pad_token_id if hasattr(self, "pad_token_id") else tok.pad_token_id

            with torch.no_grad():
                gen_ids = self.backbone.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

            gen_texts = tok.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i in gen_texts:
                print("---------")
                print(i)
        out = {"preds": preds, "features": features, "losses": losses}
        if gen_texts is not None:
            out["generated_text"] = gen_texts

        return out
class _QwenVL_CachedTextImpl(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the classifier head.")

        self.args = args
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = int(cfg.text_config.hidden_size)
        else:
            self.d_model = int(cfg.hidden_size)

        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False):
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

    def load_cls_embedding(self, path, strict_dim=True):
        ckpt = torch.load(path, map_location="cpu")
        cls_row = ckpt["cls_row"]

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        current_cls_id = int(self.cls_token_id)

        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}")

        with torch.no_grad():
            emb.weight[current_cls_id].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        if os.path.isfile(cls_path):
            self.load_cls_embedding(cls_path)

    def _apply_lora(self):
        cfg = getattr(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        h = F.layer_norm(h, (h.shape[-1],))
        return h

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(logits, labels, weight=self.args.class_weights.to(logits.device))
        return F.cross_entropy(logits, labels)

    def _build_inputs_embeds_from_cache(
            self,
            input_ids: torch.Tensor,  # (B, T)
            image_mask: torch.Tensor,  # (B, T) bool
            vision_embeds: torch.Tensor,  # (B, N, d) or (N, d)
            *,
            strict: bool = True,  # if True, require N == num_image_positions
    ):
        """
        Build inputs_embeds (B, T, d_model) where positions indicated by image_mask are
        replaced by cached vision_embeds. Does NOT require vision_len.

        If strict=True:
          - requires for each sample: image_mask[b].sum() == vision_embeds[b].shape[0]
        If strict=False:
          - uses min(count_mask, count_embeds) and truncates the longer side.
        """
        lm = self.backbone.model.language_model
        inputs_embeds = lm.embed_tokens(input_ids)  # (B, T, d_model)
        B, T, d_model = inputs_embeds.shape

        for b in range(B):
            pos = image_mask[b].nonzero(as_tuple=False).view(-1)  # indices in [0..T)
            n_mask = int(pos.numel())
            n_vis = int(vision_embeds[b].size(0))

            if (n_mask != n_vis):
                raise ValueError(
                    f"Sample {b}: image_mask has {n_mask} positions but vision_embeds has {n_vis} tokens"
                )
            inputs_embeds[b, pos, :] = vision_embeds[b, :, :].to(inputs_embeds.dtype)

        return inputs_embeds

    @torch.no_grad()
    def generate_answer(
            self,
            proc,  # the same dict you pass as x (processor output)
            max_new_tokens=128,
            min_new_tokens=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
    ):
        self.backbone.eval()

        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)

        # If you used left padding (you did), this is important for many decoders:
        pad_token_id = self.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id

        gen_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=False,
        )

        prompt_len = input_ids.shape[1]
        new_token_ids = gen_ids[:, prompt_len:]

        texts = self.processor.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return texts

    def _encode_from_inputs_embeds(self, inputs_embeds, attention_mask):
        out = self.backbone.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1]

    def forward(self, x, *, label=None, return_features=False, **kwargs):

        proc = x
        device = self.backbone.device
        tok = self.processor.tokenizer

        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        if "image_mask" in proc:
            image_mask = proc["image_mask"].to(device)
        elif "visual_pos_masks" in proc:
            image_mask = proc["visual_pos_masks"].to(device)
        # vision_embeds = proc["vision_embeds"].to(device)
        input_embeds = proc["input_embeds"].to(device)
        position_ids = proc["position_ids"].to(device)
        deep_stack_viz = proc["deepstack_visual_embeds"].to(device)

        # position_ids = position_ids.permute(1, 0, 2)

        # inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids.to(device))
        # print(vision_embeds.shape)
        # print(inputs_embeds.shape)
        # print(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]).shape)

        # inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]), vision_embeds)
        # position_ids = einops.rearrange(position_ids, "b c i j-> c b (i j)", i=1)
        # deep_stack_viz = einops.rearrange(deep_stack_viz, "b c i j -> c (b i) j")
        deep_stack_viz = [deep_stack_viz[i] for i in range(len(deep_stack_viz))]
        # print(deep_stack_viz.shape)
        # position_ids = position_ids.squeeze(dim=2)

        # inputs_embeds = self._build_inputs_embeds_from_cache(input_ids, image_mask, vision_embeds)


        # print(input_embeds.shape)
        # print(vision_embeds.shape)
        # print(deep_stack_viz.shape)

        def print_lm_input_stats(position_ids, inputs_embeds, attention_mask, image_mask, deep_stack_viz,
                                 name="LM inputs"):
            """
            Short, readable printout of shape + basic stats for each input tensor.
            """
            import torch

            def stats(t):
                if t is None:
                    return "None"
                # handle non-tensors (just in case)
                if not torch.is_tensor(t):
                    return f"{type(t)}"
                tt = t.detach()
                shape = tuple(tt.shape)
                dtype = str(tt.dtype).replace("torch.", "")
                device = str(tt.device)
                numel = tt.numel()

                # min/max/mean only for numeric tensors
                if numel == 0:
                    return f"shape={shape} dtype={dtype} device={device} numel=0"

                # Use float() for stable stats even in fp16/bf16
                if tt.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    x = tt.float()
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item():.5g} max={x.max().item():.5g} "
                            f"mean={x.mean().item():.5g} std={x.std(unbiased=False).item():.5g}")
                elif tt.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                    # for masks / ids: show min/max + %nonzero
                    x = tt
                    nz = (x != 0).float().mean().item() * 100.0
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item()} max={x.max().item()} nonzero={nz:.2f}%")
                else:
                    return f"shape={shape} dtype={dtype} device={device} numel={numel}"

            print(f"\n=== {name} ===")
            print(f"position_ids            : {stats(position_ids)}")
            print(f"inputs_embeds           : {stats(inputs_embeds)}")
            print(f"attention_mask          : {stats(attention_mask)}")
            print(f"visual_pos_masks        : {stats(image_mask)}")
            print(f"deepstack_visual_embeds : {stats(deep_stack_viz)}")

        # call it right before language_model(...)
        # print_lm_input_stats(position_ids, input_embeds, attention_mask, image_mask, deep_stack_viz)

        mask_to_null = attention_mask.bool() & image_mask.bool()
        mask_3d = mask_to_null.unsqueeze(-1)
        input_embeds = torch.where(mask_3d, torch.tensor(1e-5, device=device, dtype=input_embeds.dtype), input_embeds)
        attention_mask = (attention_mask.bool() & ~image_mask.bool()).to(input_embeds.dtype)

        is_text = (position_ids > 0) & (~image_mask.bool())
        new_pos = torch.cumsum(is_text.long(), dim=-1) - 1
        new_pos = torch.where(is_text, new_pos, torch.zeros_like(new_pos))

        out = self.backbone.model.language_model(
            input_ids=None,
            position_ids = new_pos,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            # visual_pos_masks=image_mask,
            # deepstack_visual_embeds=deep_stack_viz,
            output_hidden_states=True,
            return_dict=True,
            cache_position = None,
            use_cache= False
        )
        hidden = out.hidden_states[-1]


        # hidden = self._encode_from_inputs_embeds(inputs_embeds, attention_mask)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_loss_combined"] = self._mc_ce_loss(logits, label)

        preds = {"combined": logits}
        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        # ============================================================
        # GENERATION (uses cached vision if available)
        # ============================================================
        gen_texts = False
        do_generate = kwargs.get("do_generate", False)  # set True when you want it
        if do_generate:
            # For debugging labels, deterministic decode is usually best
            max_new_tokens = int(kwargs.get("gen_max_new_tokens", 128))
            min_new_tokens = int(kwargs.get("gen_min_new_tokens", 10))
            do_sample = bool(kwargs.get("gen_do_sample", False))
            temperature = float(kwargs.get("gen_temperature", 0.0))
            top_p = float(kwargs.get("gen_top_p", 1.0))

            eos_token_id = tok.eos_token_id
            pad_token_id = self.pad_token_id if hasattr(self, "pad_token_id") else tok.pad_token_id

            with torch.no_grad():
                gen_ids = self.backbone.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

            gen_texts = tok.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i in gen_texts:
                print("---------")
                print(i)
        out = {"preds": preds, "features": features, "losses": losses}
        if gen_texts is not None:
            out["generated_text"] = gen_texts

        return out
class _QwenVL_CachedImageImpl(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the classifier head.")

        self.args = args
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = int(cfg.text_config.hidden_size)
        else:
            self.d_model = int(cfg.hidden_size)

        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False):
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

    def load_cls_embedding(self, path, strict_dim=True):
        ckpt = torch.load(path, map_location="cpu")
        cls_row = ckpt["cls_row"]

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        current_cls_id = int(self.cls_token_id)

        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}")

        with torch.no_grad():
            emb.weight[current_cls_id].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        if os.path.isfile(cls_path):
            self.load_cls_embedding(cls_path)

    def _apply_lora(self):
        cfg = getattr(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        h = F.layer_norm(h, (h.shape[-1],))
        return h

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(logits, labels, weight=self.args.class_weights.to(logits.device))
        return F.cross_entropy(logits, labels)

    def _build_inputs_embeds_from_cache(
            self,
            input_ids: torch.Tensor,  # (B, T)
            image_mask: torch.Tensor,  # (B, T) bool
            vision_embeds: torch.Tensor,  # (B, N, d) or (N, d)
            *,
            strict: bool = True,  # if True, require N == num_image_positions
    ):
        """
        Build inputs_embeds (B, T, d_model) where positions indicated by image_mask are
        replaced by cached vision_embeds. Does NOT require vision_len.

        If strict=True:
          - requires for each sample: image_mask[b].sum() == vision_embeds[b].shape[0]
        If strict=False:
          - uses min(count_mask, count_embeds) and truncates the longer side.
        """
        lm = self.backbone.model.language_model
        inputs_embeds = lm.embed_tokens(input_ids)  # (B, T, d_model)
        B, T, d_model = inputs_embeds.shape

        for b in range(B):
            pos = image_mask[b].nonzero(as_tuple=False).view(-1)  # indices in [0..T)
            n_mask = int(pos.numel())
            n_vis = int(vision_embeds[b].size(0))

            if (n_mask != n_vis):
                raise ValueError(
                    f"Sample {b}: image_mask has {n_mask} positions but vision_embeds has {n_vis} tokens"
                )
            inputs_embeds[b, pos, :] = vision_embeds[b, :, :].to(inputs_embeds.dtype)

        return inputs_embeds

    @torch.no_grad()
    def generate_answer(
            self,
            proc,  # the same dict you pass as x (processor output)
            max_new_tokens=128,
            min_new_tokens=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
    ):
        self.backbone.eval()

        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)

        # If you used left padding (you did), this is important for many decoders:
        pad_token_id = self.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id

        gen_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=False,
        )

        print("input_ids:", input_ids.shape)
        print("gen_ids:", gen_ids.shape)
        print("new tokens:", gen_ids.shape[1] - input_ids.shape[1])
        prompt_len = input_ids.shape[1]
        new_token_ids = gen_ids[:, prompt_len:]

        texts = self.processor.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return texts

    def _encode_from_inputs_embeds(self, inputs_embeds, attention_mask):
        out = self.backbone.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1]

    def forward(self, x, *, label=None, return_features=False, **kwargs):

        proc = x
        device = self.backbone.device
        tok = self.processor.tokenizer

        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        if "image_mask" in proc:
            image_mask = proc["image_mask"].to(device)
        elif "visual_pos_masks" in proc:
            image_mask = proc["visual_pos_masks"].to(device)
        # vision_embeds = proc["vision_embeds"].to(device)
        input_embeds = proc["input_embeds"].to(device)
        position_ids = proc["position_ids"].to(device)
        deep_stack_viz = proc["deepstack_visual_embeds"].to(device)

        # position_ids = position_ids.permute(1, 0, 2)

        # inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids.to(device))
        # print(vision_embeds.shape)
        # print(inputs_embeds.shape)
        # print(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]).shape)

        # inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]), vision_embeds)
        # position_ids = einops.rearrange(position_ids, "b c i j-> c b (i j)", i=1)
        # deep_stack_viz = einops.rearrange(deep_stack_viz, "b c i j -> c (b i) j")
        deep_stack_viz = [deep_stack_viz[i] for i in range(len(deep_stack_viz))]
        # print(deep_stack_viz.shape)
        # position_ids = position_ids.squeeze(dim=2)

        # inputs_embeds = self._build_inputs_embeds_from_cache(input_ids, image_mask, vision_embeds)


        # print(input_embeds.shape)
        # print(vision_embeds.shape)
        # print(deep_stack_viz.shape)

        def print_lm_input_stats(position_ids, inputs_embeds, attention_mask, image_mask, deep_stack_viz,
                                 name="LM inputs"):
            """
            Short, readable printout of shape + basic stats for each input tensor.
            """
            import torch

            def stats(t):
                if t is None:
                    return "None"
                # handle non-tensors (just in case)
                if not torch.is_tensor(t):
                    return f"{type(t)}"
                tt = t.detach()
                shape = tuple(tt.shape)
                dtype = str(tt.dtype).replace("torch.", "")
                device = str(tt.device)
                numel = tt.numel()

                # min/max/mean only for numeric tensors
                if numel == 0:
                    return f"shape={shape} dtype={dtype} device={device} numel=0"

                # Use float() for stable stats even in fp16/bf16
                if tt.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    x = tt.float()
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item():.5g} max={x.max().item():.5g} "
                            f"mean={x.mean().item():.5g} std={x.std(unbiased=False).item():.5g}")
                elif tt.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                    # for masks / ids: show min/max + %nonzero
                    x = tt
                    nz = (x != 0).float().mean().item() * 100.0
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item()} max={x.max().item()} nonzero={nz:.2f}%")
                else:
                    return f"shape={shape} dtype={dtype} device={device} numel={numel}"

            print(f"\n=== {name} ===")
            print(f"position_ids            : {stats(position_ids)}")
            print(f"inputs_embeds           : {stats(inputs_embeds)}")
            print(f"attention_mask          : {stats(attention_mask)}")
            print(f"visual_pos_masks        : {stats(image_mask)}")
            print(f"deepstack_visual_embeds : {stats(deep_stack_viz)}")

        # call it right before language_model(...)
        # print_lm_input_stats(position_ids, input_embeds, attention_mask, image_mask, deep_stack_viz)
        hint_mask = proc["hint_mask"]
        hint_mask = hint_mask.to(device).bool()

        mask_to_null = attention_mask.bool() & hint_mask.bool()
        mask_3d = mask_to_null.unsqueeze(-1)
        input_embeds = torch.where(mask_3d, torch.tensor(1e-5, device=device, dtype=input_embeds.dtype), input_embeds)
        attention_mask = (attention_mask.bool() & ~hint_mask.bool()).to(input_embeds.dtype)

        is_text = (position_ids > 0) & (~hint_mask.bool())
        new_pos = torch.cumsum(is_text.long(), dim=-1) - 1
        new_pos = torch.where(is_text, new_pos, torch.zeros_like(new_pos))

        out = self.backbone.model.language_model(
            input_ids=None,
            position_ids = new_pos,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deep_stack_viz,
            output_hidden_states=True,
            return_dict=True,
            cache_position = None,
            use_cache= False
        )
        hidden = out.hidden_states[-1]


        # hidden = self._encode_from_inputs_embeds(inputs_embeds, attention_mask)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_loss_combined"] = self._mc_ce_loss(logits, label)

        preds = {"combined": logits}
        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        # ============================================================
        # GENERATION (uses cached vision if available)
        # ============================================================
        gen_texts = False
        do_generate = kwargs.get("do_generate", False)  # set True when you want it
        if do_generate:
            # For debugging labels, deterministic decode is usually best
            max_new_tokens = int(kwargs.get("gen_max_new_tokens", 128))
            min_new_tokens = int(kwargs.get("gen_min_new_tokens", 10))
            do_sample = bool(kwargs.get("gen_do_sample", False))
            temperature = float(kwargs.get("gen_temperature", 0.0))
            top_p = float(kwargs.get("gen_top_p", 1.0))

            eos_token_id = tok.eos_token_id
            pad_token_id = self.pad_token_id if hasattr(self, "pad_token_id") else tok.pad_token_id

            with torch.no_grad():
                gen_ids = self.backbone.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

            gen_texts = tok.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i in gen_texts:
                print("---------")
                print(i)
        out = {"preds": preds, "features": features, "losses": losses}
        if gen_texts is not None:
            out["generated_text"] = gen_texts

        return out
class _QwenVL_CachedSynIBImpl(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the classifier head.")

        self.args = args
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = int(cfg.text_config.hidden_size)
        else:
            self.d_model = int(cfg.hidden_size)

        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

        self.synergy_weight = float(self.args.get("bias_infusion", {}).get("l", 0.0))
        self.synib = SynIB_QwenFaster(args, [], self)

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False):
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

    def load_cls_embedding(self, path, strict_dim=True):
        ckpt = torch.load(path, map_location="cpu")
        cls_row = ckpt["cls_row"]

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        current_cls_id = int(self.cls_token_id)

        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}")

        with torch.no_grad():
            emb.weight[current_cls_id].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        if os.path.isfile(cls_path):
            self.load_cls_embedding(cls_path)

    def _apply_lora(self):
        cfg = getattr(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        h = F.layer_norm(h, (h.shape[-1],))
        return h

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(logits, labels, weight=self.args.class_weights.to(logits.device))
        return F.cross_entropy(logits, labels)

    def _build_inputs_embeds_from_cache( self, input_ids: torch.Tensor, image_mask: torch.Tensor, vision_embeds: torch.Tensor,  *, strict: bool = True):
        """
        Build inputs_embeds (B, T, d_model) where positions indicated by image_mask are
        replaced by cached vision_embeds. Does NOT require vision_len.

        If strict=True:
          - requires for each sample: image_mask[b].sum() == vision_embeds[b].shape[0]
        If strict=False:
          - uses min(count_mask, count_embeds) and truncates the longer side.
        """
        lm = self.backbone.model.language_model
        inputs_embeds = lm.embed_tokens(input_ids)  # (B, T, d_model)
        B, T, d_model = inputs_embeds.shape

        for b in range(B):
            pos = image_mask[b].nonzero(as_tuple=False).view(-1)  # indices in [0..T)
            n_mask = int(pos.numel())
            n_vis = int(vision_embeds[b].size(0))

            if (n_mask != n_vis):
                raise ValueError(
                    f"Sample {b}: image_mask has {n_mask} positions but vision_embeds has {n_vis} tokens"
                )
            inputs_embeds[b, pos, :] = vision_embeds[b, :, :].to(inputs_embeds.dtype)

        return inputs_embeds

    @torch.no_grad()
    def generate_answer(
            self,
            proc,  # the same dict you pass as x (processor output)
            max_new_tokens=128,
            min_new_tokens=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
    ):
        self.backbone.eval()

        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)

        # If you used left padding (you did), this is important for many decoders:
        pad_token_id = self.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id

        gen_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=False,
        )

        print("input_ids:", input_ids.shape)
        print("gen_ids:", gen_ids.shape)
        print("new tokens:", gen_ids.shape[1] - input_ids.shape[1])
        prompt_len = input_ids.shape[1]
        new_token_ids = gen_ids[:, prompt_len:]

        texts = self.processor.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return texts

    def _encode_from_inputs_embeds(self,  position_ids, input_embeds, image_mask, deep_stack_viz, attention_mask):
        out = self.backbone.model.language_model(
            input_ids=None,
            position_ids = position_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deep_stack_viz,
            output_hidden_states=True,
            return_dict=True,
            cache_position = None,
            use_cache= False
        )
        hidden = out.hidden_states[-1]
        return hidden

    def _compute_logits_from_proc(self, x, *, label=None, return_features=False, **kwargs):

        proc = x
        device = self.backbone.device
        tok = self.processor.tokenizer

        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        if "image_mask" in proc:
            image_mask = proc["image_mask"].to(device)
        elif "visual_pos_masks" in proc:
            image_mask = proc["visual_pos_masks"].to(device)
        # vision_embeds = proc["vision_embeds"].to(device)
        input_embeds = proc["input_embeds"].to(device)
        position_ids = proc["position_ids"].to(device)
        deep_stack_viz = proc["deepstack_visual_embeds"][0].to(device)

        # position_ids = position_ids.permute(1, 0, 2)

        # inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids.to(device))
        # print(vision_embeds.shape)
        # print(inputs_embeds.shape)
        # print(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]).shape)

        # inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]), vision_embeds)
        # position_ids = einops.rearrange(position_ids, "b c i j-> c b (i j)", i=1)
        # deep_stack_viz = einops.rearrange(deep_stack_viz, "b c i j -> c (b i) j")
        deep_stack_viz = [deep_stack_viz[i] for i in range(len(deep_stack_viz))]
        # print(deep_stack_viz.shape)
        # position_ids = position_ids.squeeze(dim=2)

        # inputs_embeds = self._build_inputs_embeds_from_cache(input_ids, image_mask, vision_embeds)


        # print(input_embeds.shape)
        # print(vision_embeds.shape)
        # print(deep_stack_viz.shape)

        def print_lm_input_stats(position_ids, inputs_embeds, attention_mask, image_mask, deep_stack_viz,
                                 name="LM inputs"):
            """
            Short, readable printout of shape + basic stats for each input tensor.
            """
            import torch

            def stats(t):
                if t is None:
                    return "None"
                # handle non-tensors (just in case)
                if not torch.is_tensor(t):
                    return f"{type(t)}"
                tt = t.detach()
                shape = tuple(tt.shape)
                dtype = str(tt.dtype).replace("torch.", "")
                device = str(tt.device)
                numel = tt.numel()

                # min/max/mean only for numeric tensors
                if numel == 0:
                    return f"shape={shape} dtype={dtype} device={device} numel=0"

                # Use float() for stable stats even in fp16/bf16
                if tt.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    x = tt.float()
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item():.5g} max={x.max().item():.5g} "
                            f"mean={x.mean().item():.5g} std={x.std(unbiased=False).item():.5g}")
                elif tt.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                    # for masks / ids: show min/max + %nonzero
                    x = tt
                    nz = (x != 0).float().mean().item() * 100.0
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item()} max={x.max().item()} nonzero={nz:.2f}%")
                else:
                    return f"shape={shape} dtype={dtype} device={device} numel={numel}"

            print(f"\n=== {name} ===")
            print(f"position_ids            : {stats(position_ids)}")
            print(f"inputs_embeds           : {stats(inputs_embeds)}")
            print(f"attention_mask          : {stats(attention_mask)}")
            print(f"visual_pos_masks        : {stats(image_mask)}")
            print(f"deepstack_visual_embeds : {stats(deep_stack_viz)}")

        # call it right before language_model(...)
        # print_lm_input_stats(position_ids, input_embeds, attention_mask, image_mask, deep_stack_viz)


        out = self.backbone.model.language_model(
            input_ids=None,
            position_ids = position_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deep_stack_viz,
            output_hidden_states=True,
            return_dict=True,
            cache_position = None,
            use_cache= False
        )
        hidden = out.hidden_states[-1]

        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_loss_combined"] = self._mc_ce_loss(logits, label)

        preds = {"combined": logits}
        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        # ============================================================
        # GENERATION (uses cached vision if available)
        # ============================================================
        gen_texts = False
        do_generate = kwargs.get("do_generate", False)  # set True when you want it
        if do_generate:
            # For debugging labels, deterministic decode is usually best
            max_new_tokens = int(kwargs.get("gen_max_new_tokens", 128))
            min_new_tokens = int(kwargs.get("gen_min_new_tokens", 10))
            do_sample = bool(kwargs.get("gen_do_sample", False))
            temperature = float(kwargs.get("gen_temperature", 0.0))
            top_p = float(kwargs.get("gen_top_p", 1.0))

            eos_token_id = tok.eos_token_id
            pad_token_id = self.pad_token_id if hasattr(self, "pad_token_id") else tok.pad_token_id

            with torch.no_grad():
                gen_ids = self.backbone.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

            gen_texts = tok.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i in gen_texts:
                print("---------")
                print(i)
        out = {"preds": preds, "features": features, "losses": losses}
        if gen_texts is not None:
            out["generated_text"] = gen_texts

        return out

    def apply_custom_masks(self, base_att_mask, m1, m2, m1_t, m2_t):
        combined_hint = base_att_mask.clone()
        combined_hint[m1.bool()] = m1_t[m1.bool()].long()
        combined_img = base_att_mask.clone()
        combined_img[m2.bool()] = m2_t[m2.bool()].long()
        return combined_hint, combined_img

    def _compute_logits_synib_from_proc(self, x, **kwargs):

        proc = x
        device = self.backbone.device
        tok = self.processor.tokenizer

        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        if "image_mask" in proc:
            image_mask = proc["image_mask"].to(device)
        elif "visual_pos_masks" in proc:
            image_mask = proc["visual_pos_masks"].to(device)
        # vision_embeds = proc["vision_embeds"].to(device)
        input_embeds = proc["input_embeds"].to(device)
        position_ids = proc["position_ids"].to(device)
        deep_stack_viz = proc["deepstack_visual_embeds"].to(device)

        # position_ids = position_ids.permute(1, 0, 2)

        # inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids.to(device))
        # print(vision_embeds.shape)
        # print(inputs_embeds.shape)
        # print(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]).shape)

        # inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]), vision_embeds)
        # position_ids = einops.rearrange(position_ids, "b c i j-> c b (i j)", i=1)
        # deep_stack_viz = einops.rearrange(deep_stack_viz, "b c i j -> c (b i) j")
        deep_stack_viz = [deep_stack_viz[i] for i in range(len(deep_stack_viz))]
        # print(deep_stack_viz.shape)
        # position_ids = position_ids.squeeze(dim=2)

        # inputs_embeds = self._build_inputs_embeds_from_cache(input_ids, image_mask, vision_embeds)


        # print(input_embeds.shape)
        # print(vision_embeds.shape)
        # print(deep_stack_viz.shape)

        def print_lm_input_stats(position_ids, inputs_embeds, attention_mask, image_mask, deep_stack_viz,
                                 name="LM inputs"):
            """
            Short, readable printout of shape + basic stats for each input tensor.
            """
            import torch

            def stats(t):
                if t is None:
                    return "None"
                # handle non-tensors (just in case)
                if not torch.is_tensor(t):
                    return f"{type(t)}"
                tt = t.detach()
                shape = tuple(tt.shape)
                dtype = str(tt.dtype).replace("torch.", "")
                device = str(tt.device)
                numel = tt.numel()

                # min/max/mean only for numeric tensors
                if numel == 0:
                    return f"shape={shape} dtype={dtype} device={device} numel=0"

                # Use float() for stable stats even in fp16/bf16
                if tt.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    x = tt.float()
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item():.5g} max={x.max().item():.5g} "
                            f"mean={x.mean().item():.5g} std={x.std(unbiased=False).item():.5g}")
                elif tt.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                    # for masks / ids: show min/max + %nonzero
                    x = tt
                    nz = (x != 0).float().mean().item() * 100.0
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item()} max={x.max().item()} nonzero={nz:.2f}%")
                else:
                    return f"shape={shape} dtype={dtype} device={device} numel={numel}"

            print(f"\n=== {name} ===")
            print(f"position_ids            : {stats(position_ids)}")
            print(f"inputs_embeds           : {stats(inputs_embeds)}")
            print(f"attention_mask          : {stats(attention_mask)}")
            print(f"visual_pos_masks        : {stats(image_mask)}")
            print(f"deepstack_visual_embeds : {stats(deep_stack_viz)}")

        # call it right before language_model(...)
        # print_lm_input_stats(position_ids, input_embeds, attention_mask, image_mask, deep_stack_viz)

        m1 = proc.get("hint_mask", None)
        m2 = proc.get("visual_pos_masks", None)
        if m1 is None or m2 is None:
            raise KeyError("Need proc['hint_mask'] and proc['image_mask'] for SynIB cached mode.")

        m1 = m1.to(input_ids.device).bool()
        m2 = m2.to(input_ids.device).bool()

        self.synib.z1_stats.ema_update(proc["input_embeds"][m1])
        self.synib.z2_stats.ema_update(proc["input_embeds"][m2])
        for i in range(len(proc["deepstack_visual_embeds"])):
            self.synib.z2_deepstack_stats[i].ema_update(proc["deepstack_visual_embeds"][i])

        if self.args.get("perturb", {}).get("type", "rand") == "rand":
            m1t, m2t = self.synib._random_masks(m1, m2, True, True, **kwargs)
        elif self.args.get("perturb", {}).get("type", "rand") == "learned":
            m1t, m2t = self.synib._learned_masks(m1, m2, True, True, proc={"input_ids":input_ids, "position_ids":position_ids, "input_embeds":input_embeds, "image_mask":image_mask, "deep_stack_viz":deep_stack_viz, "attention_mask":attention_mask}, **kwargs)
            m1t = 1-m1t.float()
            m2t = 1-m2t.float()
        else:
            raise ValueError(f"Unknown perturb.type: {self.args.get('perturb', {})}")

        m1forw, m2forw = self.synib._random_masks_randomp(m1, m2, True, True)

        if getattr(self.args, "run_multiple_forwards", False):
            masks = torch.stack([attention_mask, attention_mask, attention_mask], dim=0)
            image_masks = [image_mask, image_mask, m2t]
            filter_deep_stack = [image_mask[image_mask], image_mask[image_mask], m2t[image_mask]]
            hidden_all = torch.cat([self._encode_from_inputs_embeds(position_ids, input_embeds, image_masks[i], [deep_stack_viz[j][filter_deep_stack[i]] for j in range(len(deep_stack_viz))], masks[i]) for i in range(3)],dim=0)
        else:
            masks = torch.cat([attention_mask, attention_mask, attention_mask], dim=0)
            k=3
            position_ids_expanded = position_ids.repeat(1, k, 1)

            this_embed_0 = input_embeds.clone()
            if (m1!=m1forw).any():
                m1forw = m1forw.to(this_embed_0.dtype)
                this_embed_0[m1] = this_embed_0[m1] * m1forw[m1].unsqueeze(dim=1) + (
                            1 - m1forw[m1].unsqueeze(dim=1)) * self.synib.z1_stats.noise_like(this_embed_0[m1], 1.0).to(this_embed_0.dtype)
            if (m2!=m2forw).any():
                m2forw = m2forw.to(this_embed_0.dtype)
                this_embed_0[m2] = this_embed_0[m2] * m2forw[m2].unsqueeze(dim=1) + (
                            1 - m2forw[m2].unsqueeze(dim=1)) * self.synib.z2_stats.noise_like(this_embed_0[m2], 1.0).to(this_embed_0.dtype)

            this_embed_1 = input_embeds.clone()
            m1t = m1t.to(this_embed_1.dtype)
            this_embed_1[m1]=this_embed_1[m1]*m1t.unsqueeze(dim=1) + (1-m1t.unsqueeze(dim=1))*self.synib.z1_stats.noise_like(this_embed_1[m1], 1.0).to(this_embed_1.dtype)
            this_embed_2 = input_embeds.clone()
            m2t = m2t.to(this_embed_2.dtype)
            this_embed_2[m2]= this_embed_2[m2]* m2t.unsqueeze(dim=1) + (1-m2t.unsqueeze(dim=1))*self.synib.z2_stats.noise_like(this_embed_2[m2], 1.0).to(this_embed_2.dtype)
            input_embeds_expanded = torch.cat([this_embed_0, this_embed_1, this_embed_2], dim=0)
            filter_deep_stack = torch.cat([image_mask, image_mask, image_mask],dim=0)

            deep_stack_viz_extended=[]
            for i in range(len(deep_stack_viz)):
                this_embed_2 = deep_stack_viz[i].clone()
                this_embed_2 = this_embed_2*m2t.unsqueeze(dim=1)  + (1-m2t.unsqueeze(dim=1))*self.synib.z2_deepstack_stats[i].noise_like(this_embed_2, 1.0)
                deep_stack_viz_extended.append(torch.cat([deep_stack_viz[i], deep_stack_viz[i], this_embed_2], dim=0))

            hidden_all = self._encode_from_inputs_embeds(position_ids_expanded, input_embeds_expanded, filter_deep_stack, deep_stack_viz_extended, masks)

        ids_all = input_ids.repeat(k,1)
        h_cls_all = self._get_cls_token_repr(hidden_all, ids_all)
        logits_all = self.enc_0(h_cls_all)

        head_logits, head_logits_0, head_logits_1 = torch.chunk(logits_all, chunks=3, dim=0)
        h_cls, featcls_0, featcls_1 = torch.chunk(h_cls_all, chunks=3, dim=0)

        losses = {}
        if "label" in kwargs and kwargs["label"] is not None:
            losses["ce_loss_combined"] = self._mc_ce_loss(head_logits, kwargs["label"])

        preds = {"combined": head_logits, "mask0": head_logits_0, "mask1": head_logits_1}
        features = {"combined": h_cls, "mask0": featcls_0, "mask1": featcls_1}

        # ============================================================
        # GENERATION (uses cached vision if available)
        # ============================================================
        gen_texts = False
        do_generate = kwargs.get("do_generate", False)  # set True when you want it
        if do_generate:
            # For debugging labels, deterministic decode is usually best
            max_new_tokens = int(kwargs.get("gen_max_new_tokens", 128))
            min_new_tokens = int(kwargs.get("gen_min_new_tokens", 10))
            do_sample = bool(kwargs.get("gen_do_sample", False))
            temperature = float(kwargs.get("gen_temperature", 0.0))
            top_p = float(kwargs.get("gen_top_p", 1.0))

            eos_token_id = tok.eos_token_id
            pad_token_id = self.pad_token_id if hasattr(self, "pad_token_id") else tok.pad_token_id

            with torch.no_grad():
                gen_ids = self.backbone.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

            gen_texts = tok.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i in gen_texts:
                print("---------")
                print(i)
        out = {"preds": preds, "features": features, "losses": losses}
        if gen_texts is not None:
            out["generated_text"] = gen_texts

        return out

    def forward(self, x, **kwargs):
        if self.training:
            out = self._compute_logits_synib_from_proc(x, **kwargs)
        else:
            out = self._compute_logits_from_proc(x, **kwargs)

        if self.training and self.synergy_weight > 0:
            synergy_losses = self.synib.compute_training_losses(out, **kwargs)
            out["losses"].update(synergy_losses)
        return out
class _QwenVL_CachedMCRImpl(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the classifier head.")

        self.args = args
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = int(cfg.text_config.hidden_size)
        else:
            self.d_model = int(cfg.hidden_size)

        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False):
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

    def load_cls_embedding(self, path, strict_dim=True):
        ckpt = torch.load(path, map_location="cpu")
        cls_row = ckpt["cls_row"]

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        current_cls_id = int(self.cls_token_id)

        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}")

        with torch.no_grad():
            emb.weight[current_cls_id].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        if os.path.isfile(cls_path):
            self.load_cls_embedding(cls_path)

    def _apply_lora(self):
        cfg = getattr(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        h = F.layer_norm(h, (h.shape[-1],))
        return h

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(logits, labels, weight=self.args.class_weights.to(logits.device))
        return F.cross_entropy(logits, labels)

    def _build_inputs_embeds_from_cache(
            self,
            input_ids: torch.Tensor,  # (B, T)
            image_mask: torch.Tensor,  # (B, T) bool
            vision_embeds: torch.Tensor,  # (B, N, d) or (N, d)
            *,
            strict: bool = True,  # if True, require N == num_image_positions
    ):
        """
        Build inputs_embeds (B, T, d_model) where positions indicated by image_mask are
        replaced by cached vision_embeds. Does NOT require vision_len.

        If strict=True:
          - requires for each sample: image_mask[b].sum() == vision_embeds[b].shape[0]
        If strict=False:
          - uses min(count_mask, count_embeds) and truncates the longer side.
        """
        lm = self.backbone.model.language_model
        inputs_embeds = lm.embed_tokens(input_ids)  # (B, T, d_model)
        B, T, d_model = inputs_embeds.shape

        for b in range(B):
            pos = image_mask[b].nonzero(as_tuple=False).view(-1)  # indices in [0..T)
            n_mask = int(pos.numel())
            n_vis = int(vision_embeds[b].size(0))

            if (n_mask != n_vis):
                raise ValueError(
                    f"Sample {b}: image_mask has {n_mask} positions but vision_embeds has {n_vis} tokens"
                )
            inputs_embeds[b, pos, :] = vision_embeds[b, :, :].to(inputs_embeds.dtype)

        return inputs_embeds

    @torch.no_grad()
    def generate_answer(
            self,
            proc,  # the same dict you pass as x (processor output)
            max_new_tokens=128,
            min_new_tokens=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
    ):
        self.backbone.eval()

        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)

        # If you used left padding (you did), this is important for many decoders:
        pad_token_id = self.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id

        gen_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=False,
        )

        print("input_ids:", input_ids.shape)
        print("gen_ids:", gen_ids.shape)
        print("new tokens:", gen_ids.shape[1] - input_ids.shape[1])
        prompt_len = input_ids.shape[1]
        new_token_ids = gen_ids[:, prompt_len:]

        texts = self.processor.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return texts

    def _encode_from_inputs_embeds(self, inputs_embeds, attention_mask):
        out = self.backbone.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1]

    def forward(self, x, *, label=None, return_features=False, **kwargs):

        proc = x
        device = self.backbone.device
        tok = self.processor.tokenizer

        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        if "image_mask" in proc:
            image_mask = proc["image_mask"].to(device)
        elif "visual_pos_masks" in proc:
            image_mask = proc["visual_pos_masks"].to(device)
        # vision_embeds = proc["vision_embeds"].to(device)
        input_embeds = proc["input_embeds"].to(device)
        position_ids = proc["position_ids"].to(device)
        deep_stack_viz = proc["deepstack_visual_embeds"].to(device)

        # position_ids = position_ids.permute(1, 0, 2)

        # inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids.to(device))
        # print(vision_embeds.shape)
        # print(inputs_embeds.shape)
        # print(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]).shape)

        # inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]), vision_embeds)
        # position_ids = einops.rearrange(position_ids, "b c i j-> c b (i j)", i=1)
        # deep_stack_viz = einops.rearrange(deep_stack_viz, "b c i j -> c (b i) j")
        deep_stack_viz = [deep_stack_viz[i] for i in range(len(deep_stack_viz))]
        # print(deep_stack_viz.shape)
        # position_ids = position_ids.squeeze(dim=2)

        # inputs_embeds = self._build_inputs_embeds_from_cache(input_ids, image_mask, vision_embeds)


        # print(input_embeds.shape)
        # print(vision_embeds.shape)
        # print(deep_stack_viz.shape)

        def print_lm_input_stats(position_ids, inputs_embeds, attention_mask, image_mask, deep_stack_viz,
                                 name="LM inputs"):
            """
            Short, readable printout of shape + basic stats for each input tensor.
            """
            import torch

            def stats(t):
                if t is None:
                    return "None"
                # handle non-tensors (just in case)
                if not torch.is_tensor(t):
                    return f"{type(t)}"
                tt = t.detach()
                shape = tuple(tt.shape)
                dtype = str(tt.dtype).replace("torch.", "")
                device = str(tt.device)
                numel = tt.numel()

                # min/max/mean only for numeric tensors
                if numel == 0:
                    return f"shape={shape} dtype={dtype} device={device} numel=0"

                # Use float() for stable stats even in fp16/bf16
                if tt.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    x = tt.float()
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item():.5g} max={x.max().item():.5g} "
                            f"mean={x.mean().item():.5g} std={x.std(unbiased=False).item():.5g}")
                elif tt.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                    # for masks / ids: show min/max + %nonzero
                    x = tt
                    nz = (x != 0).float().mean().item() * 100.0
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item()} max={x.max().item()} nonzero={nz:.2f}%")
                else:
                    return f"shape={shape} dtype={dtype} device={device} numel={numel}"

            print(f"\n=== {name} ===")
            print(f"position_ids            : {stats(position_ids)}")
            print(f"inputs_embeds           : {stats(inputs_embeds)}")
            print(f"attention_mask          : {stats(attention_mask)}")
            print(f"visual_pos_masks        : {stats(image_mask)}")
            print(f"deepstack_visual_embeds : {stats(deep_stack_viz)}")

        # call it right before language_model(...)
        # print_lm_input_stats(position_ids, input_embeds, attention_mask, image_mask, deep_stack_viz)


        out = self.backbone.model.language_model(
            input_ids=None,
            position_ids = position_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deep_stack_viz,
            output_hidden_states=True,
            return_dict=True,
            cache_position = None,
            use_cache= False
        )
        hidden = out.hidden_states[-1]


        # hidden = self._encode_from_inputs_embeds(inputs_embeds, attention_mask)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)

        losses = {}
        if label is not None:
            losses["ce_loss_combined"] = self._mc_ce_loss(logits, label)

        preds = {"combined": logits}
        features = {"combined": h_cls}
        if return_features:
            features["hidden"] = hidden

        # ============================================================
        # GENERATION (uses cached vision if available)
        # ============================================================
        gen_texts = False
        do_generate = kwargs.get("do_generate", False)  # set True when you want it
        if do_generate:
            # For debugging labels, deterministic decode is usually best
            max_new_tokens = int(kwargs.get("gen_max_new_tokens", 128))
            min_new_tokens = int(kwargs.get("gen_min_new_tokens", 10))
            do_sample = bool(kwargs.get("gen_do_sample", False))
            temperature = float(kwargs.get("gen_temperature", 0.0))
            top_p = float(kwargs.get("gen_top_p", 1.0))

            eos_token_id = tok.eos_token_id
            pad_token_id = self.pad_token_id if hasattr(self, "pad_token_id") else tok.pad_token_id

            with torch.no_grad():
                gen_ids = self.backbone.generate(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature if do_sample else None,
                    top_p=top_p if do_sample else None,
                    eos_token_id=eos_token_id,
                    pad_token_id=pad_token_id,
                )

            gen_texts = tok.batch_decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for i in gen_texts:
                print("---------")
                print(i)
        out = {"preds": preds, "features": features, "losses": losses}
        if gen_texts is not None:
            out["generated_text"] = gen_texts

        return out
class _QwenVL_CachedMMParetoImpl(nn.Module):
    def __init__(self, args, encs=None, **kwargs):
        super().__init__()
        encs = encs or []
        if len(encs) < 1:
            raise ValueError("encs[0] must be provided as the classifier head.")

        self.args = args
        self.num_classes = getattr(args, "num_classes")

        model_name = getattr(args, "model_name", "Qwen/Qwen3-VL-2B-Instruct")
        hf_cache = getattr(self.args, "save_base_dir", None)

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)
        tok = self.processor.tokenizer
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        self.pad_token_id = tok.pad_token_id

        added = tok.add_special_tokens({"additional_special_tokens": ["<CLS>"]})
        self.cls_token_id = tok.convert_tokens_to_ids("<CLS>")

        self.backbone = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            cache_dir=hf_cache,
        )
        if added > 0:
            self.backbone.resize_token_embeddings(len(tok))

        cfg = self.backbone.config
        self.image_token_id = int(cfg.image_token_id)

        if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
            self.d_model = int(cfg.text_config.hidden_size)
        else:
            self.d_model = int(cfg.hidden_size)

        self.enc_0 = encs[0]

        self._apply_lora()
        self._load_cls_embedding()
        self._setup_trainables()

    def _setup_trainables(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

        if getattr(self.args, "lora_config", None) and self.args.lora_config.get("use_lora", False):
            for n, p in self.backbone.named_parameters():
                if "lora_" in n:
                    p.requires_grad = True

        for p in self.enc_0.parameters():
            p.requires_grad = True

        lm = self.backbone.model.language_model
        if getattr(self.args, "cls_finetune", False):
            if getattr(self.args, "train_cls_row", True) and lm is not None and hasattr(lm, "embed_tokens"):
                emb = lm.embed_tokens
                emb.weight.requires_grad = True

                cls_id = int(self.cls_token_id)
                mask = torch.zeros_like(emb.weight, dtype=torch.float32)
                mask[cls_id].fill_(1.0)

                def grad_mask_hook(grad):
                    return grad * mask.to(grad.device, grad.dtype)

                if not hasattr(self, "_cls_grad_hooked"):
                    emb.weight.register_hook(grad_mask_hook)
                    self._cls_grad_hooked = True

    def load_cls_embedding(self, path, strict_dim=True):
        ckpt = torch.load(path, map_location="cpu")
        cls_row = ckpt["cls_row"]

        lm = self.backbone.model.language_model
        if lm is None or not hasattr(lm, "embed_tokens"):
            raise RuntimeError("Language model embedding table not found")

        emb = lm.embed_tokens
        current_cls_id = int(self.cls_token_id)

        if strict_dim and cls_row.numel() != emb.weight.shape[1]:
            raise ValueError(f"CLS dim mismatch: saved {cls_row.numel()} vs model {emb.weight.shape[1]}")

        with torch.no_grad():
            emb.weight[current_cls_id].copy_(cls_row.to(emb.weight.device, emb.weight.dtype))

    def _load_cls_embedding(self):
        cls_path = getattr(self.args, "cls_emb_path", None)
        save_base_dir = getattr(self.args, "save_base_dir", None)
        if save_base_dir is None or cls_path is None:
            return
        cls_path = os.path.join(save_base_dir, cls_path)
        if os.path.isfile(cls_path):
            self.load_cls_embedding(cls_path)

    def _apply_lora(self):
        cfg = getattr(self.args, "lora_config", None)
        if not cfg or not cfg.get("use_lora", False):
            return

        lora_cfg = LoraConfig(
            r=int(cfg.get("lora_r", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 8)),
            lora_dropout=float(cfg.get("lora_dropout", 0.0)),
            target_modules=list(cfg.get("lora_target_modules", ["q_proj", "v_proj"])),
            bias=str(cfg.get("lora_bias", "none")),
            task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(self.backbone, lora_cfg)

    def _get_cls_token_repr(self, hidden, input_ids):
        B = input_ids.size(0)
        cls_pos = (input_ids == self.cls_token_id).int().argmax(dim=1)
        h = hidden[torch.arange(B, device=input_ids.device), cls_pos]
        h = F.layer_norm(h, (h.shape[-1],))
        return h

    def _mc_ce_loss(self, logits, labels):
        if hasattr(self.args, "class_weights") and self.args.class_weights is not None:
            return F.cross_entropy(logits, labels, weight=self.args.class_weights.to(logits.device))
        return F.cross_entropy(logits, labels)

    def _build_inputs_embeds_from_cache(
            self,
            input_ids: torch.Tensor,  # (B, T)
            image_mask: torch.Tensor,  # (B, T) bool
            vision_embeds: torch.Tensor,  # (B, N, d) or (N, d)
            *,
            strict: bool = True,  # if True, require N == num_image_positions
    ):
        """
        Build inputs_embeds (B, T, d_model) where positions indicated by image_mask are
        replaced by cached vision_embeds. Does NOT require vision_len.

        If strict=True:
          - requires for each sample: image_mask[b].sum() == vision_embeds[b].shape[0]
        If strict=False:
          - uses min(count_mask, count_embeds) and truncates the longer side.
        """
        lm = self.backbone.model.language_model
        inputs_embeds = lm.embed_tokens(input_ids)  # (B, T, d_model)
        B, T, d_model = inputs_embeds.shape

        for b in range(B):
            pos = image_mask[b].nonzero(as_tuple=False).view(-1)  # indices in [0..T)
            n_mask = int(pos.numel())
            n_vis = int(vision_embeds[b].size(0))

            if (n_mask != n_vis):
                raise ValueError(
                    f"Sample {b}: image_mask has {n_mask} positions but vision_embeds has {n_vis} tokens"
                )
            inputs_embeds[b, pos, :] = vision_embeds[b, :, :].to(inputs_embeds.dtype)

        return inputs_embeds

    @torch.no_grad()
    def generate_answer(
            self,
            proc,  # the same dict you pass as x (processor output)
            max_new_tokens=128,
            min_new_tokens=2,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
    ):
        self.backbone.eval()

        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)

        # If you used left padding (you did), this is important for many decoders:
        pad_token_id = self.pad_token_id
        eos_token_id = self.processor.tokenizer.eos_token_id

        gen_ids = self.backbone.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            return_dict_in_generate=False,
        )

        print("input_ids:", input_ids.shape)
        print("gen_ids:", gen_ids.shape)
        print("new tokens:", gen_ids.shape[1] - input_ids.shape[1])
        prompt_len = input_ids.shape[1]
        new_token_ids = gen_ids[:, prompt_len:]

        texts = self.processor.tokenizer.batch_decode(
            new_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return texts

    def _encode_from_inputs_embeds(self, inputs_embeds, attention_mask):
        out = self.backbone.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return out.hidden_states[-1]

    def forward(self, x, *, label=None, return_features=False, **kwargs):

        proc = x
        device = self.backbone.device
        tok = self.processor.tokenizer

        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        if "image_mask" in proc:
            image_mask = proc["image_mask"].to(device)
        elif "visual_pos_masks" in proc:
            image_mask = proc["visual_pos_masks"].to(device)
        # vision_embeds = proc["vision_embeds"].to(device)
        input_embeds = proc["input_embeds"].to(device)
        position_ids = proc["position_ids"].to(device)
        deep_stack_viz = proc["deepstack_visual_embeds"].to(device)

        # position_ids = position_ids.permute(1, 0, 2)

        # inputs_embeds = self.backbone.model.get_input_embeddings()(input_ids.to(device))
        # print(vision_embeds.shape)
        # print(inputs_embeds.shape)
        # print(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]).shape)

        # inputs_embeds = inputs_embeds.masked_scatter(image_mask.unsqueeze(dim=-1).repeat(1,1,vision_embeds.shape[-1]), vision_embeds)
        # position_ids = einops.rearrange(position_ids, "b c i j-> c b (i j)", i=1)
        # deep_stack_viz = einops.rearrange(deep_stack_viz, "b c i j -> c (b i) j")
        deep_stack_viz = [deep_stack_viz[i] for i in range(len(deep_stack_viz))]
        # print(deep_stack_viz.shape)
        # position_ids = position_ids.squeeze(dim=2)

        # inputs_embeds = self._build_inputs_embeds_from_cache(input_ids, image_mask, vision_embeds)


        # print(input_embeds.shape)
        # print(vision_embeds.shape)
        # print(deep_stack_viz.shape)

        def print_lm_input_stats(position_ids, inputs_embeds, attention_mask, image_mask, deep_stack_viz,
                                 name="LM inputs"):
            """
            Short, readable printout of shape + basic stats for each input tensor.
            """
            import torch

            def stats(t):
                if t is None:
                    return "None"
                # handle non-tensors (just in case)
                if not torch.is_tensor(t):
                    return f"{type(t)}"
                tt = t.detach()
                shape = tuple(tt.shape)
                dtype = str(tt.dtype).replace("torch.", "")
                device = str(tt.device)
                numel = tt.numel()

                # min/max/mean only for numeric tensors
                if numel == 0:
                    return f"shape={shape} dtype={dtype} device={device} numel=0"

                # Use float() for stable stats even in fp16/bf16
                if tt.dtype in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
                    x = tt.float()
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item():.5g} max={x.max().item():.5g} "
                            f"mean={x.mean().item():.5g} std={x.std(unbiased=False).item():.5g}")
                elif tt.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool):
                    # for masks / ids: show min/max + %nonzero
                    x = tt
                    nz = (x != 0).float().mean().item() * 100.0
                    return (f"shape={shape} dtype={dtype} device={device} "
                            f"min={x.min().item()} max={x.max().item()} nonzero={nz:.2f}%")
                else:
                    return f"shape={shape} dtype={dtype} device={device} numel={numel}"

            print(f"\n=== {name} ===")
            print(f"position_ids            : {stats(position_ids)}")
            print(f"inputs_embeds           : {stats(inputs_embeds)}")
            print(f"attention_mask          : {stats(attention_mask)}")
            print(f"visual_pos_masks        : {stats(image_mask)}")
            print(f"deepstack_visual_embeds : {stats(deep_stack_viz)}")

        # call it right before language_model(...)
        # print_lm_input_stats(position_ids, input_embeds, attention_mask, image_mask, deep_stack_viz)


        out = self.backbone.model.language_model(
            input_ids=None,
            position_ids = position_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deep_stack_viz,
            output_hidden_states=True,
            return_dict=True,
            cache_position = None,
            use_cache= False
        )
        hidden = out.hidden_states[-1]


        # hidden = self._encode_from_inputs_embeds(inputs_embeds, attention_mask)
        h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
        logits = self.enc_0(h_cls)
        losses = {}
        if label is not None:
            losses["ce_loss_combined"] = self._mc_ce_loss(logits, label)

        ###Unimodal text

        image_mask = image_mask.to(device).bool()
        keep = ~image_mask
        attention_mask_text = attention_mask * keep.to(attention_mask.dtype)
        out = self.backbone.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask_text,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deep_stack_viz,
            output_hidden_states=True,
            return_dict=True,
            cache_position=None,
            use_cache=False
        )
        hidden_text = out.hidden_states[-1]
        h_cls_text = self._get_cls_token_repr(hidden_text, input_ids).to(self.enc_0.linear.weight.dtype)
        logits_text = self.enc_0(h_cls_text)
        if label is not None:
            losses["ce_loss_c"] = self._mc_ce_loss(logits_text, label)

        ###Unimodal Image

        hint_mask = proc["hint_mask"]
        hint_mask = hint_mask.to(device).bool()
        keep = (~hint_mask)
        attention_mask_image = attention_mask * keep.to(attention_mask.dtype)

        out = self.backbone.model.language_model(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask_image,
            visual_pos_masks=image_mask,
            deepstack_visual_embeds=deep_stack_viz,
            output_hidden_states=True,
            return_dict=True,
            cache_position=None,
            use_cache=False
        )
        hidden_image = out.hidden_states[-1]

        h_cls_image = self._get_cls_token_repr(hidden_image, input_ids).to(self.enc_0.linear.weight.dtype)
        logits_image= self.enc_0(h_cls_image)
        if label is not None:
            losses["ce_loss_g"] = self._mc_ce_loss(logits_image, label)

        features = {"combined": h_cls, "c": h_cls_text, "g": h_cls_image}

        return {"preds": {"combined": logits,"c": logits_text,"g": logits_image}, "features":  features, "losses": losses}



class QwenVL_Cached(_QwenVL_CachedCombinedImpl):
    pass


class QwenVL_Cached_FullFT(_QwenVL_CachedCombinedImpl):
    """Supervised fine-tuning with some or all LLM layers unfrozen (no LoRA, no IHA).

    Config key ``finetune_layers`` (in model.args):
        "all"          → unfreeze the entire language model
        [20,21,22,27]  → unfreeze only those transformer block indices
    """

    def _setup_trainables(self):
        # Start frozen
        for p in self.backbone.parameters():
            p.requires_grad = False

        ft = getattr(self.args, "finetune_layers", "all")
        lm = self.backbone.model.language_model

        if ft == "all":
            for p in lm.parameters():
                p.requires_grad = True
        else:
            layers = lm.layers
            for idx in ft:
                for p in layers[int(idx)].parameters():
                    p.requires_grad = True

        # classifier head always trainable
        for p in self.enc_0.parameters():
            p.requires_grad = True


class QwenVL_Cached_Text(_QwenVL_CachedTextImpl):
    pass


class QwenVL_Cached_Image(_QwenVL_CachedImageImpl):
    pass


class QwenVL_Cached_SynIB(_QwenVL_CachedSynIBImpl):
    """Canonical cached SynIB family model for the ESNLI cache pipeline."""

    def _learned_masks_rmask(self, m1, m2, proc, label, **kwargs):
        device = proc["input_embeds"].device
        pcfg = self.args.get("perturb", {}) if isinstance(self.args, dict) else getattr(self.args, "perturb", {})
        steps = int(pcfg.get("steps", 5))
        lr = float(pcfg.get("lr", 5e-1))
        tau = float(pcfg.get("tau", 0.3))
        lsparse = float(pcfg.get("lsparse", 1.0))
        noise_std = float(pcfg.get("noise_std", 1.0))

        input_ids = proc["input_ids"]
        position_ids = proc["position_ids"]
        attention_mask = proc["attention_mask"]
        input_embeds = proc["input_embeds"]
        image_mask = proc["image_mask"]
        deep_stack_viz = proc["deep_stack_viz"]

        def run_logits(ie, im, dsv):
            hidden = self._encode_from_inputs_embeds(position_ids, ie, im, dsv, attention_mask)
            h_cls = self._get_cls_token_repr(hidden, input_ids).to(self.enc_0.linear.weight.dtype)
            return self.enc_0(h_cls)

        req = [p.requires_grad for p in self.parameters()]
        for p in self.parameters():
            p.requires_grad_(False)

        try:
            n_m1 = int(m1.sum().item())
            ell1 = torch.full((n_m1,), 1.0, device=device, dtype=torch.float32, requires_grad=True)
            opt1 = torch.optim.Adam([ell1], lr=lr)
            for _ in range(steps):
                g1 = torch.sigmoid(ell1 / tau).clamp(0, 1)
                ie = input_embeds.clone()
                g1_emb = g1.to(ie.dtype)
                ie[m1] = ie[m1] * g1_emb.unsqueeze(-1) + (1 - g1_emb.unsqueeze(-1)) * self.synib.z1_stats.noise_like(ie[m1], noise_std).to(ie.dtype)
                ie[m2] = self.synib.z2_stats.noise_like(ie[m2], noise_std).to(ie.dtype)
                this_dsv = []
                for di in range(len(deep_stack_viz)):
                    dsv_i = deep_stack_viz[di].clone()
                    dsv_i = self.synib.z2_deepstack_stats[di].noise_like(dsv_i, noise_std).to(dsv_i.dtype)
                    this_dsv.append(dsv_i)
                logits = run_logits(ie, image_mask, this_dsv)
                ce = F.cross_entropy(logits, label)
                sparsity = (1.0 - g1).mean()
                obj = (-ce) + lsparse * sparsity
                opt1.zero_grad(set_to_none=True)
                obj.backward()
                opt1.step()
            m1_keep_gate = torch.sigmoid(ell1 / tau).detach()

            n_m2 = int(m2.sum().item())
            ell2 = torch.full((n_m2,), 1.0, device=device, dtype=torch.float32, requires_grad=True)
            opt2 = torch.optim.Adam([ell2], lr=lr)
            for _ in range(steps):
                g2 = torch.sigmoid(ell2 / tau).clamp(0, 1)
                ie = input_embeds.clone()
                g2_emb = g2.to(ie.dtype)
                ie[m1] = self.synib.z1_stats.noise_like(ie[m1], noise_std).to(ie.dtype)
                ie[m2] = ie[m2] * g2_emb.unsqueeze(-1) + (1 - g2_emb.unsqueeze(-1)) * self.synib.z2_stats.noise_like(ie[m2], noise_std).to(ie.dtype)
                this_dsv = []
                for di in range(len(deep_stack_viz)):
                    dsv_i = deep_stack_viz[di].clone()
                    dsv_i = dsv_i * g2_emb.unsqueeze(-1) + (1 - g2_emb.unsqueeze(-1)) * self.synib.z2_deepstack_stats[di].noise_like(dsv_i, noise_std).to(dsv_i.dtype)
                    this_dsv.append(dsv_i)
                logits = run_logits(ie, image_mask, this_dsv)
                ce = F.cross_entropy(logits, label)
                sparsity = (1.0 - g2).mean()
                obj = (-ce) + lsparse * sparsity
                opt2.zero_grad(set_to_none=True)
                obj.backward()
                opt2.step()
            m2_keep_gate = torch.sigmoid(ell2 / tau).detach()
        finally:
            for p, r in zip(self.parameters(), req):
                p.requires_grad_(r)

        return m1_keep_gate, m2_keep_gate

    def _compute_logits_synib_from_proc(self, x, **kwargs):
        proc = x
        device = self.backbone.device
        input_ids = proc["input_ids"].to(device)
        attention_mask = proc["attention_mask"].to(device)
        if "image_mask" in proc:
            image_mask = proc["image_mask"].to(device)
        elif "visual_pos_masks" in proc:
            image_mask = proc["visual_pos_masks"].to(device)
        input_embeds = proc["input_embeds"].to(device)
        position_ids = proc["position_ids"].to(device)
        deep_stack_viz = proc["deepstack_visual_embeds"].to(device)
        deep_stack_viz = [deep_stack_viz[i] for i in range(len(deep_stack_viz))]

        m1 = proc.get("hint_mask", None)
        m2 = proc.get("visual_pos_masks", None)
        if m1 is None or m2 is None:
            raise KeyError("Need proc['hint_mask'] and proc['visual_pos_masks'] for SynIB mode.")
        m1 = m1.to(device).bool()
        m2 = m2.to(device).bool()

        self.synib.z1_stats.ema_update(proc["input_embeds"][m1])
        self.synib.z2_stats.ema_update(proc["input_embeds"][m2])
        for i in range(len(proc["deepstack_visual_embeds"])):
            self.synib.z2_deepstack_stats[i].ema_update(proc["deepstack_visual_embeds"][i])

        perturb_type = self.args.get("perturb", {}).get("type", "rand") if isinstance(self.args, dict) else getattr(self.args, "perturb", {}).get("type", "rand")
        m1forw, m2forw = self.synib._random_masks_randomp(m1, m2, True, True)

        if perturb_type == "rand":
            this_embed_1 = input_embeds.clone()
            this_embed_1[m2] = self.synib.z2_stats.noise_like(input_embeds[m2], 1.0).to(this_embed_1.dtype)
            deep_stack_viz_pass1 = [self.synib.z2_deepstack_stats[di].noise_like(deep_stack_viz[di], 1.0).to(deep_stack_viz[di].dtype) for di in range(len(deep_stack_viz))]
            this_embed_2 = input_embeds.clone()
            this_embed_2[m1] = self.synib.z1_stats.noise_like(input_embeds[m1], 1.0).to(this_embed_2.dtype)
            deep_stack_viz_pass2 = deep_stack_viz
        elif perturb_type == "learned":
            label = kwargs.get("label", None)
            proc_inner = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "input_embeds": input_embeds,
                "image_mask": image_mask,
                "deep_stack_viz": deep_stack_viz,
                "attention_mask": attention_mask,
            }
            kwargs_no_label = {k: v for k, v in kwargs.items() if k != "label"}
            m1_keep_gate, m2_keep_gate = self._learned_masks_rmask(m1, m2, proc_inner, label, **kwargs_no_label)
            m1_keep = m1_keep_gate.to(input_embeds.dtype)
            m2_keep = m2_keep_gate.to(input_embeds.dtype)
            this_embed_1 = input_embeds.clone()
            this_embed_1[m1] = input_embeds[m1] * m1_keep.unsqueeze(1) + (1 - m1_keep.unsqueeze(1)) * self.synib.z1_stats.noise_like(input_embeds[m1], 1.0).to(this_embed_1.dtype)
            this_embed_1[m2] = self.synib.z2_stats.noise_like(input_embeds[m2], 1.0).to(this_embed_1.dtype)
            deep_stack_viz_pass1 = [self.synib.z2_deepstack_stats[di].noise_like(deep_stack_viz[di], 1.0).to(deep_stack_viz[di].dtype) for di in range(len(deep_stack_viz))]
            this_embed_2 = input_embeds.clone()
            this_embed_2[m1] = self.synib.z1_stats.noise_like(input_embeds[m1], 1.0).to(this_embed_2.dtype)
            this_embed_2[m2] = input_embeds[m2] * m2_keep.unsqueeze(1) + (1 - m2_keep.unsqueeze(1)) * self.synib.z2_stats.noise_like(input_embeds[m2], 1.0).to(this_embed_2.dtype)
            deep_stack_viz_pass2 = []
            for di in range(len(deep_stack_viz)):
                dsv_i = deep_stack_viz[di].clone()
                dsv_i = dsv_i * m2_keep.unsqueeze(1) + (1 - m2_keep.unsqueeze(1)) * self.synib.z2_deepstack_stats[di].noise_like(dsv_i, 1.0).to(dsv_i.dtype)
                deep_stack_viz_pass2.append(dsv_i)
        else:
            raise ValueError(f"Unknown perturb.type: {perturb_type!r}")

        this_embed_0 = input_embeds.clone()
        deep_stack_viz_pass0 = [deep_stack_viz[di] for di in range(len(deep_stack_viz))]
        if (m1 != m1forw).any():
            m1forw = m1forw.to(this_embed_0.dtype)
            this_embed_0[m1] = this_embed_0[m1] * m1forw[m1].unsqueeze(1) + (1 - m1forw[m1].unsqueeze(1)) * self.synib.z1_stats.noise_like(this_embed_0[m1], 1.0).to(this_embed_0.dtype)
        if (m2 != m2forw).any():
            m2forw = m2forw.to(this_embed_0.dtype)
            this_embed_0[m2] = this_embed_0[m2] * m2forw[m2].unsqueeze(1) + (1 - m2forw[m2].unsqueeze(1)) * self.synib.z2_stats.noise_like(this_embed_0[m2], 1.0).to(this_embed_0.dtype)
            m2forw_keep = m2forw[m2].unsqueeze(1)
            deep_stack_viz_pass0 = []
            for di in range(len(deep_stack_viz)):
                dsv_i = deep_stack_viz[di].clone()
                dsv_i = dsv_i * m2forw_keep + (1 - m2forw_keep) * self.synib.z2_deepstack_stats[di].noise_like(dsv_i, 1.0).to(dsv_i.dtype)
                deep_stack_viz_pass0.append(dsv_i)

        if getattr(self.synib, "anchor_to_unimodal", False):
            # unimodal_text: m2 (vision) fully noised, m1 (text) fully kept
            this_embed_uni_t = input_embeds.clone()
            this_embed_uni_t[m2] = self.synib.z2_stats.noise_like(input_embeds[m2], 1.0).to(this_embed_uni_t.dtype)
            deep_stack_viz_uni_t = [self.synib.z2_deepstack_stats[di].noise_like(deep_stack_viz[di], 1.0).to(deep_stack_viz[di].dtype) for di in range(len(deep_stack_viz))]
            # unimodal_vision: m1 (text) fully noised, m2 (vision) fully kept, deep_stack_viz intact
            this_embed_uni_v = input_embeds.clone()
            this_embed_uni_v[m1] = self.synib.z1_stats.noise_like(input_embeds[m1], 1.0).to(this_embed_uni_v.dtype)
            deep_stack_viz_uni_v = [deep_stack_viz[di] for di in range(len(deep_stack_viz))]

            k = 5
            masks = torch.cat([attention_mask] * k, dim=0)
            position_ids_expanded = position_ids.repeat(1, k, 1)
            input_embeds_expanded = torch.cat([this_embed_0, this_embed_1, this_embed_2, this_embed_uni_t, this_embed_uni_v], dim=0)
            filter_deep_stack = torch.cat([image_mask] * k, dim=0)
            deep_stack_viz_extended = [torch.cat([deep_stack_viz_pass0[di], deep_stack_viz_pass1[di], deep_stack_viz_pass2[di], deep_stack_viz_uni_t[di], deep_stack_viz_uni_v[di]], dim=0) for di in range(len(deep_stack_viz))]
            hidden_all = self._encode_from_inputs_embeds(position_ids_expanded, input_embeds_expanded, filter_deep_stack, deep_stack_viz_extended, masks)
            ids_all = input_ids.repeat(k, 1)
            h_cls_all = self._get_cls_token_repr(hidden_all, ids_all)
            logits_all = self.enc_0(h_cls_all)
            head_logits, head_logits_0, head_logits_1, head_logits_uni_t, head_logits_uni_v = torch.chunk(logits_all, chunks=k, dim=0)
            h_cls, featcls_0, featcls_1, featcls_uni_t, featcls_uni_v = torch.chunk(h_cls_all, chunks=k, dim=0)
            losses = {}
            if "label" in kwargs and kwargs["label"] is not None:
                losses["ce_loss_combined"] = self._mc_ce_loss(head_logits, kwargs["label"])
            preds = {"combined": head_logits, "mask0": head_logits_0, "mask1": head_logits_1,
                     "unimodal_text": head_logits_uni_t, "unimodal_vision": head_logits_uni_v}
            features = {"combined": h_cls, "mask0": featcls_0, "mask1": featcls_1,
                        "unimodal_text": featcls_uni_t, "unimodal_vision": featcls_uni_v}
        else:
            masks = torch.cat([attention_mask, attention_mask, attention_mask], dim=0)
            position_ids_expanded = position_ids.repeat(1, 3, 1)
            input_embeds_expanded = torch.cat([this_embed_0, this_embed_1, this_embed_2], dim=0)
            filter_deep_stack = torch.cat([image_mask, image_mask, image_mask], dim=0)
            deep_stack_viz_extended = [torch.cat([deep_stack_viz_pass0[di], deep_stack_viz_pass1[di], deep_stack_viz_pass2[di]], dim=0) for di in range(len(deep_stack_viz))]
            hidden_all = self._encode_from_inputs_embeds(position_ids_expanded, input_embeds_expanded, filter_deep_stack, deep_stack_viz_extended, masks)
            ids_all = input_ids.repeat(3, 1)
            h_cls_all = self._get_cls_token_repr(hidden_all, ids_all)
            logits_all = self.enc_0(h_cls_all)
            head_logits, head_logits_0, head_logits_1 = torch.chunk(logits_all, chunks=3, dim=0)
            h_cls, featcls_0, featcls_1 = torch.chunk(h_cls_all, chunks=3, dim=0)
            losses = {}
            if "label" in kwargs and kwargs["label"] is not None:
                losses["ce_loss_combined"] = self._mc_ce_loss(head_logits, kwargs["label"])
            preds = {"combined": head_logits, "mask0": head_logits_0, "mask1": head_logits_1}
            features = {"combined": h_cls, "mask0": featcls_0, "mask1": featcls_1}
        return {"preds": preds, "features": features, "losses": losses}


class QwenVL_Cached_SynIBU(QwenVL_Cached_SynIB):
    """SynIB Unimodal-anchored (SynIBU): KL toward unimodal baselines instead of random prior.

    Drives each masked-modality prediction toward the unimodal prediction of the
    complementary unmasked modality, measuring true synergy above unimodal baselines.
    Use config key synergy_type=\"unimodal_anchor\" in model args.
    """
    pass


class QwenVL_SynIBFaster(QwenVL_Cached_SynIB):
    pass


class QwenVL_Cached_MCR(_QwenVL_CachedMCRImpl):
    pass


class QwenVL_Cached_MMPareto(_QwenVL_CachedMMParetoImpl):
    pass


class QwenVL_Cached_DnR(QwenVL_Cached_MMPareto):
    def forward(self, x, *, label=None, return_features=False, **kwargs):
        out = super().forward(x, label=label, return_features=return_features, **kwargs)
        out["losses"].pop("ce_loss_combined", None)
        return out


class QwenVL_Cached_ReconBoost(QwenVL_Cached_MMPareto):
    def forward(self, x, *, label=None, return_features=False, **kwargs):
        out = super().forward(x, label=label, return_features=return_features, **kwargs)
        out["losses"].pop("ce_loss_combined", None)
        return out
