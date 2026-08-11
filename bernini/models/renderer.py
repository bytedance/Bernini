# Copyright (c) 2026 Bytedance Ltd. and/or its affiliate
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bernini Renderer model: a UMT5 text encoder + a Wan2.2 dual-expert diffusion decoder."""

import torch
import contextlib

from transformers import UMT5EncoderModel
from transformers.configuration_utils import PretrainedConfig
from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

from .wan_diffusion import GEN_Wanx22


# Captured at import time, before VeOmni's build_foundation_model wraps model
# construction in init_empty_weights() (which monkeypatches register_parameter
# to push params to the meta device). Submodules below load real pretrained
# weights via transformers v5 from_pretrained, which is incompatible with that
# patch; restore the original around those loads. FSDP2 reshards afterwards.
_ORIG_REGISTER_PARAMETER = nn.Module.register_parameter


@contextlib.contextmanager
def load_pretrained_submodules():
    patched = nn.Module.register_parameter
    nn.Module.register_parameter = _ORIG_REGISTER_PARAMETER
    try:
        yield
    finally:
        nn.Module.register_parameter = patched


class BerniniRendererConfig(PretrainedConfig):
    model_type = "bernini_renderer"

    def __init__(
        self,
        wan22_base: str = None,
        diff_dec_config_path: str = None,
        skip_transformer_1: bool = False,
        skip_transformer_2: bool = False,
        switch_dit_boundary: float = 0.875,
        max_sequence_length: int = 512,
        shift: float = 3.0,
        boundary_ratio: float = 0.3,
        use_unipc: bool = True,
        scratch: bool = False,
        ema_decay: float = None,
        use_src_id_rotary_emb: bool = True,
        interpolate_src_id: bool = True,
        max_trained_src_id: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.wan22_base = wan22_base or diff_dec_config_path
        self.diff_dec_config_path = diff_dec_config_path or wan22_base
        self.skip_transformer_1 = skip_transformer_1
        self.skip_transformer_2 = skip_transformer_2
        self.switch_dit_boundary = switch_dit_boundary
        self.max_sequence_length = max_sequence_length
        self.shift = shift
        self.boundary_ratio = boundary_ratio
        self.use_unipc = use_unipc
        self.scratch = scratch
        self.ema_decay = ema_decay
        self.use_src_id_rotary_emb = use_src_id_rotary_emb
        # When the number of reference sources exceeds `max_trained_src_id`
        # (the largest source_id seen in training), evenly map their ids into
        # the trained range [1, max_trained_src_id] instead of extrapolating to
        # unseen integer ids. The noisy target keeps source_id 0.
        self.interpolate_src_id = interpolate_src_id
        self.max_trained_src_id = max_trained_src_id
        self.architectures = ["BerniniRendererModel"]


@dataclass
class BerniniRendererOutputWithPast(CausalLMOutputWithPast):
    diff_loss: Optional[torch.Tensor] = None


class EMAModule(nn.Module):
    def __init__(self, model: nn.Module, ema_decay: float = 0.9999, scratch: bool = False):
        import copy

        super().__init__()
        self.ema_decay = ema_decay
        self.scratch = scratch
        for name, module in model.named_children():
            self.add_module(name, copy.deepcopy(module))
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def update(self, model: nn.Module):
        model_params = dict(model.named_parameters())
        ema_params = dict(self.named_parameters())
        for name, param in model_params.items():
            name = name.removeprefix("_fsdp_wrapped_module.")
            if name not in ema_params:
                continue
            if self.scratch:
                ema_params[name].data.copy_(param.detach().data)
            else:
                ema_params[name].data.mul_(self.ema_decay).add_(param.detach().data, alpha=1 - self.ema_decay)
        self.scratch = False
        self.eval()


class BerniniRendererModel(PreTrainedModel):
    config_class = BerniniRendererConfig
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _no_split_modules = ["UMT5Block", "WanTransformerBlock"]

    def __init__(self, config: BerniniRendererConfig):
        super().__init__(config)
        self.max_sequence_length = config.max_sequence_length
        # Match the parent model dtype so FSDP2 sees uniform parameter dtypes.
        model_dtype = getattr(config, "dtype", None) or torch.bfloat16
        with load_pretrained_submodules():
            self.t5_text_encoder = UMT5EncoderModel.from_pretrained(
                config.wan22_base, subfolder="text_encoder", torch_dtype=model_dtype
            )
            self.diff_dec = GEN_Wanx22(config)
        self.t5_text_encoder.requires_grad_(False)
        if config.ema_decay is not None:
            self.ema = EMAModule(self, ema_decay=config.ema_decay, scratch=config.scratch)
        self.post_init()

    def _set_gradient_checkpointing(self, enable=True, gradient_checkpointing_func=None):
        for module in self.modules():
            if hasattr(module, "gradient_checkpointing"):
                module.gradient_checkpointing = enable
                if gradient_checkpointing_func is not None:
                    module._gradient_checkpointing_func = gradient_checkpointing_func

    def init_weights(self):
        """Re-materialize pretrained weights instead of HF random init.

        ``post_init`` (during ``__init__``) and VeOmni's meta + FSDP2 flow both
        call this. In ``__init__`` the submodules already hold real pretrained
        weights, so this is a no-op. Under FSDP2 the framework calls
        ``to_empty()`` (turning params into uninitialized sharded DTensors) then
        ``init_weights()``; HF's default would random-init and clobber the
        Wan2.2/UMT5 pretrained weights (and the DTensor RNG broadcast it issues
        hangs the run). Instead, reload the pretrained submodules and copy them
        into the (possibly sharded) params.

        To avoid every rank materializing the (large fp32) pretrained
        submodules, only rank0 builds the reference state dict; the other ranks
        receive each tensor through ``distribute_tensor``'s rank0 scatter
        (``src_data_rank=0``). Whether a param has a source is decided purely
        from its name so every rank issues the same collectives.
        """
        import torch.distributed as dist
        from torch.distributed.tensor import DTensor, distribute_tensor

        params = dict(self.named_parameters())
        if not any(p.is_meta or isinstance(p, DTensor) for p in params.values()):
            return

        is_dist = dist.is_available() and dist.is_initialized()
        is_rank0 = (not is_dist) or dist.get_rank() == 0
        reference = self._pretrained_reference_state_dict() if is_rank0 else {}

        def _source_key(name: str):
            # EMA params mirror the main weights (matches the deepcopy init in
            # EMAModule); strip the ``ema.`` prefix to find their source tensor.
            base = name.removeprefix("ema.")
            if base.startswith("t5_text_encoder.") or base.startswith("diff_dec."):
                return base
            return None

        for name, param in params.items():
            key = _source_key(name)
            if key is None:
                # Deterministic across ranks: no collective issued for these.
                continue
            if isinstance(param, DTensor):
                if is_rank0:
                    source = reference[key].to(dtype=param.dtype)
                else:
                    # Placeholder for shape/dtype only; data is ignored because
                    # distribute_tensor scatters from src_data_rank=0.
                    source = torch.empty(param.shape, dtype=param.dtype)
                sharded = distribute_tensor(
                    source,
                    param.device_mesh,
                    param.placements,
                    src_data_rank=0,
                )
                param.data.copy_(sharded)
            else:
                # Unsharded meta path (single process): load locally on rank0.
                param.data.copy_(reference[key].to(device=param.device, dtype=param.dtype))

    def _pretrained_reference_state_dict(self):
        """Build a CPU state dict of the pretrained UMT5 + Wan2.2 submodules."""
        with load_pretrained_submodules():
            t5 = UMT5EncoderModel.from_pretrained(
                self.config.wan22_base, subfolder="text_encoder", torch_dtype=torch.float32
            )
            diff_dec = GEN_Wanx22(self.config)
        state_dict = {f"t5_text_encoder.{k}": v for k, v in t5.state_dict().items()}
        state_dict.update({f"diff_dec.{k}": v for k, v in diff_dec.state_dict().items()})
        return state_dict

    def get_ignore_modules_in_mixed_precision(self):
        from diffusers.models.embeddings import TimestepEmbedding
        from diffusers.models.normalization import FP32LayerNorm

        return (TimestepEmbedding, FP32LayerNorm)

    def encode_prompt(self, input_ids, attention_mask):
        """Encode token ids into padded T5 embeddings `[B, max_len, hidden]`."""
        seq_lens = attention_mask.gt(0).sum(dim=1).long()
        with torch.no_grad():
            prompt_embeds = self.t5_text_encoder(input_ids, attention_mask).last_hidden_state
        prompt_embeds = [u[: min(v, self.max_sequence_length)] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(self.max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )
        return prompt_embeds

    def get_t5_text_embeddings(self, input_ids, attention_mask, input_lens):
        input_ids = input_ids.squeeze(0)
        attention_mask = attention_mask.squeeze(0)
        input_lens = input_lens.squeeze(0)
        # SequenceParallelCollator pads 1-D packed metadata (including
        # ``t5_input_lens``) to the Ulysses size with zeros. Drop those
        # padding entries and trim matching padded token tails before splitting.
        input_lens = input_lens[input_lens > 0]
        valid_token_len = int(input_lens.sum().item())
        input_ids = input_ids[:valid_token_len]
        attention_mask = attention_mask[:valid_token_len]
        input_ids_list = torch.split(input_ids, input_lens.tolist())
        attention_mask_list = torch.split(attention_mask, input_lens.tolist())
        padded_input_ids = []
        padded_attention_mask = []
        seq_lens = torch.clamp(input_lens, max=self.max_sequence_length)
        for ids, mask in zip(input_ids_list, attention_mask_list):
            seq_len = ids.size(0)
            if seq_len < self.max_sequence_length:
                pad_len = self.max_sequence_length - seq_len
                ids = torch.cat([ids, ids.new_zeros(pad_len)])
                mask = torch.cat([mask, mask.new_zeros(pad_len)])
            else:
                ids = ids[: self.max_sequence_length]
                mask = mask[: self.max_sequence_length]
            padded_input_ids.append(ids)
            padded_attention_mask.append(mask)

        with torch.no_grad():
            prompt_embeds = self.t5_text_encoder(
                torch.stack(padded_input_ids), torch.stack(padded_attention_mask)
            ).last_hidden_state
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(self.max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )
        return [self.max_sequence_length] * len(input_lens), prompt_embeds.view(1, -1, prompt_embeds.size(-1))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        t5_input_lens: Optional[torch.Tensor] = None,
        input_vae_latents: Optional[torch.Tensor] = None,
        input_vae_rope: Optional[torch.Tensor] = None,
        vae_latents_mask: Optional[torch.Tensor] = None,
        vae_seqlen: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
        target_velocity: Optional[Any] = None,
        **kwargs,
    ):
        if hasattr(target_velocity, "tensor"):
            target_velocity = target_velocity.tensor
        batch_text_seqlen, batch_text_embs = self.get_t5_text_embeddings(input_ids, attention_mask, t5_input_lens)
        valid_sample_count = len(batch_text_seqlen)
        if vae_seqlen is not None:
            vae_seqlen = vae_seqlen.squeeze(0)
            vae_seqlen = vae_seqlen[vae_seqlen > 0].unsqueeze(0)
        if timesteps is not None:
            timesteps = timesteps.squeeze(0)[:valid_sample_count].unsqueeze(0)
        diff_loss = self.diff_dec(
            input_vae_latents=input_vae_latents,
            input_vae_rope=input_vae_rope,
            vae_latents_mask=vae_latents_mask,
            vae_seqlen=vae_seqlen,
            text_embs=batch_text_embs,
            batch_text_seqlen=batch_text_seqlen,
            timesteps=timesteps,
            target_velocity=target_velocity,
        )
        return BerniniRendererOutputWithPast(diff_loss=diff_loss)

    @torch.no_grad()
    def sample(
        self,
        input_ids,
        attention_mask,
        uncond_input_ids,
        uncond_attention_mask,
        image_vae_latents=None,
        multi_video_vae_latents=None,
        multi_image_vae_latents=None,
        num_frames: int = 1,
        width: int = 832,
        height: int = 480,
        num_inference_steps: int = 50,
        guidance_mode: str = "rv2v",
        omega_vid: float = 3.0,
        omega_img: float = 3.0,
        omega_txt: float = 4.0,
        omega_scale: float = 0.75,
        flow_shift: float = 5.0,
        seed: int = 42,
        device="cuda",
        eta: float = 0.5,
        norm_threshold=(50.0, 50.0),
        momentum: float = -0.5,
    ):
        self.t5_text_encoder = self.t5_text_encoder.to(device)
        prompt_embeds = self.encode_prompt(input_ids, attention_mask)
        uncond_prompt_embeds = (
            self.encode_prompt(uncond_input_ids, uncond_attention_mask)
            if uncond_input_ids is not None
            else None
        )
        self.t5_text_encoder = self.t5_text_encoder.to("cpu")
        torch.cuda.empty_cache()

        return self.diff_dec.sample(
            prompt_embeds=prompt_embeds,
            uncond_prompt_embeds=uncond_prompt_embeds,
            image_vae_latents=image_vae_latents,
            multi_video_vae_latents=multi_video_vae_latents,
            multi_image_vae_latents=multi_image_vae_latents,
            num_frames=num_frames,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_mode=guidance_mode,
            omega_vid=omega_vid,
            omega_img=omega_img,
            omega_txt=omega_txt,
            omega_scale=omega_scale,
            flow_shift=flow_shift,
            seed=seed,
            device=device,
            eta=eta,
            norm_threshold=norm_threshold,
            momentum=momentum,
        )
