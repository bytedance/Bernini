import io
import json
import math
import random
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Callable, Dict, List, Optional

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from transformers import PreTrainedTokenizerBase

from bernini.models.scheduler import FlowMatchScheduler


SYSTEM_PROMPTS = {
    "default": "You are a helpful assistant.",
    "t2i": "You are a helpful assistant specialized in text-to-image generation.",
    "t2v": "You are a helpful assistant specialized in text-to-video generation.",
    "i2i": "You are a helpful assistant specialized in image editing.",
    "r2i": "You are a helpful assistant specialized in subject-to-image generation.",
    "i2v": "You are a helpful assistant specialized in image-to-video generation.",
    "v2v": "You are a helpful assistant specialized in video editing.",
    "r2v": "You are a helpful assistant specialized in subject-to-video generation.",
    "vi2v": "You are a helpful assistant specialized in video editing on content propagation.",
    "vr2v": "You are a helpful assistant specialized in video editing with reference.",
    "ads2v": "You are a helpful assistant specialized in ads insertion.",
    "vrc2v": "You are a helpful assistant for editing. You may need to adjust the subject's action or position.",
    "mv2v": "You are a helpful assistant for editing. You might need to adjust the video's style, lighting, colors, textures, and the subject's pose or action.",
}


def _is_non_empty(value: Any) -> bool:
    return isinstance(value, Iterable) and not isinstance(value, (str, bytes)) and len(value) > 0


def _load_tensor(blob: Any) -> torch.Tensor:
    if isinstance(blob, torch.Tensor):
        return blob
    buffer = io.BytesIO(blob)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu")


def shift2boundary(shift, sigma_min=0, sigma_max=1, denoising_strength=1.0, num_steps=1000):
    sigma_start = sigma_min + (sigma_max - sigma_min) * denoising_strength
    sigmas = torch.linspace(sigma_start, sigma_min, num_steps + 1)[:-1]
    return shift * sigmas / (1 + (shift - 1) * sigmas)


def find_nearest_boundary(sigmas, sigma_value):
    return int(torch.argmin((sigmas - sigma_value).abs()).item())


def compute_density_for_timestep_sampling(
    weighting_scheme: str,
    batch_size: int,
    logit_mean: float = 0.5,
    logit_std: float = 1.0,
    mode_scale: float = 1.29,
    min_t: float = 0.0,
    max_t: float = 1.0,
):
    samples = []
    for _ in range(batch_size):
        while True:
            if weighting_scheme == "logit_normal":
                u = torch.sigmoid(torch.normal(mean=logit_mean, std=logit_std, size=(1,), device="cpu"))
            elif weighting_scheme == "mode":
                raw = torch.rand(size=(1,), device="cpu")
                u = 1 - raw - mode_scale * (torch.cos(math.pi * raw / 2) ** 2 - 1 + raw)
            else:
                u = torch.rand(size=(1,), device="cpu") * (max_t - min_t) + min_t
            if min_t <= float(u.item()) <= max_t:
                samples.append(u)
                break
    return torch.cat(samples, dim=0)


class NoiseScheduler:
    def __init__(
        self,
        shift_config: Optional[Dict[str, float]] = None,
        weighting_scheme_config: Optional[Dict[str, str]] = None,
        noise_tmin: float = 0.0,
        noise_tmax: float = 1.0,
        logit_mean: float = 0.5,
        logit_std: float = 1.0,
        mode_scale: float = 1.29,
    ):
        self.shift_config = {"default": 5.0, **(shift_config or {})}
        self.weighting_scheme_config = {
            "default": "logit_normal",
            "image": "logit_normal",
            "video": "mode",
            **(weighting_scheme_config or {}),
        }
        self.noise_tmin = noise_tmin
        self.noise_tmax = noise_tmax
        self.logit_mean = logit_mean
        self.logit_std = logit_std
        self.mode_scale = mode_scale
        self.flow_scheduler = {}
        for shift in set(self.shift_config.values()):
            sigmas = shift2boundary(shift)
            bound1 = find_nearest_boundary(sigmas, self.noise_tmin) / 1000
            bound2 = find_nearest_boundary(sigmas, self.noise_tmax) / 1000
            scheduler = FlowMatchScheduler(shift=shift, sigma_min=0.0, extra_one_step=True)
            scheduler.set_timesteps(1000, training=True, device="cpu")
            self.flow_scheduler[shift] = {
                "tmin": min(bound1, bound2),
                "tmax": max(bound1, bound2),
                "scheduler": scheduler,
            }

    def get_noise_sigma(self, task_name: str):
        if task_name in self.weighting_scheme_config:
            weighting_name = task_name
        elif "2" in task_name and task_name.rsplit("2", 1)[-1] == "i":
            weighting_name = "image"
        elif "2" in task_name and task_name.rsplit("2", 1)[-1] == "v":
            weighting_name = "video"
        else:
            weighting_name = "default"
        shift_name = task_name if task_name in self.shift_config else "default"
        cfg = self.flow_scheduler[self.shift_config[shift_name]]
        u = compute_density_for_timestep_sampling(
            self.weighting_scheme_config[weighting_name],
            batch_size=1,
            logit_mean=self.logit_mean,
            logit_std=self.logit_std,
            mode_scale=self.mode_scale,
            min_t=cfg["tmin"],
            max_t=cfg["tmax"],
        )
        scheduler = cfg["scheduler"]
        timestep_id = (u * scheduler.num_train_timesteps).long()
        timestep = scheduler.timesteps[timestep_id]
        sigma = scheduler.get_noise_sigma(timestep)
        return sigma.reshape(-1), timestep.reshape(-1)


def encode_renderer_messages(
    conversations: List[Dict[str, Any]],
    tokenizer: PreTrainedTokenizerBase,
    task_name: str,
    drop_text: bool,
    drop_video: bool,
    drop_img: bool,
) -> Dict[str, torch.Tensor]:
    image_vit_mask, video_vit_mask = [], []
    image_drop_mask, video_drop_mask = [], []
    vae_type_list = []
    texts = []
    for message in conversations:
        msg_type = message.get("type")
        if msg_type == "special_token":
            continue
        has_loss = message.get("has_loss", 1 if msg_type == "video_gen" else 0)
        if msg_type == "cot_text":
            msg_type, has_loss = "text", 0
        if msg_type == "text" and has_loss == 0:
            if not drop_text:
                texts.append(message.get("text", ""))
        elif msg_type in ("image", "image_gen"):
            image_vit_mask.append(has_loss)
            image_drop_mask.append(int(drop_img))
            if not drop_img or has_loss == 1:
                vae_type_list.append(0)
        elif msg_type in ("video", "frame_gen", "video_gen"):
            video_vit_mask.append(has_loss)
            video_drop_mask.append(int(drop_video))
            if not drop_video or has_loss == 1:
                vae_type_list.append(1)
        else:
            raise ValueError(f"Unknown message type: {msg_type}")

    prompt = " ".join(texts)
    prompt = SYSTEM_PROMPTS.get(task_name, SYSTEM_PROMPTS["default"]) + prompt_clean(prompt)
    tokenized = tokenizer(prompt, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    input_ids = tokenized.input_ids.squeeze(0)
    attention_mask = tokenized.attention_mask.squeeze(0)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "t5_input_lens": torch.tensor([input_ids.shape[0]], dtype=torch.long),
        "image_vit_mask": torch.tensor(image_vit_mask, dtype=torch.bool),
        "video_vit_mask": torch.tensor(video_vit_mask, dtype=torch.bool),
        "image_drop_mask": torch.tensor(image_drop_mask, dtype=torch.bool),
        "video_drop_mask": torch.tensor(video_drop_mask, dtype=torch.bool),
        "vae_type_list": torch.tensor(vae_type_list, dtype=torch.long),
    }


def _filter_out_target_clip(inputs, mask, embeds_key, grid_key):
    if embeds_key not in inputs or grid_key not in inputs:
        return
    embeds, grids = [], []
    for emb, grid, drop in zip(inputs[embeds_key], inputs[grid_key], mask):
        if not bool(drop):
            embeds.append(emb)
            grids.append(grid)
    if embeds:
        inputs[embeds_key] = torch.cat(embeds, dim=0)
        inputs[grid_key] = torch.stack(grids)
    else:
        inputs.pop(embeds_key)
        inputs.pop(grid_key)


def _filter_source_vae(inputs, vae_latents, vit_mask, drop_vision, latent_key, mask_key):
    kept_latents, kept_mask = [], []
    for vae_emb, is_target in zip(vae_latents, vit_mask):
        if not drop_vision or bool(is_target):
            kept_latents.append(vae_emb)
            kept_mask.append(bool(is_target))
    inputs[latent_key] = kept_latents
    inputs[mask_key] = torch.tensor(kept_mask, dtype=torch.bool)


def _rearrange_vae_feature(vae_emb):
    return rearrange(vae_emb, "c (t pt) (h ph) (w pw) -> (t h w) c pt ph pw", pt=1, ph=2, pw=2)


def pack_vae_latents(
    vae_rope_func: Callable,
    vae_type_list: torch.Tensor,
    image_inputs: Dict[str, Any],
    video_inputs: Dict[str, Any],
    noise_sigma: torch.Tensor,
    max_vae_frames: Optional[int] = None,
):
    image_vae_list = iter(image_inputs.pop("image_vae_latents", []))
    image_vae_mask_list = iter(image_inputs.pop("image_vae_mask", []))
    video_vae_list = iter(video_inputs.pop("video_vae_latents", []))
    video_vae_mask_list = iter(video_inputs.pop("video_vae_mask", []))
    input_vae_latents, input_vae_rope, vae_latents_mask = [], [], []
    target_velocity, target_lens = [], []
    for idx, vae_type in enumerate(vae_type_list.tolist()):
        if vae_type == 0:
            vae_emb = next(image_vae_list)
            vae_mask = bool(next(image_vae_mask_list))
        else:
            vae_emb = next(video_vae_list)
            vae_mask = bool(next(video_vae_mask_list))

        if max_vae_frames is not None and vae_emb.shape[1] > max_vae_frames:
            vae_emb = vae_emb[:, :max_vae_frames]

        source_id = 0 if vae_mask else idx + 1
        vae_rope = vae_rope_func(vae_emb.unsqueeze(0), source_id=source_id).squeeze(0)
        input_vae_rope.append(vae_rope)
        packed = _rearrange_vae_feature(vae_emb)
        vae_latents_mask.extend([vae_mask] * packed.shape[0])
        if vae_mask:
            noise = torch.randn_like(packed, dtype=torch.float32)
            input_vae_latents.append((1 - noise_sigma) * packed + noise_sigma * noise)
            target_velocity.append(noise - packed.float())
            target_lens.append(packed.shape[0])
        else:
            input_vae_latents.append(packed)

    input_vae_latents = torch.cat(input_vae_latents, dim=0)
    target_velocity = torch.cat(target_velocity, dim=0)
    input_vae_rope = torch.cat(input_vae_rope, dim=1)
    return {
        "input_vae_latents": input_vae_latents,
        "input_vae_rope": input_vae_rope.permute(1, 0, 2),
        "vae_latents_mask": torch.tensor(vae_latents_mask, dtype=torch.bool),
        "vae_seqlen": torch.tensor([input_vae_latents.shape[0]], dtype=torch.long),
        "target_velocity": target_velocity,
        "target_lens": torch.tensor(target_lens, dtype=torch.long),
    }


def process_renderer_sample(
    sample: Dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    vae_rope_func: Callable,
    vae_latent_mean: torch.Tensor,
    vae_latent_std: torch.Tensor,
    noise_scheduler: NoiseScheduler,
    text_dropout_rate: float = 0.0,
    img_dropout_rate: float = 0.0,
    video_dropout_rate: float = 0.0,
    max_vae_frames: Optional[int] = None,
    source_name: str = "",
    **kwargs,
):
    source_name = source_name or sample.get("source_name", "") or ""
    task_name = source_name.split("$")[0].lower() or "default"
    noise_sigma, noise_timestep = noise_scheduler.get_noise_sigma(task_name)
    drop_text = random.random() < text_dropout_rate
    drop_img = random.random() < img_dropout_rate
    drop_video = random.random() < video_dropout_rate
    tokenized = encode_renderer_messages(
        json.loads(sample["inputs"]), tokenizer, task_name, drop_text, drop_video, drop_img
    )

    image_inputs, video_inputs = {}, {}
    if _is_non_empty(sample.get("image_embeds", [])):
        image_inputs = {"image_embeds": [], "image_grid_thw": []}
        for emb, thw in zip(sample["image_embeds"], sample["image_grid_thw"]):
            image_inputs["image_embeds"].append(_load_tensor(emb))
            image_inputs["image_grid_thw"].append(torch.as_tensor(thw))
        image_inputs["image_grid_thw"] = torch.stack(image_inputs["image_grid_thw"])
    if _is_non_empty(sample.get("video_embeds", [])):
        video_inputs = {"video_embeds": [], "video_grid_thw": []}
        for emb, thw in zip(sample["video_embeds"], sample["video_grid_thw"]):
            video_inputs["video_embeds"].append(_load_tensor(emb))
            video_inputs["video_grid_thw"].append(torch.as_tensor(thw))
        video_inputs["video_grid_thw"] = torch.stack(video_inputs["video_grid_thw"])

    image_drop_mask = tokenized.pop("image_drop_mask")
    video_drop_mask = tokenized.pop("video_drop_mask")
    image_vit_mask = tokenized.pop("image_vit_mask")
    video_vit_mask = tokenized.pop("video_vit_mask")
    if "image_embeds" in image_inputs:
        _filter_out_target_clip(image_inputs, image_vit_mask | image_drop_mask, "image_embeds", "image_grid_thw")
    if "video_embeds" in video_inputs:
        _filter_out_target_clip(video_inputs, video_vit_mask | video_drop_mask, "video_embeds", "video_grid_thw")

    if _is_non_empty(sample.get("image_vae_latents", [])):
        image_latents = []
        for vae_blob in sample["image_vae_latents"]:
            vae = DiagonalGaussianDistribution(_load_tensor(vae_blob)).sample().squeeze(0)
            image_latents.append((vae - vae_latent_mean) / vae_latent_std)
        _filter_source_vae(image_inputs, image_latents, image_vit_mask, drop_img, "image_vae_latents", "image_vae_mask")
    if _is_non_empty(sample.get("video_vae_latents", [])):
        video_latents = []
        for vae_blob in sample["video_vae_latents"]:
            vae = DiagonalGaussianDistribution(_load_tensor(vae_blob)).sample().squeeze(0)
            video_latents.append((vae - vae_latent_mean) / vae_latent_std)
        _filter_source_vae(video_inputs, video_latents, video_vit_mask, drop_video, "video_vae_latents", "video_vae_mask")

    vae_type_list = tokenized.pop("vae_type_list")
    packed = pack_vae_latents(
        vae_rope_func,
        vae_type_list,
        image_inputs,
        video_inputs,
        noise_sigma,
        max_vae_frames=max_vae_frames,
    )
    tokenized.update(packed)
    tokenized["timesteps"] = noise_timestep
    tokenized["vlm_seqlen"] = torch.tensor([tokenized["attention_mask"].sum()], dtype=torch.long)
    tokenized["num_tokens"] = torch.tensor([tokenized["vae_seqlen"][0] + tokenized["vlm_seqlen"][0]], dtype=torch.long)
    return [tokenized]
