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

import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import timedelta
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist


# Compatibility shims for VeOmni v0.1.11 import paths.
import logging as builtin_logging
import types

try:
    from pyiceberg.metrics import LoggingMetricsReporter as _LoggingMetricsReporter  # noqa: F401
except Exception:
    metrics = types.ModuleType("pyiceberg.metrics")

    class LoggingMetricsReporter:  # noqa: D401
        _logger = builtin_logging.getLogger("LoggingMetricsReporter")

    metrics.LoggingMetricsReporter = LoggingMetricsReporter
    sys.modules["pyiceberg.metrics"] = metrics

try:
    from transformers.initialization import no_init_weights as _no_init_weights  # noqa: F401
except Exception:
    from transformers.modeling_utils import no_init_weights

    initialization = types.ModuleType("transformers.initialization")
    initialization.no_init_weights = no_init_weights
    sys.modules["transformers.initialization"] = initialization

from torch.utils.checkpoint import set_checkpoint_debug_enabled
from tqdm import trange

from bernini.models.transformer_wan import WanRotaryPosEmbed
from bernini.training.data import NoiseScheduler, process_renderer_sample
from bernini.training.register import register_bernini_renderer_to_veomni
from veomni.arguments import DataArguments, ModelArguments, TrainingArguments, VeOmniArguments, parse_args, save_args
from veomni.checkpoint import build_checkpointer
from veomni.data.data_collator import DataCollateInfo, MainCollator
import veomni.data.data_loader as veomni_data_loader
from veomni.data.data_loader import build_dataloader
import veomni.data.dataset as veomni_dataset
from veomni.data.dataset import build_dataset
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.offloading import build_activation_offloading_context
from veomni.distributed.parallel_state import get_parallel_state, init_parallel_state
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import build_foundation_model, save_model_assets
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils import helper
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device, synchronize
from veomni.utils.dist_utils import all_reduce
from veomni.utils.multisource_utils import parse_multisource_config


logger = helper.create_logger(__name__)


def seed_everything(seed: int = 0) -> None:
    """Seed Python, NumPy, and Torch RNGs.

    The renderer dataloader/transform uses Python ``random`` for condition
    dropout, NumPy may be used by worker-side dependencies, and Torch RNG is
    used for diffusion noise / VAE posterior sampling.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def patch_dataloader_worker_seed(base_seed: int) -> None:
    """Seed dataloader workers deterministically inside each SP group.

    VeOmni's native dataloader only installs a worker init function when
    ``worker_num_threads`` is set. The training main below enables that path for
    multi-worker renderer runs, and this patch extends the init function to also
    seed all RNGs. Workers with the same ``worker_id`` in the same SP group get
    the same seed; different SP/data-parallel groups use different base seeds.
    """

    def _build_worker_init_fn(worker_num_threads: int):
        def worker_init_fn(worker_id: int) -> None:
            torch.set_num_threads(worker_num_threads)
            seed_everything(base_seed + worker_id)

        return worker_init_fn

    veomni_data_loader._build_worker_init_fn = _build_worker_init_fn


@dataclass
class BerniniRendererModelArguments(ModelArguments):
    vae_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Wan VAE config.json with latents_mean/latents_std."},
    )


@dataclass
class BerniniRendererDataArguments(DataArguments):
    data_type: str = "diffusion"
    mm_configs: Optional[Dict[str, Any]] = field(default_factory=dict)
    noise_scheduler_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    text_dropout_rate: float = 0.0
    img_dropout_rate: float = 0.0
    video_dropout_rate: float = 0.0
    max_vae_frames: Optional[int] = None
    lowr_noise_aug_range: Optional[List[float]] = None
    lowr_noise_add_range: Optional[List[float]] = None

    def __post_init__(self):
        # Diffusion data uses a custom transform and does not read text_keys, but
        # the base DataArguments rejects data_type="diffusion" when text_keys is unset.
        if self.text_keys is None:
            self.text_keys = "caption"
        super().__post_init__()
        self.task_lists = []
        if self.enable_multisource:
            cfg = parse_multisource_config(self.train_path)
            self.task_lists = sorted(
                {name.split("$")[0].lower() for name in cfg.get("names", []) if not name.lower().startswith("und")}
            )


@dataclass
class BerniniRendererTrainingArguments(TrainingArguments):
    freeze_modules: Optional[List[str]] = field(default_factory=lambda: ["t5_text_encoder"])
    detach_modules: Optional[List[str]] = field(default_factory=list)
    use_src_id_rotary_emb: bool = True


@dataclass
class BerniniRendererArguments(VeOmniArguments):
    model: BerniniRendererModelArguments = field(default_factory=BerniniRendererModelArguments)
    data: BerniniRendererDataArguments = field(default_factory=BerniniRendererDataArguments)
    train: BerniniRendererTrainingArguments = field(default_factory=BerniniRendererTrainingArguments)



def get_renderer_diffusion_length(sample: Dict[str, Any]) -> int:
    """Return diffusion token count for dynamic batching.

    VeOmni's default dynamic batching uses ``attention_mask.sum()``, which only
    counts text tokens and badly underestimates renderer diffusion samples. The
    renderer transform already computes ``num_tokens = vae_seqlen + vlm_seqlen``;
    use it when present and fall back to VAE/text lengths for robustness.
    """
    num_tokens = sample.get("num_tokens")
    if isinstance(num_tokens, torch.Tensor):
        return int(num_tokens.sum().item())
    if num_tokens is not None:
        return int(num_tokens)

    vae_seqlen = sample.get("vae_seqlen")
    vlm_seqlen = sample.get("vlm_seqlen")
    total = 0
    if isinstance(vae_seqlen, torch.Tensor):
        total += int(vae_seqlen.sum().item())
    elif vae_seqlen is not None:
        total += int(vae_seqlen)
    if isinstance(vlm_seqlen, torch.Tensor):
        total += int(vlm_seqlen.sum().item())
    elif vlm_seqlen is not None:
        total += int(vlm_seqlen)
    if total > 0:
        return total

    attention_mask = sample.get("attention_mask")
    if isinstance(attention_mask, torch.Tensor):
        return int(attention_mask.sum().item())
    return int(len(sample.get("input_ids", [])))


def patch_renderer_dynamic_batching_length() -> None:
    """Monkey patch VeOmni dynamic batching to use renderer diffusion tokens."""
    veomni_dataset.get_length_by_attention_mask_fn = get_renderer_diffusion_length
    veomni_data_loader.get_length_by_attention_mask_fn = get_renderer_diffusion_length

    original_put_item = veomni_data_loader.TextBatchingStrategy.put_item

    def put_item_by_diffusion_length(self, item):
        if item["input_ids"].shape[-1] <= 1:
            print("WARNING: EMPTY STRING.")
            return
        self.buffer.append(item)
        self.buffer._buffer_sample_lens[-1] = get_renderer_diffusion_length(item)
        self.buffer.all_token_cnt = sum(self.buffer._buffer_sample_lens)

    put_item_by_diffusion_length._renderer_diffusion_patch = True
    if not getattr(original_put_item, "_renderer_diffusion_patch", False):
        veomni_data_loader.TextBatchingStrategy.put_item = put_item_by_diffusion_length


def patch_disable_sp_padding() -> None:
    """Disable SP padding/slicing in VeOmni's SequenceParallelCollator.

    Renderer inputs come out of the dataset transform already in their final
    form; the only collator step we want is sequence packing. We still need
    ``add_flash_attention_kwargs_from_position_ids`` to populate
    ``cu_seq_lens_*``/``max_length_*`` on the batch (the model relies on these
    for FlashAttention), so we keep that call but neutralize ``sp_padding`` and
    ``sp_slice`` into pure pass-throughs.
    """
    from veomni.data import data_collator as _dc

    SPColl = _dc.SequenceParallelCollator
    if getattr(SPColl, "_renderer_no_sp_pad_patch", False):
        return

    def sp_padding_passthrough(self, key, feature, dim=-1, pad_value=0, pad_scale=1):
        return feature

    def sp_slice_passthrough(self, key, feature, dim=-1):
        return feature

    def __call__(self, batch):
        if not self.seq_classification:
            labels = batch["labels"][..., 1:].contiguous()
            labels = torch.nn.functional.pad(labels, (0, 1), "constant", _dc.IGNORE_INDEX)
            batch["labels"] = labels
        # Still derive cu_seqlens / max_length / etc. from position_ids so the
        # model's FlashAttention call sites get the kwargs they expect.
        _dc.add_flash_attention_kwargs_from_position_ids(batch)
        # Hand the (un-padded) batch to the model-provided metadata hook, if any.
        if self.metadata_collate_func is not None:
            self.metadata_collate_func(batch, {"pixel_values": 0, "pixel_values_videos": 0})
        return batch

    SPColl.sp_padding = sp_padding_passthrough
    SPColl.sp_slice = sp_slice_passthrough
    SPColl.__call__ = __call__
    SPColl._renderer_no_sp_pad_patch = True


def freeze_module(model: torch.nn.Module, module_name: str, detach: bool = False):
    wrapped = getattr(model, "module", model)
    module = getattr(wrapped, module_name)
    module.eval()
    for param in module.parameters():
        param.requires_grad_(False)
    if detach:
        forward = module.forward

        def no_grad_forward(*args, **kwargs):
            with torch.no_grad():
                return forward(*args, **kwargs)

        module.forward = no_grad_forward


def get_param_groups(model: torch.nn.Module, default_lr: float):
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info_rank0(f"Train parameter: {name}")
            params.append(param)
    return [{"params": params, "lr": default_lr}]


def build_renderer_collator(pad_to_length=False):
    # Mirror the OmniDataCollatorWithPacking spec used in
    # ``tasks/editing/train_qwen2_5_vl_editing.py``:
    #   - packing_features → ``pack_dim=-1``  (seq packing, then unsqueeze(0))
    #   - concat_features  → ``pack_dim=0``   (concat along dim 0)
    # All keys are configured for packing/concat only, with NO SP padding —
    # ``sp_pad_value=None`` makes ``SequenceParallelCollator`` skip padding
    # for that key (see veomni/data/data_collator.py:357). The dataset
    # transform (``process_renderer_sample``) is responsible for producing
    # samples ready to be packed and consumed by the model directly.
    #
    # Note: ``attention_mask`` keeps ``sp_pad_value=1`` only to satisfy the
    # hard ``assert`` in ``MainCollator.__post_init__``; it is neutralized at
    # runtime by ``patch_disable_sp_padding``.
    packing_features = [
        "input_ids",
        "attention_mask",
        "t5_input_lens",
        "labels",
        "position_ids",
        "input_image_mask",
        "output_image_mask",
        "input_video_mask",
        "output_video_mask",
        "vae_latents_mask",
        "vae_seqlen",
        "timesteps",
        "num_tokens",
        "vlm_seqlen",
        "target_lens",
    ]
    concat_features = [
        "image_embeds",
        "video_embeds",
        "input_vae_latents",
        "input_vae_rope",
        "target_velocity",
    ]

    data_collate_info: Dict[str, DataCollateInfo] = {}
    for name in packing_features:
        if name == "attention_mask":
            data_collate_info[name] = DataCollateInfo(-1, False, 1, 1)
        else:
            data_collate_info[name] = DataCollateInfo(-1, False, None, None)
    for name in concat_features:
        data_collate_info[name] = DataCollateInfo(0, False, None, None)

    return MainCollator(data_collate_info=data_collate_info, pad_to_length=pad_to_length, seq_classification=True)


def move_to_device(micro_batch: Dict[str, Any], device: str):
    return {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in micro_batch.items()}


def main():
    register_bernini_renderer_to_veomni()
    patch_renderer_dynamic_batching_length()
    patch_disable_sp_padding()
    if any(arg in ("--help", "-h") for arg in sys.argv[1:]):
        print("Usage: train_bernini_renderer.py <config.yaml> [--nested.arg value ...]")
        print("Key config groups: model, data, train. See configs/bernini_renderer/train_cfg/bernini_renderer_high.yaml")
        return
    args = parse_args(BerniniRendererArguments)
    logger.info_rank0(json.dumps(asdict(args), indent=2))

    get_torch_device().set_device(f"{get_device_type()}:{args.train.local_rank}")
    if not dist.is_initialized():
        # A large default timeout: every rank loads the full Wan2.2 + UMT5 weights
        # from disk before the first collective, which can take a while.
        dist.init_process_group(backend=get_dist_comm_backend(), timeout=timedelta(minutes=60))
    logger.info(f"Process rank: {args.train.global_rank}, world size: {args.train.world_size}")

    init_parallel_state(
        dp_size=args.train.accelerator.dp_size,
        dp_replicate_size=args.train.accelerator.dp_replicate_size,
        dp_shard_size=args.train.accelerator.dp_shard_size,
        tp_size=args.train.accelerator.tp_size,
        pp_size=args.train.accelerator.pp_size,
        cp_size=args.train.accelerator.cp_size,
        ulysses_size=args.train.accelerator.ulysses_size,
        extra_parallel_sizes=args.train.accelerator.extra_parallel_sizes,
        extra_parallel_placement_innermost=args.train.accelerator.extra_parallel_placement_innermost,
        extra_parallel_names=args.train.accelerator.extra_parallel_names,
        dp_mode=args.train.accelerator.fsdp_config.fsdp_mode,
        async_enabled=args.train.accelerator.enable_async,
    )
    from bernini.parallel import init_parallel_state as init_bernini_parallel_state

    init_bernini_parallel_state(ulysses_size=args.train.accelerator.ulysses_size)
    helper.set_seed(args.train.seed, args.train.enable_full_determinism)
    helper.enable_high_precision_for_bf16()
    if args.train.local_rank == 0:
        helper.enable_third_party_logging()
    if args.train.global_rank == 0:
        save_args(args, args.train.checkpoint.output_dir)
    set_checkpoint_debug_enabled(args.train.gradient_checkpointing.debug)

    model = build_foundation_model(
        config_path=args.model.config_path,
        weights_path=args.model.model_path,
        torch_dtype="float32" if args.train.accelerator.fsdp_config.mixed_precision.enable else "bfloat16",
        init_device=args.train.init_device,
        ops_implementation=args.model.ops_implementation,
        config_kwargs=args.model.model_config,
    )
    model_config = model.config

    tokenizer_path = getattr(model_config, "wan22_base", None) or args.model.tokenizer_path
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path, subfolder="tokenizer", padding_side="right", trust_remote_code=True
    )
    rope = WanRotaryPosEmbed(128, (1, 2, 2), 1024, use_src_id_rotary_emb=args.train.use_src_id_rotary_emb)
    vae_config_path = args.model.vae_config_path or os.path.join(model_config.wan22_base, "vae", "config.json")
    with open(vae_config_path) as f:
        vae_config = json.load(f)
    vae_latent_mean = torch.tensor(vae_config["latents_mean"], device="cpu").view(vae_config["z_dim"], 1, 1, 1)
    vae_latent_std = torch.tensor(vae_config["latents_std"], device="cpu").view(vae_config["z_dim"], 1, 1, 1)
    noise_scheduler = NoiseScheduler(**args.data.noise_scheduler_config)
    transform = partial(
        process_renderer_sample,
        tokenizer=tokenizer,
        vae_rope_func=rope,
        vae_latent_mean=vae_latent_mean,
        vae_latent_std=vae_latent_std,
        noise_scheduler=noise_scheduler,
        text_dropout_rate=args.data.text_dropout_rate,
        img_dropout_rate=args.data.img_dropout_rate,
        video_dropout_rate=args.data.video_dropout_rate,
        max_vae_frames=args.data.max_vae_frames,
        lowr_noise_aug_range=tuple(args.data.lowr_noise_aug_range) if args.data.lowr_noise_aug_range else None,
        lowr_noise_add_range=tuple(args.data.lowr_noise_add_range) if args.data.lowr_noise_add_range else None,
        **args.data.mm_configs,
    )

    if args.data.dataloader.num_workers and args.data.dataloader.num_workers > 0:
        dp_rank = getattr(get_parallel_state(), "dp_rank", 0)
        patch_dataloader_worker_seed(args.train.seed + dp_rank * 100000)
        if args.data.dataloader.worker_num_threads is None:
            args.data.dataloader.worker_num_threads = 1

    train_dataset = build_dataset(
        dataset_name=args.data.dataset_name,
        transform=transform,
        seed=args.train.seed,
        **asdict(args.data),
    )
    # VeOmni's build_weighted_multisource_dataset does not pass source_name to
    # each sub-dataset's transform, so process_renderer_sample defaults to the
    # "default" noise schedule. Patch each sub-dataset transform with the correct
    # source_name so task-specific noise parameters (shift, weighting) are used.
    if hasattr(train_dataset, "_datasets") and hasattr(train_dataset, "_source_names"):
        for sub_ds, src_name in zip(train_dataset._datasets, train_dataset._source_names):
            if hasattr(sub_ds, "_transform") and sub_ds._transform is not None:
                sub_ds._transform = partial(transform, source_name=src_name)
    dataset_length = None if not hasattr(train_dataset, "__len__") else len(train_dataset)
    if args.data.datasets_type == "mapping" and dataset_length is not None:
        dataset_length = dataset_length / args.train.accelerator.dp_size
    args.compute_train_steps(dataset_length)

    collate_fn = build_renderer_collator(args.train.pad_to_length)
    train_dataloader = build_dataloader(
        dataloader_type=args.data.dataloader.type,
        dataset=train_dataset,
        micro_batch_size=args.train.micro_batch_size,
        global_batch_size=args.train.global_batch_size,
        dataloader_batch_size=args.train.dataloader_batch_size,
        max_seq_len=args.data.max_seq_len,
        train_steps=args.train_steps,
        bsz_warmup_ratio=args.train.bsz_warmup_ratio,
        bsz_warmup_init_mbtoken=args.train.bsz_warmup_init_mbtoken,
        dyn_bsz=args.train.dyn_bsz,
        dyn_bsz_runtime=args.train.dyn_bsz_runtime,
        dyn_bsz_buffer_size=args.data.dyn_bsz_buffer_size,
        seed=args.train.seed,
        collate_fn=collate_fn,
        **{k: v for k, v in asdict(args.data.dataloader).items() if k != "type"},
    )

    cpu_load_param_name = getattr(model.get_parallel_plan(), "cpu_load_param_name", None) if hasattr(model, "get_parallel_plan") else None
    model = build_parallelize_model(
        model,
        init_device=args.train.init_device,
        weights_path=args.model.model_path,
        enable_reshard_after_forward=args.train.accelerator.fsdp_config.reshard_after_forward,
        mixed_precision=args.train.accelerator.fsdp_config.mixed_precision,
        enable_gradient_checkpointing=args.train.gradient_checkpointing.enable,
        basic_modules=list(set(getattr(model, "_no_split_modules", None) or []) | set(args.model.basic_modules)),
        enable_reentrant=args.train.gradient_checkpointing.enable_reentrant,
        enable_forward_prefetch=args.train.accelerator.fsdp_config.forward_prefetch,
        enable_fsdp_offload=args.train.accelerator.fsdp_config.offload,
        broadcast_model_weights_from_rank0=args.train.broadcast_model_weights_from_rank0,
        cpu_load_param_name=cpu_load_param_name,
        max_load_broadcast_size=args.train.accelerator.fsdp_config.max_load_broadcast_size,
    )
    model.train()
    for module_name in args.train.freeze_modules:
        freeze_module(model, module_name, detach=False)
    for module_name in args.train.detach_modules:
        freeze_module(model, module_name, detach=True)

    optimizer = build_optimizer(
        model,
        lr=args.train.optimizer.lr,
        weight_decay=args.train.optimizer.weight_decay,
        fused=False,
        optimizer_type=args.train.optimizer.type,
        param_groups=get_param_groups(model, args.train.optimizer.lr),
    )
    lr_scheduler = build_lr_scheduler(
        optimizer,
        train_steps=args.train_steps * args.train.num_train_epochs,
        lr=args.train.optimizer.lr,
        lr_min=args.train.optimizer.lr_min,
        lr_decay_style=args.train.optimizer.lr_decay_style,
        lr_decay_ratio=args.train.optimizer.lr_decay_ratio,
        lr_warmup_ratio=args.train.optimizer.lr_warmup_ratio,
        lr_start=args.train.optimizer.lr_start,
    )
    checkpointer = build_checkpointer(
        dist_backend=args.train.accelerator.fsdp_config.fsdp_mode, ckpt_manager=args.train.checkpoint.manager
    )
    if args.train.global_rank == 0:
        save_model_assets(args.train.checkpoint.model_assets_dir, [model_config, tokenizer])

    start_epoch, start_step, global_step = 0, 0, 0
    environ_meter = helper.EnvironMeter(
        config=model_config,
        global_batch_size=args.train.global_batch_size,
        empty_cache_steps=args.train.empty_cache_steps,
        enable_multisource=args.data.enable_multisource,
        dataloader=train_dataloader,
        data_path=args.data.train_path,
    )
    if args.train.checkpoint.load_path:
        state = {"model": model, "optimizer": optimizer, "extra_state": {}}
        checkpointer.load(args.train.checkpoint.load_path, state)
        global_step = state["extra_state"]["global_step"]
        start_epoch = global_step // args.train_steps
        start_step = global_step % args.train_steps
        lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        dist.barrier()

    model_fwd_context, model_bwd_context = build_activation_offloading_context(
        args.train.accelerator.offload_config.enable_activation,
        args.train.gradient_checkpointing.enable,
        args.train.accelerator.offload_config.activation_gpu_limit,
    )

    for epoch in range(start_epoch, args.train.num_train_epochs):
        if hasattr(train_dataloader, "set_epoch"):
            train_dataloader.set_epoch(epoch)
        iterator = iter(train_dataloader)
        pbar = trange(args.train_steps, initial=start_step, disable=args.train.local_rank != 0)
        for _ in range(start_step, args.train_steps):
            global_step += 1
            micro_batches: List[Dict[str, Any]] = next(iterator)
            total_loss = 0.0
            total_diff_loss = 0.0
            task_loss = defaultdict(list)
            diff_length = torch.tensor(0, dtype=torch.int32, device=get_device_type())
            for micro_batch in micro_batches:
                target_lens_for_count = micro_batch["target_lens"]
                diff_length += target_lens_for_count[target_lens_for_count > 0].numel()
            diff_length = all_reduce(diff_length, op="sum", group=get_parallel_state().fsdp_group)
            synchronize()
            start_time = time.time()
            if hasattr(model, "ema"):
                model.ema.update(model)

            for micro_step, micro_batch in enumerate(micro_batches):
                if (
                    args.train.accelerator.fsdp_config.fsdp_mode == "fsdp2"
                    and not args.train.accelerator.fsdp_config.reshard_after_backward
                    and len(micro_batches) > 1
                ):
                    model.set_reshard_after_backward(micro_step == len(micro_batches) - 1)
                source_name = micro_batch.pop("source_name", None)
                environ_meter.add(micro_batch)
                micro_batch.pop("ds_idx", None)
                micro_batch.pop("cur_token_num", None)
                task_names = [x.split("$")[0].lower() for x in source_name] if source_name else []
                task_names = [x for x in task_names if not x.startswith("und")]
                micro_batch = move_to_device(micro_batch, get_device_type())
                with model_fwd_context:
                    output = model(**micro_batch, use_cache=False)
                    diff_loss = output.diff_loss
                    sample_losses = diff_loss.mean(dim=-1).squeeze(0)
                    target_lens = micro_batch["target_lens"].squeeze(0)
                    target_lens = target_lens[target_lens > 0]
                    sample_losses = [s.mean() for s in torch.split(sample_losses, target_lens.tolist())]
                    for sample_loss, task_name in zip(sample_losses, task_names):
                        task_loss[task_name].append(sample_loss.item())
                    loss = torch.stack(sample_losses).mean() * target_lens.numel() / diff_length * get_parallel_state().fsdp_size
                with model_bwd_context:
                    loss.backward()
                total_loss += loss.item()
                total_diff_loss += loss.item()

            grad_norm = veomni_clip_grad_norm(model, args.train.optimizer.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            if hasattr(grad_norm, "full_tensor"):
                grad_norm = grad_norm.full_tensor().item()
            total_loss, total_diff_loss, grad_norm = all_reduce(
                (total_loss, total_diff_loss, grad_norm), group=get_parallel_state().fsdp_group
            )
            log_losses = {"diff_loss": total_diff_loss}
            for task_name in args.data.task_lists:
                gathered = [task_loss[task_name]]
                gathered_obj = [None for _ in range(dist.get_world_size())]
                dist.all_gather_object(gathered_obj, gathered)
                flat = [v for rank_list in gathered_obj for inner in rank_list for v in inner]
                if flat:
                    log_losses[f"{task_name}_diff_loss"] = sum(flat) / len(flat)
            train_metrics = environ_meter.step(delta_time=time.time() - start_time, global_step=global_step)
            lr = max(lr_scheduler.get_last_lr())
            postfix_info = {
                "loss": f"{total_loss:.4f}",
                "grad_norm": f"{grad_norm:.4f}",
                "lr": f"{lr:.2e}",
            }
            postfix_info.update({k: f"{v:.4f}" for k, v in log_losses.items()})
            postfix_str = ", ".join([f"{k}: {v}" for k, v in postfix_info.items()])
            pbar.set_postfix_str(postfix_str, refresh=False)
            pbar.update()

            if args.train.global_rank == 0 and args.train.wandb.enable:
                import wandb

                if global_step == 1:
                    wandb.init(
                        project=args.train.wandb.project,
                        name=args.train.wandb.name,
                        id=args.train.wandb.id,
                        resume="allow" if args.train.wandb.id else None,
                        config=asdict(args),
                    )
                train_metrics.update({f"training/{k}": v for k, v in log_losses.items()})
                train_metrics.update({"training/loss": total_loss, "training/grad_norm": grad_norm, "training/lr": max(lr_scheduler.get_last_lr())})
                wandb.log(train_metrics, step=global_step)

            if args.train.checkpoint.save_steps and global_step % args.train.checkpoint.save_steps == 0:
                save_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{global_step}")
                state = {
                    "model": model,
                    "optimizer": optimizer,
                    "extra_state": {
                        "global_step": global_step,
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "train_dataloader": train_dataloader.state_dict(),
                        "environ_meter": environ_meter.state_dict(),
                        "torch_rng_state": torch.get_rng_state(),
                    },
                }
                checkpointer.save(save_path, state, save_async=args.train.checkpoint.save_async)
                dist.barrier()
        pbar.close()
        start_step = 0
    synchronize()
    if args.train.global_rank == 0 and args.train.wandb.enable:
        import wandb

        wandb.finish()


if __name__ == "__main__":
    main()
