# Bernini-R — renderer only

[← Back to main README](../README.md)

This document describes the full training workflow for the Bernini Renderer (Bernini-R),
which can be broken down into four stages:
**environment setup → data preprocessing → launching training → ckpt aggregation after training**.

---

## 1. Environment Setup

The project declares its dependencies in `pyproject.toml`. We recommend using
[uv](https://docs.astral.sh/uv/) to manage the Python environment. The `[tool.uv.index]` /
`[tool.uv.sources]` sections in `pyproject.toml` already route `torch` / `torchvision` to
`https://download.pytorch.org/whl/cu126` and pin `veomni` to its official git tag, so you
do not need to specify `--extra-index-url` manually.

### One-shot installation

```bash
# Install uv (if not yet installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create .venv at the repository root and install core dependencies
uv sync
uv sync --extra all
```

`uv sync` automatically creates and uses `.venv`. You can either activate it with
`source .venv/bin/activate`, or run commands inside it via `uv run <cmd>`.

### Installing flash-attn

`flash-attn` requires a pre-installed torch at build time and must use
`--no-build-isolation`, so it is not listed as a regular dependency in `pyproject.toml`.
Install it separately after `uv sync` completes:

```bash
uv pip install --no-build-isolation "flash-attn==2.8.3"
```

---

## 2. Data Preprocessing

Use `tools/preprocess_data.py` to convert raw parquet data into the format expected by
training, pre-extracting the Qwen2.5-VL ViT features and the Wan VAE latents so that we
do not have to run those models during training (which would slow training down).

The script reads every `.parquet` file under the input directory and, for each sample:

- if an `images` column is present, extracts the ViT pixel embeddings and the VAE latent
  for each image;
- if a `videos` column is present, samples frames at `vit_fps` / `vae_fps` respectively
  and writes out the ViT/VAE features;
- writes the new columns back into a parquet file under `output_dir` (local path).

### Input data format

The input is one or more `.parquet` files, where each row represents one training sample.
Required / supported columns per row:

| Field | Type | Description |
| ---- | ---- | ---- |
| `inputs` | `string` (JSON) | A JSON-serialized list of conversation messages used during training. Each message must contain a `type` field; supported `type` values are `text` / `cot_text` / `image` / `image_gen` / `video` / `frame_gen` / `video_gen` / `special_token`. The `*_gen` types denote items that participate in the loss (i.e. generation targets). `text`-typed messages additionally need a `text` field; `*_gen` messages default to `has_loss=1`, while plain `image` / `video` default to `has_loss=0`. |
| `images` | `list` | (Optional) Provided when the sample contains image inputs and/or image ground truths. Each element must be a reference loadable by `veomni.data.multimodal.image_utils.fetch_images` (local path, HTTP/HTTPS URL, `hdfs://` path, or base64 string). The order of images in the list must match the order of `image` / `image_gen` messages in `inputs`. |
| `videos` | `list[dict]` | (Optional) Provided when the sample contains video inputs and/or video ground truths. Each element is a dict with the following fields:<br>· `video_path` *(required)*: video file path readable by `PathVideoReader`;<br>· `duration` *(optional)*: clip duration in seconds; defaults to reading the whole video;<br>· `crop_method` *(optional)*: cropping method along the time axis; defaults to `left`.<br>The order of videos in the list must match the order of `video` / `frame_gen` / `video_gen` messages in `inputs`. |

Notes:

- At least one of `images` / `videos` must exist for the corresponding feature
  extraction to be triggered. Samples that fail to process are dropped and will not
  appear in the output parquet.
- Video processing is forced to `batch_size=1`, so there is no hard limit on the number
  of `videos` per row, but each video is bounded by `vit_fps` / `vae_fps` and
  `*_max_n_frames`.

### Output data format

The output parquet preserves all original columns (e.g. `inputs`, `images`, `videos`,
…) and additionally appends the following columns when applicable:

| Field | Source | Description |
| ---- | ---- | ---- |
| `image_embeds` | `images` | `list[bytes]`; each element is a `torch.save`-serialized Qwen2.5-VL ViT output tensor. |
| `image_grid_thw` | `images` | `list[list[int]]`; the `(t, h, w)` patch grid for each image, aligned with `image_embeds`. |
| `image_vae_latents` | `images` | `list[bytes]`; the per-image Wan VAE latent distribution parameters (mean/logvar), `torch.save`-serialized. Only produced when `--only_vit` is not set. |
| `video_embeds` | `videos` | `list[bytes]`; per-video ViT-sampled embeddings, serialized as above. |
| `video_grid_thw` | `videos` | `list[list[int]]`; `(t, h, w)` aligned with `video_embeds`. |
| `video_vae_latents` | `videos` | `list[bytes]`; per-video VAE-sampled latent distribution parameters, serialized as above. Only produced when `--only_vit` is not set. |

The output parquet is also chunked into row groups according to `--row_group_size` (in
MB), and each file is shuffled once (`random_state=42`).

### Basic usage

```bash
uv run python tools/preprocess_data.py \
  <input_dir> <output_dir> \
  --vlm_config /path/to/Qwen2.5-VL-7B-Instruct \
  --vae_config /path/to/Wan2.2-T2V-A14B-Diffusers/vae \
  --num_workers 16
```

### Key arguments

| Argument | Description | Default |
| ---- | ---- | ---- |
| `input_dir` | Input data directory or a single parquet file | — |
| `output_dir` | Output directory (local path) | — |
| `--vlm_config` | Qwen2.5-VL model path used for ViT feature extraction | Qwen2.5-VL-7B-Instruct |
| `--vae_config` | Wan VAE path used for latent extraction | Wan2.2-T2V-A14B-Diffusers/vae |
| `--vit_min_pixels` / `--vit_max_pixels` | Min/max pixel count of the ViT input | 3136 / 50176 |
| `--vae_max_pixels` | Upper bound of pixels for the VAE input | 230400 |
| `--vae_max_image_size` / `--vae_min_image_size` | Max/min side length of the VAE image | 480 / 240 |
| `--vit_fps` / `--vae_fps` | Video sampling FPS | 2 / 16 |
| `--vit_max_n_frames` / `--vae_max_n_frames` | Max ViT/VAE frame count | 12 / 96 |
| `--only_vit` | Only extract ViT features, skip VAE | False |
| `--num_workers` | Number of DataLoader workers | 16 |
| `--batch_size` | Forced to 1 for video processing | 1 |
| `--row_group_size` | Target parquet row-group size in MB | 64 |

---

## 3. Launching Training

### Entry script

`scripts/train_bernini_renderer.sh` invokes the training entrypoint
`tasks/bernini_renderer/train_bernini_renderer.py` via `torchrun`. By default it uses
`configs/bernini_renderer/train_cfg/bernini_renderer_high.yaml` and forwards extra
arguments (`"$@"`) to the VeOmni argument parser.

```bash
bash scripts/train_bernini_renderer.sh
# Override config items, for example:
bash scripts/train_bernini_renderer.sh --train.max_steps 1000
# Switch to the low-noise expert:
bash scripts/train_bernini_renderer.sh \
  configs/bernini_renderer/train_cfg/bernini_renderer_low.yaml
```

The script automatically infers the distributed topology from environment variables:

- `NNODES` / `NPROC_PER_NODE` / `NODE_RANK` / `MASTER_ADDR` / `MASTER_PORT`; if not set
  it falls back to a single host with the visible GPU count;
- It sets `NCCL_P2P_LEVEL=NVL` and similar parameters, and clears network-plugin
  related environment variables that could interfere with NCCL initialization;
- Before launching, it prepends `VeOmni/` and the project root to `PYTHONPATH`.

If you manage the environment with `uv`, you can prepend `uv run` to the script or first
run `source .venv/bin/activate`.

### Config files

The training configs live under `configs/bernini_renderer/`:

- `train_cfg/bernini_renderer_high.yaml`: High-noise expert
  (`skip_transformer_2: true`, `noise_tmin/tmax = 0.875/1.0`,
  `output_dir: bernini_renderer_train`);
- `train_cfg/bernini_renderer_low.yaml`: Low-noise expert
  (`skip_transformer_1: true`, `noise_tmin/tmax = 0.0/0.875`,
  `output_dir: bernini_renderer_train_low`);
- `train_cfg/bernini_renderer_test.yaml`: minimal config used for smoke testing;
- `data_cfg/example_weighted_multisource.yaml`: example multi-source weighted data
  config. The `name` field uses the `<task>$<dataset>` convention to identify the task
  type (`t2i`, `t2v`, `i2i`, `r2i`, `r2v`, `v2v`, `i2v`, `vi2v`, `vr2v`, `vrc2v`,
  `mv2v`).

Each task type has its own `shift_config` and `weighting_scheme_config` inside
`noise_scheduler_config`. To plug real datasets in, replace the `sources` / `names` /
`schedule.weights` fields in `example_weighted_multisource.yaml` with your data,
keyed by task type.

### Training entrypoint highlights

`tasks/bernini_renderer/train_bernini_renderer.py` is built on the VeOmni framework:

- `register_bernini_renderer_to_veomni` registers the Bernini model and dataset into
  VeOmni;
- `process_renderer_sample` + `NoiseScheduler` jointly handle training-sample assembly
  and flow-matching noise scheduling;
- It uses FSDP2, Ulysses sequence parallelism (`ulysses_size: 8`), activation
  recomputation, and dynamic batch size (`dyn_bsz: true`);
- The checkpoint manager defaults to `dcp` and saves asynchronously every
  `save_steps: 1000`; `load_path: auto` means automatically resuming from the latest
  ckpt under `output_dir`;
- wandb is enabled by default (`project: bernini_renderer`, run names `high` / `low`).

---

## 4. Aggregating ckpts after training

VeOmni emits checkpoints in PyTorch DCP (Distributed Checkpoint) format by default.
`tools/merge_dcp_to_hf_pt.py` converts a DCP checkpoint into HuggingFace-compatible
`safetensors` weights, so that inference scripts can load them directly.

### Basic usage

```bash
uv run python tools/merge_dcp_to_hf_pt.py \
  --load-dir bernini_renderer_train/checkpoints/global_step_xxxx \
  --save-dir /path/to/hf_output \
  --model-assets-dir configs/bernini_renderer_wan22 \
  --shard-size 5000000000
```

### Arguments

| Argument | Description | Default |
| ---- | ---- | ---- |
| `--load-dir` | DCP checkpoint directory (required) | — |
| `--save-dir` | HuggingFace output directory | `<load-dir>/hf_ckpt` |
| `--model-assets-dir` | Asset directory whose `AutoConfig`/`AutoProcessor` will be saved alongside | None |
| `--shard-size` | Maximum number of bytes per shard | `5_000_000_000` (~5GB) |

Once aggregation finishes, the resulting directory can be loaded as
`high_noise_ckpt` / `low_noise_ckpt`. You can further convert it into the `diffusers`
format with `tools/convert_hf_to_diffusers.py`.
