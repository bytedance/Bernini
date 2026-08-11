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

import argparse
import glob
import io
import os
import time
from random import shuffle

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from diffusers.models import AutoencoderKLWan
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from vae_transform import VAEImageTransform
from veomni.data.multimodal.image_utils import fetch_images

from video_utils import PathVideoReader, smart_video_nframes

parser = argparse.ArgumentParser()
parser.add_argument(
    'input_dir', type=str
)
parser.add_argument(
    'output_dir', type=str
)
parser.add_argument(
    '--vlm_config', default='Qwen/Qwen2.5-VL-7B-Instruct', type=str
)
parser.add_argument(
    '--vae_config', default='Wan-AI/Wan2.2-T2V-A14B-Diffusers/vae', type=str
)
parser.add_argument(
    '--vit_min_pixels', default=3136, type=int
)
parser.add_argument(
    '--vit_max_pixels', default=50176, type=int
)
parser.add_argument(
    '--vae_max_pixels', default=230400, type=int
)
parser.add_argument(
    '--vae_max_image_size', default=480, type=int
)
parser.add_argument(
    '--vae_min_image_size', default=240, type=int
)
parser.add_argument(
    '--only_vit', default=False, action='store_true'
)
parser.add_argument(
    '--vit_fps', default=2, type=int
)
parser.add_argument(
    '--vae_fps', default=16, type=int
)
parser.add_argument(
    '--vit_max_n_frames', default=12, type=int
)
parser.add_argument(
    '--vae_max_n_frames', default=96, type=int
)
parser.add_argument(
    '--num_workers', type=int, default=16, help='Number of worker processes for data loading.'
)
parser.add_argument(
    '--batch_size', type=int, default=1, help='Batch size. Must be 1 for video processing.'
)
parser.add_argument(
    '--row_group_size', type=int, default=64, help='Row group size for Parquet output.'
)

# 全局变量，用于在Dataset中访问
args = None
vit_processor = None
vae_transform = None


def tensor_to_bytes(x):
    buffer = io.BytesIO()
    torch.save(x, buffer)
    buffer.seek(0)
    return buffer.getvalue()


def get_image_vit_features(vlm_model, pixel_values, image_grid_thw):
    pixel_values = pixel_values.type(vlm_model.dtype).to(vlm_model.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        image_embeds = vlm_model.visual(pixel_values, grid_thw=image_grid_thw)
    split_sizes = (image_grid_thw.prod(-1) //
                   vlm_model.visual.spatial_merge_size**2).tolist()
    image_embeds = torch.split(image_embeds, split_sizes)
    return image_embeds


def get_image_vae_features(vae_model, x):
    x = x.unsqueeze(0).unsqueeze(2).type(vae_model.dtype).to(vae_model.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        latent_dist = vae_model.encode(x).latent_dist
    latent_dist = latent_dist.parameters.detach().cpu()
    return tensor_to_bytes(latent_dist)


def get_video_vae_features(vae_model, x):
    x = x.unsqueeze(0).type(vae_model.dtype).to(vae_model.device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        latent_dist = vae_model.encode(x).latent_dist
    latent_dist = latent_dist.parameters.detach().cpu()
    return tensor_to_bytes(latent_dist)


class FeatureExtractionDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        result = {'index': idx, 'valid': True}
        video_path = "UNKNOWN_PATH"

        if 'images' in self.df.columns:
            try:
                images = fetch_images(row['images'])
                image_inputs = vit_processor.image_processor(
                    images=images,
                    return_tensors="pt",
                    min_pixels=args.vit_min_pixels,
                    max_pixels=args.vit_max_pixels
                )
                result['image_pixel_values'] = image_inputs['pixel_values']
                result['image_grid_thw'] = image_inputs['image_grid_thw']

                if not args.only_vit:
                    image_tensors = [vae_transform(img) for img in images]
                    result['image_vae_tensors'] = image_tensors
            except Exception as e:
                print(
                    f"[ERROR] Processing images at index {idx}: {e}. This row will be dropped.")
                result['valid'] = False
                return result

        if 'videos' in self.df.columns:
            result.update(
                {
                    "video_grid_thw": [],
                    "video_pixel_values": [],
                    "video_vae_tensors": [],
                }
            )
            for video_meta in row['videos']:
                try:
                    video_path = video_meta.get('video_path')
                    duration = video_meta.get('duration', None)
                    crop_method = video_meta.get('crop_method', 'left')
                    if video_path is None:
                        raise ValueError(
                            f"'video_path' is None at index {idx}.")

                    video_reader = PathVideoReader(
                        video_path, duration=duration, crop_method=crop_method)

                    vit_idx = smart_video_nframes(
                        total_frames=video_reader.length,
                        video_fps=video_reader.fps,
                        fps=args.vit_fps,
                        frame_factor=2,
                        max_frames=args.vit_max_n_frames,
                        add_one=False
                    )
                    video_for_vit = video_reader.sample(vit_idx)

                    video_inputs = vit_processor.video_processor(
                        videos=video_for_vit,
                        return_tensors="pt",
                        size={
                            'shortest_edge': args.vit_min_pixels,
                            'longest_edge': args.vit_max_pixels
                        },
                    )

                    result['video_pixel_values'].append(
                        video_inputs['pixel_values_videos'])
                    result['video_grid_thw'].append(
                        video_inputs['video_grid_thw'])

                    if not args.only_vit:
                        vae_idx = smart_video_nframes(
                            total_frames=video_reader.length,
                            video_fps=video_reader.fps,
                            fps=args.vae_fps,
                            frame_factor=16,
                            max_frames=args.vae_max_n_frames,
                            add_one=True
                        )
                        video_for_vae = video_reader.sample(vae_idx)
                        video_tensor = torch.stack(
                            [vae_transform(frame) for frame in video_for_vae], dim=1)
                        result['video_vae_tensors'].append(video_tensor)

                except Exception as e:
                    print(
                        f"\n[FATAL ERROR] Processing video at index {idx}, path: {video_path}")
                    print(
                        f"[FATAL ERROR] Error type: {type(e).__name__}, Message: {e}")
                    print("[FATAL ERROR] This row will be dropped.\n")
                    result['valid'] = False
                    return result

        return result


def write_table(table, output_path):
    try:
        table = pa.Table.from_pandas(table)
        target_bytes = args.row_group_size * 1024 * 1024  # 64MB
        avg_row_bytes = table.nbytes / table.num_rows if table.num_rows > 0 else target_bytes
        row_group_size = max(1, int(target_bytes / avg_row_bytes))
        print(
            f"Row group size: {row_group_size}; Avg: {avg_row_bytes/1024/1024:.2f} MB.")
        pq.write_table(
            table,
            output_path,
            write_batch_size=1,
            write_page_index=True,
            row_group_size=row_group_size
        )
    except Exception as e:
        print(f"[ERROR] Writing table to {output_path}: {e}")
        raise e


def main():
    global args, vit_processor, vae_transform

    args = parser.parse_args()

    if args.batch_size != 1:
        print(f"WARNING: Batch size must be 1 for video processing. Resetting batch_size to 1.")
        args.batch_size = 1

    print("Loading models...")
    vlm_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_config)
    vlm_model.eval()
    vlm_model = vlm_model.cuda()
    print('finish creating vlm model')

    vit_processor = AutoProcessor.from_pretrained(
        args.vlm_config, padding_side="right", trust_remote_code=True)
    print('finish creating vit processor')

    vae_model = AutoencoderKLWan.from_pretrained(
        args.vae_config, torch_dtype=torch.float32)
    vae_model.eval()
    vae_model = vae_model.cuda()
    vae_transform = VAEImageTransform(
        max_image_size=args.vae_max_image_size,
        min_image_size=args.vae_min_image_size,
        image_stride=8*2,
        max_pixels=args.vae_max_pixels,
    )
    print("Models loaded.")

    if os.path.isdir(args.input_dir):
        files = glob.glob(f'{args.input_dir}/*.parquet')
    else:
        files = [args.input_dir]
    os.makedirs(args.output_dir, exist_ok=True)

    shuffle(files)
    print('start to process files')

    for file in files:
        out_path = os.path.join(args.output_dir, os.path.basename(file))
        if os.path.exists(out_path):
            print(f'Skip {out_path}')
            continue

        print(f'\nProcessing file: {file}')
        t0 = time.time()
        df = pd.read_parquet(file)
        t1 = time.time()
        print(
            f'Finished reading file {file} in {t1-t0:.2f} s. Number of rows: {len(df)}')

        dataset = FeatureExtractionDataset(df)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
        )

        processed_rows = [None] * len(df)

        t_process_start = time.time()
        for i, batch in enumerate(dataloader):
            idx = batch['index'].item()

            if not batch['valid'].item():
                processed_rows[idx] = None
                continue

            row_data = {}

            if 'image_pixel_values' in batch and 'image_grid_thw' in batch:
                pixel_values = batch['image_pixel_values'][0]
                image_grid_thw = batch['image_grid_thw'][0]

                image_embeds = get_image_vit_features(
                    vlm_model, pixel_values, image_grid_thw)
                image_embeds = [tensor_to_bytes(image_embed.detach().cpu())
                                for image_embed in image_embeds]
                row_data['image_embeds'] = image_embeds
                row_data['image_grid_thw'] = image_grid_thw.numpy().tolist()

                if not args.only_vit and 'image_vae_tensors' in batch:
                    image_vae_tensors = batch['image_vae_tensors']
                    image_vae_latents = [get_image_vae_features(
                        vae_model, img_tensor.squeeze(0)) for img_tensor in image_vae_tensors]
                    row_data['image_vae_latents'] = image_vae_latents

            if 'video_pixel_values' in batch and 'video_grid_thw' in batch:
                row_data.update(
                    {
                        "video_embeds": [],
                        "video_grid_thw": [],
                        "video_vae_latents": [],
                    }
                )
                for pixel_values, grid_thw in zip(batch['video_pixel_values'], batch['video_grid_thw']):
                    pixel_values = pixel_values.squeeze(0)
                    grid_thw = grid_thw.squeeze(0)
                    grid_thw = torch.tensor(grid_thw)
                    video_embeds = get_image_vit_features(
                        vlm_model, pixel_values, grid_thw)
                    row_data['video_embeds'].extend(
                        [tensor_to_bytes(video_embed.detach().cpu()) for video_embed in video_embeds])
                    row_data['video_grid_thw'].extend(
                        grid_thw.numpy().tolist())

                if not args.only_vit and 'video_vae_tensors' in batch:
                    for video_vae_tensor in batch['video_vae_tensors']:
                        video_vae_latent = get_video_vae_features(
                            vae_model, video_vae_tensor.squeeze(0))
                        row_data['video_vae_latents'].append(
                            video_vae_latent)

            processed_rows[idx] = row_data

            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{len(dataloader)} samples.")

        t_process_end = time.time()
        print(
            f"GPU processing finished in {t_process_end - t_process_start:.2f} s.")

        print("Reconstructing DataFrame...")
        valid_mask = [row is not None for row in processed_rows]
        valid_processed_rows = [
            row for row in processed_rows if row is not None]
        valid_df = df[valid_mask].copy().reset_index(drop=True)

        if not valid_processed_rows:
            print("No valid rows processed. Saving empty DataFrame.")
        else:
            new_columns_df = pd.DataFrame(valid_processed_rows)
            valid_df = pd.concat([valid_df, new_columns_df], axis=1)

        valid_df = valid_df.sample(
            frac=1, random_state=42).reset_index(drop=True)
        write_table(valid_df, out_path)

        t2 = time.time()
        print(
            f"Total time for {file}: {t2 - t0:.2f} s. Processed {len(valid_df)}/{len(df)} samples.")


if __name__ == '__main__':
    main()
