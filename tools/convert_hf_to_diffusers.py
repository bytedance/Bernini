#!/usr/bin/env python3
"""Convert Bernini Renderer checkpoints into a Wan2.2-Diffusers-format directory.

The Bernini high/low checkpoints store the transformer weights under the keys
``ema.diff_dec.transformer.<bare>`` / ``ema.diff_dec.transformer_2.<bare>``.
After stripping those prefixes the bare keys are identical to the Wan2.2
diffusers ``transformer/`` keys, so we can re-shard them into a standard
diffusers layout (``diffusion_pytorch_model-*.safetensors`` +
``diffusion_pytorch_model.safetensors.index.json`` + ``config.json``). This lets
``WanTransformer3DModel.from_pretrained`` load the Bernini weights directly in a
single pass instead of loading the Wan base and then overwriting it.
"""

import argparse
import json
import os
import shutil
from collections import defaultdict

from safetensors import safe_open
from safetensors.torch import save_file

INDEX_NAME = "diffusion_pytorch_model.safetensors.index.json"


def build_weight_map(ckpt_dir: str) -> dict:
    index_files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith("safetensors.index.json"))
    if not index_files:
        raise FileNotFoundError(f"no *.safetensors.index.json in {ckpt_dir}")
    with open(os.path.join(ckpt_dir, index_files[0])) as f:
        return json.load(f)["weight_map"]


def convert_transformer(src_dir: str, prefix: str, out_dir: str, config_src: str):
    os.makedirs(out_dir, exist_ok=True)
    weight_map = build_weight_map(src_dir)

    # bare_target -> source_key, selecting only this prefix (prefer ema copies).
    chosen = {}
    for key in weight_map:
        is_ema = key.startswith("ema.")
        base = key[len("ema.") :] if is_ema else key
        if not base.startswith(prefix):
            continue
        target = base[len(prefix) :]
        if target not in chosen or (is_ema and not chosen[target].startswith("ema.")):
            chosen[target] = key
    if not chosen:
        raise ValueError(f"no keys matching prefix '{prefix}' in {src_dir}")
    print(f"[{out_dir}] {len(chosen)} transformer tensors (prefix '{prefix}')")

    # Group target keys by their source shard so each source shard is opened once.
    by_shard = defaultdict(list)
    for target, source in chosen.items():
        by_shard[weight_map[source]].append((source, target))

    src_shards = sorted(by_shard)
    n = len(src_shards)
    new_weight_map = {}
    total_size = 0
    for i, shard_fn in enumerate(src_shards, start=1):
        out_fn = f"diffusion_pytorch_model-{i:05d}-of-{n:05d}.safetensors"
        tensors = {}
        with safe_open(os.path.join(src_dir, shard_fn), framework="pt", device="cpu") as f:
            for source, target in by_shard[shard_fn]:
                t = f.get_tensor(source)
                tensors[target] = t
                new_weight_map[target] = out_fn
                total_size += t.numel() * t.element_size()
        save_file(tensors, os.path.join(out_dir, out_fn), metadata={"format": "pt"})
        print(f"  wrote {out_fn}: {len(tensors)} tensors")

    index = {"metadata": {"total_size": total_size}, "weight_map": new_weight_map}
    with open(os.path.join(out_dir, INDEX_NAME), "w") as f:
        json.dump(index, f, indent=2)
    shutil.copyfile(config_src, os.path.join(out_dir, "config.json"))
    print(f"  index + config.json written to {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--high_dir", required=True)
    p.add_argument("--low_dir", required=True)
    p.add_argument("--wan_base", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    config_src = os.path.join(args.wan_base, "transformer", "config.json")
    convert_transformer(args.high_dir, "diff_dec.transformer.",
                        os.path.join(args.out, "transformer"), config_src)
    convert_transformer(args.low_dir, "diff_dec.transformer_2.",
                        os.path.join(args.out, "transformer_2"), config_src)
    print("done")


if __name__ == "__main__":
    main()
