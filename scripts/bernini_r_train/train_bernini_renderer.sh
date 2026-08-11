#!/usr/bin/env bash
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

set -euo pipefail
unset NCCL_NET_PLUGIN NCCL_NET_GCP_FASTRAK NCCL_NET_PLUGIN_PATH
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export MODELING_BACKEND=hf

# Defaults for local debug. Override these env vars for multi-node runs.
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}
MASTER_PORT=${MASTER_PORT%%,*}

python - <<'PY'
import torch
import torch.distributed.fsdp as fsdp
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"device_count={torch.cuda.device_count()}")
print(f"CPUOffloadPolicy={hasattr(fsdp, 'CPUOffloadPolicy')}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in this shell; please run from a GPU-visible environment.")
if not hasattr(fsdp, "CPUOffloadPolicy"):
    raise SystemExit("This torch version is incompatible with VeOmni v0.1.11; please use torch >= 2.7.1, preferably the version required by VeOmni.")
PY

torchrun \
  --nnodes="$NNODES" \
  --nproc-per-node="$NPROC_PER_NODE" \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" \
  --master-port="$MASTER_PORT" \
  tasks/bernini_renderer/train_bernini_renderer.py \
  configs/bernini_renderer_train/train_cfg/bernini_renderer_high.yaml \
  "$@"
