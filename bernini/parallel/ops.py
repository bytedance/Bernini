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

"""Sequence-parallel tensor ops.

Every op is a no-op when Ulysses is disabled (single-GPU) and otherwise
delegates to Open-VeOmni, imported lazily so single-GPU inference carries no
dependency on it.
"""

import torch
import torch.nn.functional as F

from .state import get_parallel_state


def gather_seq_scatter_heads(x, seq_dim, head_dim, unpadded_dim_size=0):
    """All-to-all: gather the sequence dim, scatter the head dim."""
    if not get_parallel_state().ulysses_enabled:
        return x
    from veomni.distributed.sequence_parallel import gather_seq_scatter_heads as _f

    return _f(x, seq_dim=seq_dim, head_dim=head_dim, unpadded_dim_size=unpadded_dim_size)


def gather_heads_scatter_seq(x, head_dim, seq_dim):
    """All-to-all: gather the head dim, scatter the sequence dim."""
    if not get_parallel_state().ulysses_enabled:
        return x
    from veomni.distributed.sequence_parallel import gather_heads_scatter_seq as _f

    return _f(x, head_dim=head_dim, seq_dim=seq_dim)


def slice_input_tensor(x, dim):
    """Keep only this rank's slice of `x` along `dim`."""
    if not get_parallel_state().ulysses_enabled:
        return x
    from veomni.distributed.sequence_parallel import slice_input_tensor as _f

    return _f(x, dim=dim)


def slice_input_tensor_scale_grad(x, dim):
    """`slice_input_tensor` variant used inside autograd-tracked code paths."""
    if not get_parallel_state().ulysses_enabled:
        return x
    from veomni.distributed.sequence_parallel import slice_input_tensor_scale_grad as _f

    return _f(x, dim=dim)


def gather_outputs(x, gather_dim, padding_dim=None, unpad_dim_size=None, group=None):
    """Gather a sequence-sharded tensor back to its full length."""
    if not get_parallel_state().ulysses_enabled:
        return x
    from veomni.distributed.sequence_parallel import gather_outputs as _f

    return _f(x, gather_dim=gather_dim, padding_dim=padding_dim, unpad_dim_size=unpad_dim_size)


def padding_tensor_for_seqeunce_parallel(x, dim):
    """Pad `x` along `dim` so its size is divisible by the Ulysses world size."""
    if not get_parallel_state().ulysses_enabled:
        return x
    from veomni.distributed.sequence_parallel.utils import (
        padding_tensor_for_seqeunce_parallel as _f,
    )

    return _f(x, dim=dim)


def pad_tensor(x, dim, padding_size, padding_value=0):
    """Append `padding_size` entries along `dim` (F.pad based, low peak memory)."""
    pad_config = [0, 0] * x.ndim
    pad_config[(x.ndim - 1 - dim) * 2 + 1] = padding_size
    return F.pad(x, pad_config, value=padding_value)


def unpad_tensor(x, dim, padding_size):
    """Inverse of `pad_tensor`: drop the last `padding_size` entries along `dim`."""
    slc = [slice(None)] * x.ndim
    slc[dim] = slice(0, -padding_size)
    return x[slc]


def gen_cu_seqlens_for_cross_attn(q_len, batch_seqlens_q, batch_seqlens_k, device="cpu"):
    """cu_seqlens / max_seqlens for cross-attention under Ulysses sequence parallel.

    Each rank holds a contiguous ``q_len / sp_world`` slice of the query
    sequence; this maps the per-sample query/key lengths onto that local slice.
    """
    ps = get_parallel_state()
    sp_world = ps.ulysses_size
    rank = ps.ulysses_rank
    rank_q_len = (q_len + ((sp_world - (q_len % sp_world)) % sp_world)) // sp_world
    start = rank_q_len * rank
    end = min(q_len, start + rank_q_len)
    offset = 0
    cu_seqlens_q = [start]
    index = []
    max_seqlen_q = -1
    max_seqlen_k = -1
    for i, length in enumerate(batch_seqlens_q):
        offset = min(offset + length, end)
        if offset <= start:
            continue
        cu_seqlens_q.append(offset)
        index.append(i)
        max_seqlen_q = max(max_seqlen_q, cu_seqlens_q[-1] - cu_seqlens_q[-2])
        max_seqlen_k = max(max_seqlen_k, batch_seqlens_k[i])
        if offset >= end:
            break
    cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device=device)
    max_seqlen_q = torch.tensor(max_seqlen_q, device=device)
    cu_seqlens_q -= start
    cu_seqlens_k = torch.zeros(len(batch_seqlens_k) + 1, dtype=torch.int32, device=device)
    cu_seqlens_k[1:] = torch.tensor(batch_seqlens_k, dtype=torch.int32, device=device).cumsum(dim=0)
    cu_seqlens_k = cu_seqlens_k[index[0] : index[-1] + 2]
    max_seqlen_k = torch.tensor(max_seqlen_k, device=device)
    return cu_seqlens_k, cu_seqlens_q, max_seqlen_k, max_seqlen_q, rank_q_len
