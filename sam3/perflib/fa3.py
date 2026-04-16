# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

import os

import torch

_FORCE_FA2 = os.environ.get("USE_FLASH_ATTN_2", "0") == "1"

if not _FORCE_FA2:
    try:
        from flash_attn_interface import flash_attn_func as _fa3
        _USE_FA3 = True
    except ImportError:
        _USE_FA3 = False
else:
    _USE_FA3 = False


@torch.library.custom_op("flash::flash_attn_func", mutates_args=())
def flash_attn_func_op(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    if _USE_FA3:
        return _fa3(q, k, v)
    else:
        from flash_attn import flash_attn_func as fa2

        return fa2(q, k, v)


def flash_attn_func(q, k, v):
    dtype = torch.float8_e4m3fn if _USE_FA3 else torch.bfloat16
    return flash_attn_func_op(q.to(dtype), k.to(dtype), v.to(dtype)).to(q.dtype)


@flash_attn_func_op.register_fake
def _(q, k, v, **kwargs):
    # two outputs:
    # 1. output: (batch, seq_len, num_heads, head_dim)
    # 2. softmax_lse: (batch, num_heads, seq_len) with dtype=torch.float32
    # output needs to be bfloat16, not float8!
    meta_q = torch.empty_like(q, dtype=torch.bfloat16).contiguous()
    return meta_q
