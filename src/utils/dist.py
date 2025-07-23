"""分布式训练工具"""
from __future__ import annotations

import os
import torch
import torch.distributed as dist


def init_distributed(local_rank: int):
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
    else:
        dist.init_process_group(backend="gloo")


def is_rank_zero() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group() 