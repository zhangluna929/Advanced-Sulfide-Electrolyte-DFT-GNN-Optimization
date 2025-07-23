"""分布式工具测试"""
import pytest
import torch
import torch.distributed as dist

from src.utils.dist import init_distributed, is_rank_zero, get_world_size, get_rank


def test_dist_utils():
    assert is_rank_zero() == True
    assert get_world_size() == 1
    assert get_rank() == 0


def test_init_distributed():
    if torch.cuda.is_available():
        init_distributed(0)
        assert dist.is_initialized()
        dist.destroy_process_group()
    else:
        init_distributed(0)
        assert dist.is_initialized()
        dist.destroy_process_group() 