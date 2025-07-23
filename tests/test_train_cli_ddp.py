import torch
import types

from src.gnn.train_cli_ddp import build_model


def test_build_model_variants():
    """确保 build_model 支持 gcn/mpnn/gat"""
    gcn = build_model("gcn", in_dim=4, edge_dim=1, hidden=8)
    mpnn = build_model("mpnn", in_dim=4, edge_dim=1, hidden=8)
    gat = build_model("gat", in_dim=4, edge_dim=1, hidden=8)
    trans = build_model("transformer", in_dim=4, edge_dim=1, hidden=8)

    # 简单形状检查 (输入特征数)
    assert gcn.conv1.in_channels == 4
    assert mpnn.layer1.mlp[0].in_features == 4 + 1  # node + edge
    assert gat.conv1.in_channels == 4
    assert trans.conv1.in_channels == 4


def test_freeze_base_logic():
    """模拟 freeze_base=True 时, 除最后 lin 层外参数均被冻结"""
    model = build_model("gcn", in_dim=4, edge_dim=1, hidden=8)

    # 模拟脚本中的 freeze_base 操作
    for name, param in model.named_parameters():
        if "lin" not in name:
            param.requires_grad = False

    # 验证
    frozen = [p.requires_grad for n, p in model.named_parameters() if "lin" not in n]
    trainable = [p.requires_grad for n, p in model.named_parameters() if "lin" in n]

    assert all(flag is False for flag in frozen)
    assert all(flag is True for flag in trainable) 