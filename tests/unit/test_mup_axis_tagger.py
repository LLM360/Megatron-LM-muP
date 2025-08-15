"""Unit tests for axis-aware μP tagging.

The tests cover role classification heuristics, shape-based fallbacks and the
``apply_mup_lr_mult`` helper without relying on the full Megatron stack.
"""

import torch
from torch import nn
import pytest

from user_hooks.mup_axis_tagger import tag_axis_aware, apply_mup_lr_mult


def test_embedding_is_to_width() -> None:
    h = 8
    model = nn.Module()
    model.embed = nn.Embedding(10, h)
    tag_axis_aware(model, hidden_size=h, base_hidden=h)
    assert model.embed.weight.mup_role == "to_width"


def test_head_is_from_width() -> None:
    h = 8
    model = nn.Module()
    model.lm_head = nn.Linear(h, 20, bias=False)
    tag_axis_aware(model, hidden_size=h, base_hidden=h)
    assert model.lm_head.weight.mup_role == "from_width"


def test_linear_in_tracks_width_is_from() -> None:
    h = 8
    model = nn.Module()
    model.lin = nn.Linear(h, 2 * h, bias=False)
    tag_axis_aware(model, hidden_size=h, base_hidden=h)
    assert model.lin.weight.mup_role == "from_width"


def test_linear_out_tracks_width_is_to() -> None:
    h = 8
    model = nn.Module()
    model.lin = nn.Linear(2 * h, h, bias=False)
    tag_axis_aware(model, hidden_size=h, base_hidden=h)
    assert model.lin.weight.mup_role == "to_width"


def test_shape_fallback_unknown_module() -> None:
    h = 8

    class Unknown(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(4, h))

    model = Unknown()
    tag_axis_aware(model, hidden_size=h, base_hidden=h)
    assert model.weight.mup_role == "from_width"


def test_router_is_from_width() -> None:
    h = 8

    class Router(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(h, h))
            self.num_experts = 2
            self.top_k = 1

    model = Router()
    tag_axis_aware(model, hidden_size=h, base_hidden=h)
    assert model.weight.mup_role == "from_width"


def test_idempotent_tagging() -> None:
    h = 8
    model = nn.Module()
    model.lin = nn.Linear(h, h, bias=False)
    tag_axis_aware(model, hidden_size=h, base_hidden=h)
    first_role = model.lin.weight.mup_role
    tag_axis_aware(model, hidden_size=h, base_hidden=h)
    assert model.lin.weight.mup_role == first_role


def test_apply_mup_lr_mult_groups() -> None:
    h = 8
    model = nn.Module()
    model.from_w = nn.Linear(h, h * 2, bias=False)
    model.to_w = nn.Linear(h * 2, h, bias=False)
    tag_axis_aware(model, hidden_size=h, base_hidden=h // 2)

    opt = torch.optim.SGD(
        [
            {"params": [model.from_w.weight, model.to_w.weight]},
        ],
        lr=1.0,
    )
    apply_mup_lr_mult(opt)
    lr_mults = [g["lr_mult"] for g in opt.param_groups]
    expected = sorted([1.0, (h / (h // 2)) ** -0.5])
    assert sorted(lr_mults) == expected


def test_tp_layers_if_available() -> None:
    pytest.importorskip("numpy")
    pytest.importorskip("megatron.core.tensor_parallel.layers")
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
    from megatron.core.model_parallel_config import ModelParallelConfig

    h = 8
    config = ModelParallelConfig(use_cpu_initialization=True)
    model = nn.Module()
    model.col = ColumnParallelLinear(h, 3 * h, config=config, gather_output=True, bias=False)
    model.row = RowParallelLinear(h, h, config=config, input_is_parallel=False, bias=False)
    tag_axis_aware(model, hidden_size=h, base_hidden=h)
    assert model.col.weight.mup_role == "to_width"
    assert model.row.weight.mup_role == "from_width"
