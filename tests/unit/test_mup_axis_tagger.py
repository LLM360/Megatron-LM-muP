"""Unit tests for the axis-aware µP tagger.

These tests use simple stand-in modules (including dummy tensor-parallel
sharding stubs) to exercise the core classification logic and multiplier
assignment without depending on the full Megatron stack.
"""

import math
import torch
from torch import nn

from user_hooks.mup_axis_tagger import tag_axis_aware


class DummyModel(nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.layernorm = nn.LayerNorm(h)
        self.linear_qkv = nn.Linear(h, 3 * h, bias=False)
        self.linear_proj = nn.Linear(3 * h, h, bias=False)
        self.bias = nn.Parameter(torch.zeros(h))
        self.embed = nn.Embedding(10, h)
        self.out_head = nn.Linear(h, 20, bias=False)


class DummyIO(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(din, dout))
        self.input_size = din
        self.output_size = dout


class DummyRowParallel(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.input_is_parallel = True
        self.weight = nn.Parameter(torch.randn(din, dout))
        self.input_size = din
        self.output_size = dout


class DummyColParallel(nn.Module):
    def __init__(self, din: int, dout: int):
        super().__init__()
        self.gather_output = True
        self.weight = nn.Parameter(torch.randn(din, dout))
        self.input_size = din
        self.output_size = dout


class Wrapper(nn.Module):
    """Compose the dummy modules into a single test fixture."""

    def __init__(self, h: int):
        super().__init__()
        self.model = DummyModel(h)
        self.conflict = DummyIO(20, h)
        self.linear_proj_conflict = self.conflict  # name hint vs axis
        self.row = DummyRowParallel(h, h)
        self.col = DummyColParallel(h, h)
        self.top_weight = nn.Parameter(torch.randn(h, h))


def test_tag_axis_aware_sets_expected_metadata() -> None:
    """Tag the dummy model and verify roles and multipliers."""

    h = 32
    model = Wrapper(h)
    tag_axis_aware(model, hidden_size=h, base_hidden=h // 2)
    r = (h / (h // 2)) ** -0.5

    for name, p in model.named_parameters():
        assert hasattr(p, "mup_role")
        assert hasattr(p, "mup_lr_mult")
        assert hasattr(p, "mup_init_scale")
        assert hasattr(p, "megatron_name") and p.megatron_name == name

    assert model.model.layernorm.weight.mup_role == "neutral"
    assert model.model.layernorm.bias.mup_role == "neutral"
    assert model.model.bias.mup_role == "neutral"

    assert model.model.embed.weight.mup_role == "from_width"
    assert model.model.linear_qkv.weight.mup_role == "to_width"
    assert model.model.linear_proj.weight.mup_role == "from_width"
    assert model.model.out_head.weight.mup_role == "from_width"

    assert model.row.weight.mup_role == "from_width"
    assert model.col.weight.mup_role == "to_width"

    assert model.linear_proj_conflict.weight.mup_role == "to_width"
    assert model.top_weight.mup_role == "neutral"

    assert math.isclose(model.model.embed.weight.mup_lr_mult, r)
    assert math.isclose(model.model.linear_proj.weight.mup_init_scale, r)
    assert model.model.linear_qkv.weight.mup_lr_mult == 1.0
    assert model.linear_proj_conflict.weight.mup_lr_mult == 1.0
