"""Integration test covering Megatron-Core tensor-parallel layers."""

import torch
from torch import nn
import pytest

from user_hooks.mup_axis_tagger import tag_axis_aware

try:  # pragma: no cover - optional import
    from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
    from megatron.core.model_parallel_config import ModelParallelConfig
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Megatron Core unavailable: {exc}", allow_module_level=True)


class TinyMegatron(nn.Module):
    """Minimal module composed of Megatron's parallel linear layers."""

    def __init__(self, h: int, config: ModelParallelConfig):
        super().__init__()
        self.qkv = ColumnParallelLinear(
            h,
            3 * h,
            config=config,
            gather_output=True,
            bias=False,
            init_method=torch.nn.init.ones_,
        )
        self.proj = RowParallelLinear(
            h,
            h,
            config=config,
            input_is_parallel=False,
            bias=False,
            init_method=torch.nn.init.ones_,
            skip_bias_add=False,
        )


def test_tag_axis_aware_with_megatron_layers() -> None:
    """The tagger should recognize tensor-parallel sharding semantics."""

    config = ModelParallelConfig(use_cpu_initialization=True)
    model = TinyMegatron(8, config)
    tag_axis_aware(model, hidden_size=8, base_hidden=8)
    assert model.qkv.weight.mup_role == "to_width"
    assert model.proj.weight.mup_role == "from_width"
