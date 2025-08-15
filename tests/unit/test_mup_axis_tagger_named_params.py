"""Integration test exercising the tagger on a real Megatron GPT model.

The goal is to iterate over the major parameter types produced by Megatron
(core embedding, attention, MLP, and normalization weights) and ensure the
axis-aware tagger assigns sensible roles to all of them.
"""
import tempfile

import torch
import torch.distributed as dist
import pytest

from user_hooks.mup_axis_tagger import tag_axis_aware

try:  # pragma: no cover - optional imports
    from megatron.core.transformer.transformer_config import TransformerConfig
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core import parallel_state
except Exception as exc:  # pragma: no cover
    pytest.skip(f"Megatron Core unavailable: {exc}", allow_module_level=True)


def test_tag_axis_aware_handles_named_parameter_types() -> None:
    """Build a tiny GPT model and verify tagging for key parameter groups."""

    with tempfile.NamedTemporaryFile() as tmp:
        dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp.name}")
        parallel_state.initialize_model_parallel()
        try:
            config = TransformerConfig(
                num_layers=1,
                hidden_size=8,
                num_attention_heads=4,
                ffn_hidden_size=32,
                use_cpu_initialization=True,
            )
            layer_spec = get_gpt_layer_local_spec()
            model = GPTModel(
                config=config,
                transformer_layer_spec=layer_spec,
                vocab_size=16,
                max_sequence_length=8,
            )
            tag_axis_aware(model, hidden_size=8, base_hidden=8)

            roles = {n: p.mup_role for n, p in model.named_parameters()}

            # Embedding and readout parameters consume width.
            assert roles["embedding.word_embeddings.weight"] == "from_width"
            assert roles["output_layer.weight"] == "from_width"

            # Attention projections.
            assert roles["decoder.layers.0.self_attention.linear_qkv.weight"] == "to_width"
            assert roles["decoder.layers.0.self_attention.linear_proj.weight"] == "from_width"

            # MLP projections.
            assert roles["decoder.layers.0.mlp.linear_fc1.weight"] == "to_width"
            assert roles["decoder.layers.0.mlp.linear_fc2.weight"] == "from_width"

            # Norm weights stay neutral.
            assert roles["decoder.layers.0.input_layernorm.weight"] == "neutral"

            # Ensure every parameter received a valid role.
            for _, p in model.named_parameters():
                assert hasattr(p, "mup_role")
                assert p.mup_role in {"to_width", "from_width", "neutral"}
        finally:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()
