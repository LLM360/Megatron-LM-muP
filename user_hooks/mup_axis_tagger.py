"""Axis-aware \u03bcP tagger for Megatron-LM models.

The tagger attaches \u03bcP metadata to every parameter, describing its role
relative to the model width. Parameters receive the following attributes:

- ``mup_role``: ``"to_width"``, ``"from_width"``, or ``"neutral"``
- ``mup_lr_mult``: learning-rate multiplier
- ``mup_init_scale``: initialization scaling to apply post-construction
- ``megatron_name``: dotted parameter name (copied from ``named_parameters``)

The classification determines whether the parameter's rows or columns track
the model width. For a weight ``W \u2208 R^{din \u00d7 dout}``:

* ``"to_width"``: columns map into width space (``dout`` scales with width).
* ``"from_width"``: rows originate from width space (``din`` scales with width).
* ``"neutral"``: biases, norms, positional embeddings and other non-scaling
  parameters.

The function is idempotent and performs a single pass over all parameters.
Example
-------
>>> model = build_model()
>>> tag_axis_aware(model, hidden_size=4096, base_hidden=1024)
>>> for n, p in list(model.named_parameters())[:3]:
...     print(n, p.mup_role, p.mup_lr_mult, p.mup_init_scale)
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

# Megatron Core imports are optional.
try:  # pragma: no cover - Import is optional.
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
except Exception:  # pragma: no cover - Import is optional.
    ColumnParallelLinear = tuple()  # type: ignore
    RowParallelLinear = tuple()  # type: ignore

__all__ = ["tag_axis_aware"]


def _is_multiple_or_equal(x: Optional[int], y: int) -> bool:
    """Return True if ``x`` is equal to or an integer multiple of ``y``."""
    if x is None or y <= 0:
        return False
    if abs(x - y) < 1e-6:
        return True
    return x % y == 0


def tag_axis_aware(model: nn.Module, hidden_size: int, base_hidden: int) -> None:
    """Attach \u03bcP axis tags to all parameters in ``model``.

    Parameters
    ----------
    model:
        The model whose parameters will be tagged.
    hidden_size:
        Current model width.
    base_hidden:
        Base width used when defining \u03bcP scalings.

    Notes
    -----
    The function adds four attributes to every :class:`torch.nn.Parameter`:

    ``mup_role`` : ``str``
        One of ``{"to_width", "from_width", "neutral"}``.
    ``mup_lr_mult`` : ``float``
        Learning-rate multiplier. ``r**-0.5`` for ``"from_width"`` parameters,
        otherwise ``1.0``.
    ``mup_init_scale`` : ``float``
        Initialization scaling identical to ``mup_lr_mult`` for ``"from_width"``
        parameters and ``1.0`` otherwise.
    ``megatron_name`` : ``str``
        The dotted parameter name; set only if the attribute does not already
        exist.

    The pass is deterministic and does not mutate parameter data.
    """
    r = float(hidden_size) / float(base_hidden)

    modules: Dict[str, nn.Module] = dict(model.named_modules())

    to_width_hints = ("query_key_value", "linear_qkv", "dense_h_to_4h")
    from_width_hints = ("linear_proj", "attention.dense", "dense_4h_to_h")
    embed_like = (
        "embedding",
        "embed",
        "lm_head",
        "output_layer",
        "readout",
        "head",
        "router",
        "gating",
    )

    for name, p in model.named_parameters(recurse=True):
        lower_name = name.lower()
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent_module = modules.get(parent_name)

        role = "neutral"

        # 1) Norms or biases are neutral.
        if lower_name.endswith("bias") or ".bias" in lower_name or "norm" in lower_name:
            role = "neutral"
        # Positional embeddings are neutral.
        elif "position" in lower_name or "pos_emb" in lower_name:
            role = "neutral"
        # 2) Embeddings/router/gating/lm_head/output_layer -> from_width.
        elif isinstance(parent_module, nn.Embedding) or any(
            token in lower_name for token in embed_like
        ):
            role = "from_width"
        # 3) RowParallelLinear -> from_width.
        elif isinstance(parent_module, RowParallelLinear):
            role = "from_width"
        # 4) ColumnParallelLinear -> to_width.
        elif isinstance(parent_module, ColumnParallelLinear):
            role = "to_width"
        else:
            # 5) Parent exposes input_size/output_size.
            din = getattr(parent_module, "input_size", None) if parent_module else None
            dout = getattr(parent_module, "output_size", None) if parent_module else None
            if isinstance(din, int) and isinstance(dout, int):
                din_tracks = _is_multiple_or_equal(din, hidden_size)
                dout_tracks = _is_multiple_or_equal(dout, hidden_size)
                if din_tracks and not dout_tracks:
                    role = "from_width"
                elif dout_tracks:
                    role = "to_width"
            # 6) Name fallback patterns.
            elif any(h in lower_name for h in from_width_hints):
                role = "from_width"
            elif any(h in lower_name for h in to_width_hints):
                role = "to_width"
            else:
                role = "neutral"

        if p.ndim <= 1 and role not in {"from_width", "to_width"}:
            role = "neutral"

        if role == "from_width":
            lr_mult = r ** -0.5
            init_scale = lr_mult
        else:
            lr_mult = 1.0
            init_scale = 1.0

        setattr(p, "mup_role", role)
        setattr(p, "mup_lr_mult", lr_mult)
        setattr(p, "mup_init_scale", init_scale)
        if not hasattr(p, "megatron_name"):
            setattr(p, "megatron_name", name)
