"""Axis-aware \u03bcP tagging for Megatron-LM.

Run :func:`tag_axis_aware` immediately after model construction to attach
\u03bcP metadata to every parameter.  The function classifies each parameter by
how its dimensions relate to the model width and stores attributes that later
stages (e.g., initialization rescaling or optimizer policies) can consume.

Each parameter receives four attributes:

``mup_role``
    One of ``{"to_width", "from_width", "neutral"}``.
``mup_lr_mult``
    Learning-rate multiplier (``1.0`` except ``r**-0.5`` for ``"from_width"``).
``mup_init_scale``
    Initialization multiplier (same as ``mup_lr_mult``).
``megatron_name``
    Dotted parameter name (set only if missing).

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

__all__ = ["tag_axis_aware"]

try:  # pragma: no cover - optional import
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
except Exception:  # pragma: no cover - optional import
    ColumnParallelLinear = None  # type: ignore
    RowParallelLinear = None  # type: ignore

Role = str


def _is_row_parallel(module: Optional[nn.Module]) -> bool:
    """Return ``True`` if ``module`` behaves like a row-parallel linear layer."""
    if module is None:
        return False
    try:
        if RowParallelLinear is not None and isinstance(module, RowParallelLinear):
            return True
    except Exception:
        pass
    return bool(getattr(module, "input_is_parallel", False))


def _is_col_parallel(module: Optional[nn.Module]) -> bool:
    """Return ``True`` if ``module`` behaves like a column-parallel linear layer."""
    if module is None:
        return False
    try:
        if ColumnParallelLinear is not None and isinstance(module, ColumnParallelLinear):
            return True
    except Exception:
        pass
    return bool(getattr(module, "gather_output", False))


def _is_multiple_or_equal(x: int, y: int) -> bool:
    """Return ``True`` if ``x`` equals or is an integer multiple of ``y``."""
    if x <= 0 or y <= 0:
        return False
    return x == y or x % y == 0 or y % x == 0


def tag_axis_aware(model: nn.Module, hidden_size: int, base_hidden: int) -> None:
    """Attach \u03bcP width metadata to all parameters in ``model``.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be tagged.
    hidden_size : int
        Global model width of the scaled model.
    base_hidden : int
        Reference model width. ``r = hidden_size / base_hidden`` is used for
        multiplier computations.

    Notes
    -----
    The function performs a deterministic single pass without mutating parameter
    data. It is idempotent: running it multiple times overwrites the same
    attributes with identical values.
    """
    r = float(hidden_size) / float(base_hidden)
    modules: Dict[str, nn.Module] = dict(model.named_modules())

    def tracks_width_dim(dim: Optional[int]) -> bool:
        return isinstance(dim, int) and _is_multiple_or_equal(dim, hidden_size)

    def classify(name: str, parent: Optional[nn.Module], p: nn.Parameter) -> Role:
        lname = name.lower()

        if p.ndim <= 1 or "norm" in lname or "position" in lname or "pos_emb" in lname:
            return "neutral"

        if isinstance(parent, nn.Embedding) or any(
            t in lname for t in ("embedding", "embed", "lm_head", "output_layer", "readout", "router", "gating")
        ):
            return "from_width"

        if _is_row_parallel(parent):
            return "from_width"
        if _is_col_parallel(parent):
            return "to_width"

        din = None
        dout = None
        if parent is not None:
            din = getattr(parent, "input_size", getattr(parent, "in_features", None))
            dout = getattr(parent, "output_size", getattr(parent, "out_features", None))
        din_tracks = tracks_width_dim(din)
        dout_tracks = tracks_width_dim(dout)
        if din_tracks and not dout_tracks:
            return "from_width"
        if dout_tracks and not din_tracks:
            return "to_width"

        if any(h in lname for h in ("dense_4h_to_h", "linear_proj", "attention.dense")):
            return "from_width"
        if any(h in lname for h in ("query_key_value", "linear_qkv", "dense_h_to_4h")):
            return "to_width"

        return "neutral"

    for name, p in model.named_parameters(recurse=True):
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = modules.get(parent_name)
        role = classify(name, parent, p)
        mult = r ** -0.5 if role == "from_width" else 1.0
        setattr(p, "mup_role", role)
        setattr(p, "mup_lr_mult", float(mult))
        setattr(p, "mup_init_scale", float(mult))
        if not hasattr(p, "megatron_name"):
            setattr(p, "megatron_name", name)
