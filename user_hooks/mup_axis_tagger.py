"""Axis-aware \u03bcP tagging utilities for Megatron-LM models.

This module provides utilities to classify parameters by their relationship
to the model width. ``tag_axis_aware`` annotates each parameter with a width
role and stores the dotted Megatron name. Additional helpers compute
learning-rate and initialization multipliers, apply them to the parameters,
and build optimizer parameter groups.

Example
-------
>>> model = build_model()
>>> tag_axis_aware(model, hidden_size=4096, base_hidden=1024)
>>> apply_mup_policy(model, r=4.0)
>>> for name, p in model.named_parameters():
...     print(name, p.mup_role, p.mup_lr_mult, p.mup_init_scale)
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn

__all__ = [
    "tag_axis_aware",
    "apply_mup_policy",
    "realize_init_rescales",
    "build_param_groups_from_tags",
]

try:  # pragma: no cover - optional import
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
except ImportError:  # pragma: no cover - optional import
    ColumnParallelLinear = None  # type: ignore
    RowParallelLinear = None  # type: ignore

Role = str
_ROLE_AXIS = {"to_width": 1, "from_width": -1, "neutral": 0}


def _is_row_parallel(module: Optional[nn.Module]) -> bool:
    """Return True if ``module`` behaves like a row-parallel linear layer."""
    if module is None:
        return False
    try:
        if RowParallelLinear is not None and isinstance(module, RowParallelLinear):
            return True
    except Exception:
        logging.debug("Exception in _is_row_parallel: ", exc_info=True)
    return bool(getattr(module, "input_is_parallel", False))


def _is_col_parallel(module: Optional[nn.Module]) -> bool:
    """Return True if ``module`` behaves like a column-parallel linear layer."""
    if module is None:
        return False
    try:
        if ColumnParallelLinear is not None and isinstance(module, ColumnParallelLinear):
            return True
    except Exception:
        pass
    return bool(getattr(module, "gather_output", False))


def tag_axis_aware(
    model: nn.Module,
    hidden_size: int,
    base_hidden: int,
    *,
    tp_world_size: int = 1,
    role_overrides: Optional[Dict[str, Role]] = None,
) -> None:
    """Attach \u03bcP width-role tags to ``model`` parameters.

    Parameters
    ----------
    model:
        Model whose parameters will be tagged.
    hidden_size:
        Global model width.
    base_hidden:
        Base width used for \u03bcP scaling (unused directly).
    tp_world_size:
        Tensor-parallel world size used when identifying width-tracking
        dimensions.
    role_overrides:
        Optional mapping of lowercase substrings to explicit roles, allowing
        manual overrides for special layers.

    Notes
    -----
    Each :class:`torch.nn.Parameter` receives the following attributes:

    ``mup_role`` : ``str``
        One of ``{"to_width", "from_width", "neutral"}``.
    ``mup_axis`` : ``int``
        ``+1`` for ``"to_width"``, ``-1`` for ``"from_width"``, ``0`` otherwise.
    ``megatron_name`` : ``str``
        The dotted parameter name.
    """
    modules: Dict[str, nn.Module] = dict(model.named_modules())
    overrides = role_overrides or {}

    def tracks_width_dim(dim: Optional[int]) -> bool:
        if not isinstance(dim, int) or dim <= 0:
            return False
        if dim == hidden_size or dim * tp_world_size == hidden_size:
            return True
        return dim % hidden_size == 0 or hidden_size % dim == 0

    def classify(name: str, parent: Optional[nn.Module], p: nn.Parameter) -> Role:
        lname = name.lower()

        if p.ndim <= 1:
            return "neutral"

        for pat, role in overrides.items():
            if pat in lname:
                return role

        if "norm" in lname or "position" in lname or "pos_emb" in lname:
            return "neutral"

        if isinstance(parent, nn.Embedding) or any(
            t in lname
            for t in (
                "embedding",
                "embed",
                "lm_head",
                "output_layer",
                "readout",
                "head",
                "router",
                "gating",
            )
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
        if din_tracks and dout_tracks and isinstance(din, int) and isinstance(dout, int):
            return "to_width" if dout >= din else "from_width"

        if any(h in lname for h in ("dense_4h_to_h", "down_proj", "linear_proj", "attention.dense")):
            return "from_width"
        if any(h in lname for h in ("query_key_value", "linear_qkv", "dense_h_to_4h", "gate_proj", "up_proj")):
            return "to_width"

        return "neutral"

    for name, p in model.named_parameters(recurse=True):
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = modules.get(parent_name)
        role = classify(name, parent, p)
        setattr(p, "mup_role", role)
        setattr(p, "mup_axis", _ROLE_AXIS[role])
        if not hasattr(p, "megatron_name"):
            setattr(p, "megatron_name", name)


def apply_mup_policy(
    model: nn.Module,
    r: float,
    *,
    role_to_lr_exp: Optional[Dict[Role, float]] = None,
    role_to_init_exp: Optional[Dict[Role, float]] = None,
) -> None:
    """Compute learning-rate and init multipliers from tagged roles.

    Parameters
    ----------
    model:
        Model whose parameters have been tagged.
    r:
        Width ratio ``hidden_size / base_hidden``.
    role_to_lr_exp:
        Mapping from role to exponent for learning-rate multipliers.
    role_to_init_exp:
        Mapping from role to exponent for initialization multipliers. Defaults
        to ``role_to_lr_exp`` when ``None``.
    """
    lr_exp = {"from_width": -0.5, "to_width": 0.0, "neutral": 0.0}
    if role_to_lr_exp:
        lr_exp.update(role_to_lr_exp)
    init_exp = lr_exp.copy()
    if role_to_init_exp:
        init_exp.update(role_to_init_exp)

    for _, p in model.named_parameters():
        role = getattr(p, "mup_role", "neutral")
        lr_mult = r ** lr_exp.get(role, 0.0)
        init_mult = r ** init_exp.get(role, 0.0)
        setattr(p, "mup_lr_mult", float(lr_mult))
        setattr(p, "mup_init_scale", float(init_mult))


def realize_init_rescales(model: nn.Module) -> None:
    """In-place multiply weight tensors by ``mup_init_scale``."""
    with torch.no_grad():
        for _, p in model.named_parameters():
            if p.ndim <= 1:
                continue
            scale = float(getattr(p, "mup_init_scale", 1.0))
            if scale != 1.0:
                p.mul_(scale)


def build_param_groups_from_tags(
    model: nn.Module, base_lr: float, weight_decay: float
) -> Iterable[Dict[str, object]]:
    """Create optimizer parameter groups based on tagged multipliers.

    Parameters
    ----------
    model:
        Tagged model.
    base_lr:
        Base learning rate.
    weight_decay:
        Weight decay to apply to non-bias, non-norm parameters.
    """
    buckets: Dict[Tuple[float, float], Dict[str, object]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        wd = 0.0 if (p.ndim <= 1 or "norm" in lname) else weight_decay
        lr_mult = float(getattr(p, "mup_lr_mult", 1.0))
        sig = (round(base_lr * lr_mult, ROUND_PRECISION), round(wd, ROUND_PRECISION))
        group = buckets.setdefault(sig, {"params": [], "lr": sig[0], "weight_decay": sig[1]})
        group["params"].append(p)
    return list(buckets.values())
