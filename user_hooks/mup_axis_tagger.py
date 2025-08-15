"""Axis-aware \u03bcP tagging utilities.

The :func:`tag_axis_aware` function scans a model once and attaches
micro-parameterization (\u03bcP) metadata to each :class:`~torch.nn.Parameter`.
The metadata describes how a parameter's axes relate to the model's *width*
dimension so that later stages (e.g., optimizer policies or initialization
rescaling) can apply width-aware rules without re-inspecting module structure.

For every parameter ``p`` the following attributes are created:

``p.mup_role``
    Textual role: ``"to_width"`` when ``p`` projects *into* the model width,
    ``"from_width"`` when it projects *out of* width, and ``"neutral"`` for
    scalars/vectors or parameters unrelated to width.
``p.mup_lr_mult``
    Recommended learning-rate multiplier. ``1.0`` for most parameters and
    ``r**-0.5`` for ``"from_width"`` parameters where ``r = hidden_size / base_hidden``.
``p.mup_init_scale``
    Initialization-time multiplier (mirrors ``mup_lr_mult`` in this default
    policy).
``p.megatron_name``
    Dotted parameter name as produced by :meth:`~torch.nn.Module.named_parameters`.

Embeddings map vocabulary indices into width space and are therefore tagged
``"to_width"`` even when their weights are reused as a readout layer (tied
embeddings). Output heads and routing/gating modules are tagged
``"from_width"``.

Use :func:`apply_mup_lr_mult` to surface the ``mup_lr_mult`` attribute to a
``torch.optim.Optimizer`` via its ``lr_mult`` param-group field.

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

__all__ = ["tag_axis_aware", "apply_mup_lr_mult"]

try:  # pragma: no cover - optional import
    # Megatron-Core provides metadata-rich parallel linear layers.  We only
    # import them if available so that this module has no hard dependency on
    # Megatron at import time.
    from megatron.core.tensor_parallel.layers import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
except Exception:  # pragma: no cover - optional import
    # When Megatron-Core is absent the type checks below simply fall back to
    # attribute-based heuristics.
    ColumnParallelLinear = None  # type: ignore
    RowParallelLinear = None  # type: ignore

Role = str  # Narrow type alias for readability


def _is_row_parallel(module: Optional[nn.Module]) -> bool:
    """Best-effort check for Megatron's row-parallel linear layer.

    ``RowParallelLinear`` shards its *input* dimension across tensor-parallel
    ranks.  We detect it either via an ``isinstance`` check (when the class is
    available) or by probing the lightweight ``input_is_parallel`` attribute used
    by several implementations.
    """

    if module is None:
        return False
    try:
        if RowParallelLinear is not None and isinstance(module, RowParallelLinear):
            return True
    except Exception:
        # If ``isinstance`` fails for any reason, fall through to the heuristic.
        pass
    return bool(getattr(module, "input_is_parallel", False))


def _is_col_parallel(module: Optional[nn.Module]) -> bool:
    """Best-effort check for Megatron's column-parallel linear layer.

    ``ColumnParallelLinear`` shards its *output* dimension across tensor-parallel
    ranks.  Similar to :func:`_is_row_parallel`, we fall back to probing the
    ``gather_output`` attribute when the Megatron class is unavailable.
    """

    if module is None:
        return False
    try:
        if ColumnParallelLinear is not None and isinstance(module, ColumnParallelLinear):
            return True
    except Exception:
        pass
    return bool(getattr(module, "gather_output", False))


def _is_multiple_or_equal(x: int, y: int) -> bool:
    """Return ``True`` if ``x`` equals or is an integer multiple of ``y``.

    The check is symmetric so that either ``x`` being a multiple of ``y`` or
    vice versa counts as tracking width. Negative or zero dimensions always
    return ``False``.
    """

    if x <= 0 or y <= 0:
        return False
    return x == y or y % x == 0


def tag_axis_aware(
    model: nn.Module,
    hidden_size: int,
    base_hidden: int,
    extra_hints: Optional[Dict[str, Role]] = None,
) -> None:
    """Attach \u03bcP width metadata to every parameter.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be tagged.
    hidden_size : int
        Global model width of the scaled model.
    base_hidden : int
        Reference width of the base model. ``r = hidden_size / base_hidden``
        feeds into the default \u03bcP scaling rule ``r**-0.5`` for ``from_width``
        parameters.
    extra_hints : dict[str, Role], optional
        Mapping from lowercase substrings to explicit roles. These hints are
        consulted *last*, after all structural and shape-based rules.

    Notes
    -----
    The pass is deterministic, does not mutate parameter *data*, and is
    idempotent: re-running it simply overwrites the same attributes with
    identical values.
    """

    r = float(hidden_size) / float(base_hidden)

    # Map module names to instances so we can quickly retrieve the parent module
    # for any given parameter.
    modules: Dict[str, nn.Module] = dict(model.named_modules())

    def tracks_width_dim(dim: Optional[int]) -> bool:
        """Return ``True`` if ``dim`` tracks the global ``hidden_size``.

        A dimension is considered width-tracking when it equals the global width
        or when the two are integer multiples of each other.  This makes the
        check robust to tensor-parallel or expert-parallel sharding where local
        dimensions are fractional views of the global width.
        """

        return isinstance(dim, int) and _is_multiple_or_equal(dim, hidden_size)

    def classify(name: str, parent: Optional[nn.Module], p: nn.Parameter) -> Role:
        """Infer the \u03bcP role for a parameter.

        Rules are ordered from most explicit to most heuristic; earlier matches
        take precedence.
        """

        lname = name.lower()

        # 1. Scalars/vectors and normalisation/positional parameters never track
        #    the model width.
        if p.ndim <= 1 or "norm" in lname or "position" in lname or "pos_emb" in lname:
            return "neutral"

        # 2. Embeddings map vocab -> width (to_width). Output heads and routing
        #    modules consume width (from_width).
        if isinstance(parent, nn.Embedding) or "embedding" in lname or "embed" in lname:
            return "to_width"

        if any(t in lname for t in ("lm_head", "output_layer", "readout")):
            return "from_width"

        if (
            parent is not None
            and hasattr(parent, "num_experts")
            and hasattr(parent, "top_k")
        ) or "router" in lname or "gating" in lname:
            return "from_width"

        # 3. Explicit tensor-parallel modules provide the clearest signal.
        if _is_row_parallel(parent):
            return "from_width"
        if _is_col_parallel(parent):
            return "to_width"

        # 4. Use structural metadata (input/output dims) when available.
        din = dout = None
        if parent is not None:
            din = getattr(parent, "input_size", getattr(parent, "in_features", None))
            dout = getattr(parent, "output_size", getattr(parent, "out_features", None))
        din_tracks = tracks_width_dim(din)
        dout_tracks = tracks_width_dim(dout)
        if din_tracks and not dout_tracks:
            return "from_width"
        if dout_tracks and not din_tracks:
            return "to_width"

        # 5. Shape-based fallback for generic 2-D parameters.
        if p.ndim == 2:
            # Without metadata we only trust exact matches for width-tracking.
            in_tracks = p.shape[1] == hidden_size
            out_tracks = p.shape[0] == hidden_size
            if in_tracks and not out_tracks:
                return "from_width"
            if out_tracks and not in_tracks:
                return "to_width"

        # 6. Final name-based hints (built-in + user-provided).
        hints: Dict[str, Role] = {
            "up_proj": "to_width",
            "gate_proj": "to_width",
            "qkv": "to_width",
            "query_key_value": "to_width",
            "linear_qkv": "to_width",
            "down_proj": "from_width",
            "o_proj": "from_width",
        }
        if extra_hints:
            hints.update({k.lower(): v for k, v in extra_hints.items()})

        for key, role in hints.items():
            if key in lname:
                return role

        return "neutral"

    for name, p in model.named_parameters(recurse=True):
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = modules.get(parent_name)
        role = classify(name, parent, p)

        # Compute default \u03bcP multipliers. Only ``from_width`` parameters receive
        # the ``r**-0.5`` scaling; others remain at ``1.0``.
        mult = r ** -0.5 if role == "from_width" else 1.0

        setattr(p, "mup_role", role)
        setattr(p, "mup_lr_mult", float(mult))
        setattr(p, "mup_init_scale", float(mult))
        if not hasattr(p, "megatron_name"):
            setattr(p, "megatron_name", name)


def apply_mup_lr_mult(optimizer: torch.optim.Optimizer) -> None:
    """Write ``lr_mult`` fields in an optimizer's param groups.

    Each parameter's ``mup_lr_mult`` attribute is surfaced as a param-group
    ``lr_mult`` so that Megatron's optimizer and scheduler machinery can read
    it natively. Groups containing parameters with mixed multipliers are split
    to preserve correctness.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer whose ``param_groups`` will be annotated in-place.
    """

    new_groups = []
    for group in list(optimizer.param_groups):
        params = group.get("params", [])
        multipliers = {}
        for p in params:
            m = float(getattr(p, "mup_lr_mult", 1.0))
            multipliers.setdefault(m, []).append(p)

        if len(multipliers) <= 1:
            group.setdefault("lr_mult", next(iter(multipliers.keys())) if multipliers else 1.0)
            continue

        base = {k: v for k, v in group.items() if k != "params"}
        first_mult, first_params = next(iter(multipliers.items()))
        group["params"] = first_params
        group["lr_mult"] = first_mult
        for mult, ps in list(multipliers.items())[1:]:
            ng = base.copy()
            ng["params"] = ps
            ng["lr_mult"] = mult
            new_groups.append(ng)

    optimizer.param_groups.extend(new_groups)
