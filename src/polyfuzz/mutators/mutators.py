"""Mutators rewrite a seed's source into a new candidate.

Each mutator is a pure function: source: str -> Optional[str]. Returning
None signals "this mutator can't apply here", and the fuzzer picks
another. Mutators are intentionally simple: numeric perturbation,
shape rewrites, dtype swaps, op-type swaps. No LLM in the loop.

Why this is enough: PolyFuzz's contribution is the merged coverage
*signal*, not the mutator engine. With a unified bitmap and a corpus
that retains seeds whenever either Python or C/C++ coverage moves,
even a simple mutator catalogue plays the long game well enough to
expose multi-language paths.
"""

from __future__ import annotations

import ast
import dataclasses
import random
import re
from typing import Callable, List, Optional


@dataclasses.dataclass
class Mutator:
    name: str
    fn: Callable[[str, random.Random], Optional[str]]
    weight: float = 1.0

    def apply(self, source: str, rng: random.Random) -> Optional[str]:
        return self.fn(source, rng)


# ---------------------------------------------------------------------------
# Numeric mutators
# ---------------------------------------------------------------------------

_BOUNDARY_INTS = [-1, 0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 63, 64, 65, 127, 128,
                  255, 256, 1023, 1024, 1025, 4096]


def _mutate_int_literal(source: str, rng: random.Random) -> Optional[str]:
    """Perturb a single integer literal to a boundary or random value."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    int_nodes: List[ast.Constant] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, int) \
                and not isinstance(node.value, bool):
            int_nodes.append(node)
    if not int_nodes:
        return None
    target = rng.choice(int_nodes)
    new_val = rng.choice(_BOUNDARY_INTS)
    if new_val == target.value:
        new_val = rng.choice(_BOUNDARY_INTS)
    target.value = new_val
    try:
        return ast.unparse(tree)
    except Exception:
        return None


def _mutate_shape_list(source: str, rng: random.Random) -> Optional[str]:
    """Find a Python list of small ints and perturb one entry."""
    pattern = re.compile(r"\[(\s*-?\d+\s*(?:,\s*-?\d+\s*)+)\]")
    matches = list(pattern.finditer(source))
    if not matches:
        return None
    m = rng.choice(matches)
    nums = [int(x.strip()) for x in m.group(1).split(",")]
    idx = rng.randrange(len(nums))
    nums[idx] = rng.choice(_BOUNDARY_INTS)
    new_lit = "[" + ", ".join(str(n) for n in nums) + "]"
    return source[: m.start()] + new_lit + source[m.end():]


# ---------------------------------------------------------------------------
# Dtype / op-type swap mutators
# ---------------------------------------------------------------------------

_DTYPE_NAMES = [
    "DTYPE_I8", "DTYPE_I16", "DTYPE_I32", "DTYPE_I64",
    "DTYPE_FP16", "DTYPE_BF16", "DTYPE_FP32", "DTYPE_FP64",
]
_TORCH_DTYPES = [
    "torch.float32", "torch.float16", "torch.bfloat16", "torch.float64",
    "torch.int8", "torch.int16", "torch.int32", "torch.int64",
    "torch.bool",
]
_OP_NAMES = ["OP_MATMUL", "OP_CONV", "OP_REDUCE", "OP_POINTWISE"]
_LAYOUT_NAMES = ["LAYOUT_CONTIGUOUS", "LAYOUT_CHANNELS_LAST", "LAYOUT_STRIDED"]


def _swap_token(source: str, rng: random.Random,
                vocab: List[str]) -> Optional[str]:
    """Swap one occurrence of a known constant for a different one in vocab.

    The replacement is restricted to constants ALREADY imported by the
    seed (or already mentioned anywhere in the source). This keeps
    mutated children syntactically valid: we don't introduce names that
    the seed never pulled in. For files that import the whole module
    surface (e.g. `from m import *` or large explicit imports), this
    reduces to "anything in vocab".
    """
    candidates = [v for v in vocab if v in source]
    if len(candidates) < 2:
        # Need at least two known names so we have something to swap to.
        return None
    found = rng.choice(candidates)
    replacement = rng.choice([v for v in candidates if v != found])
    idx = source.find(found)
    return source[:idx] + replacement + source[idx + len(found):]


def _mutate_dtype(source: str, rng: random.Random) -> Optional[str]:
    return (_swap_token(source, rng, _DTYPE_NAMES)
            or _swap_token(source, rng, _TORCH_DTYPES))


def _mutate_op_type(source: str, rng: random.Random) -> Optional[str]:
    return _swap_token(source, rng, _OP_NAMES)


def _mutate_layout(source: str, rng: random.Random) -> Optional[str]:
    return _swap_token(source, rng, _LAYOUT_NAMES)


# ---------------------------------------------------------------------------
# Boolean / structural toggles
# ---------------------------------------------------------------------------

def _toggle_bool(source: str, rng: random.Random) -> Optional[str]:
    bools = [m for m in re.finditer(r"\b(True|False)\b", source)]
    if not bools:
        return None
    m = rng.choice(bools)
    new = "True" if m.group() == "False" else "False"
    return source[: m.start()] + new + source[m.end():]


def _bump_fuse_hint(source: str, rng: random.Random) -> Optional[str]:
    pattern = re.compile(r"fuse_hint\s*=\s*(\d+)")
    matches = list(pattern.finditer(source))
    if not matches:
        return None
    m = rng.choice(matches)
    cur = int(m.group(1))
    return source[: m.start()] + f"fuse_hint={1 - bool(cur)}" + source[m.end():]


# ---------------------------------------------------------------------------
# Composite mutator: apply 2 random mutators in sequence
# ---------------------------------------------------------------------------

def _composite(source: str, rng: random.Random) -> Optional[str]:
    base_pool = [
        _mutate_int_literal, _mutate_shape_list,
        _mutate_dtype, _mutate_op_type, _mutate_layout,
        _toggle_bool, _bump_fuse_hint,
    ]
    rng.shuffle(base_pool)
    out = source
    applied = 0
    for fn in base_pool:
        new = fn(out, rng)
        if new is not None and new != out:
            out = new
            applied += 1
        if applied >= 2:
            break
    return out if applied >= 1 else None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def get_mutator_registry() -> List[Mutator]:
    """Return the full mutator registry."""
    return [
        Mutator("int_literal",  _mutate_int_literal,  weight=2.0),
        Mutator("shape_list",   _mutate_shape_list,   weight=2.0),
        Mutator("dtype_swap",   _mutate_dtype,        weight=1.5),
        Mutator("op_type_swap", _mutate_op_type,      weight=1.0),
        Mutator("layout_swap",  _mutate_layout,       weight=1.0),
        Mutator("toggle_bool",  _toggle_bool,         weight=0.5),
        Mutator("fuse_hint",    _bump_fuse_hint,      weight=0.5),
        Mutator("composite",    _composite,           weight=1.5),
    ]
