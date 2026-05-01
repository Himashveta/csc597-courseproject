"""
mock_compiler.py — Python frontend over the C mock compiler.

Mirrors the layout of a real DL stack (Dynamo / Inductor):
    user input  ->  Python frontend (validate, normalise, decide)
                ->  C backend       (lower, fuse, codegen)

The frontend has its own branch logic that we want covered separately
from the C side. Many real PyTorch bugs sit at this seam — the frontend
accepts something the backend can't actually handle, or vice versa.

This module is what the fuzzer drives. Seeds construct OpDescriptor
instances and pass them to compile().
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
import pathlib
from typing import Optional

# ---------------------------------------------------------------------------
# Constants mirrored from mock_compiler.h.
# ---------------------------------------------------------------------------

MC_MAX_RANK = 6

OP_MATMUL    = 1
OP_CONV      = 2
OP_REDUCE    = 3
OP_POINTWISE = 4

DTYPE_I8   = 1
DTYPE_I16  = 2
DTYPE_I32  = 3
DTYPE_I64  = 4
DTYPE_FP16 = 5
DTYPE_BF16 = 6
DTYPE_FP32 = 7
DTYPE_FP64 = 8

LAYOUT_CONTIGUOUS    = 1
LAYOUT_CHANNELS_LAST = 2
LAYOUT_STRIDED       = 3

STATUS_OK = 0


class MCOp(ctypes.Structure):
    _fields_ = [
        ("op_type", ctypes.c_int),
        ("dtype", ctypes.c_int),
        ("layout", ctypes.c_int),
        ("rank", ctypes.c_int),
        ("shape", ctypes.c_int * MC_MAX_RANK),
        ("reduce_dim", ctypes.c_int),
        ("fuse_hint", ctypes.c_int),
    ]


# ---------------------------------------------------------------------------
# Backend loader.
# ---------------------------------------------------------------------------

def _default_lib_path() -> str:
    here = pathlib.Path(__file__).resolve().parent
    candidates = [
        here.parent.parent / "target" / "libmock_compiler.so",
        here / "libmock_compiler.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise FileNotFoundError(
        "libmock_compiler.so not found. Build it with `make -C target`."
    )


_LIB: Optional[ctypes.CDLL] = None


def load_backend(path: Optional[str] = None) -> ctypes.CDLL:
    """Load the C backend. Idempotent."""
    global _LIB
    if _LIB is not None:
        return _LIB
    lib_path = path or os.environ.get("MOCK_COMPILER_LIB") or _default_lib_path()
    lib = ctypes.CDLL(lib_path)
    lib.mc_compile.argtypes = [ctypes.POINTER(MCOp), ctypes.c_char_p, ctypes.c_size_t]
    lib.mc_compile.restype = ctypes.c_int
    lib.mc_status_str.argtypes = [ctypes.c_int]
    lib.mc_status_str.restype = ctypes.c_char_p
    _LIB = lib
    return lib


# ---------------------------------------------------------------------------
# Input descriptor and result.
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class OpDescriptor:
    """A single op for the mock compiler.

    Seeds mutate these. Mirrors the C struct but lives in Python so we
    can validate / normalise before crossing the FFI boundary.
    """
    op_type: int
    dtype: int
    rank: int
    shape: list
    layout: int = LAYOUT_CONTIGUOUS
    reduce_dim: int = 0
    fuse_hint: int = 0

    def to_c(self) -> MCOp:
        op = MCOp()
        op.op_type = self.op_type
        op.dtype = self.dtype
        op.layout = self.layout
        op.rank = self.rank
        for i in range(MC_MAX_RANK):
            op.shape[i] = self.shape[i] if i < len(self.shape) else 0
        op.reduce_dim = self.reduce_dim
        op.fuse_hint = self.fuse_hint
        return op


@dataclasses.dataclass
class CompileResult:
    status: int
    status_name: str
    output: str
    graph_broke: bool = False
    break_reason: str = ""


# ---------------------------------------------------------------------------
# Frontend dispatch — the Python branches that coverage.py will track.
# ---------------------------------------------------------------------------

class GraphBreak(Exception):
    """Raised when the frontend refuses to lower into the backend."""


def _validate_dtype(dtype: int) -> None:
    if dtype < 1 or dtype > 8:
        raise GraphBreak(f"unsupported dtype {dtype}")


def _validate_op(op: OpDescriptor) -> None:
    if op.op_type not in (OP_MATMUL, OP_CONV, OP_REDUCE, OP_POINTWISE):
        raise GraphBreak(f"unknown op_type {op.op_type}")
    _validate_dtype(op.dtype)
    if op.rank < 0 or op.rank > MC_MAX_RANK:
        raise GraphBreak(f"rank out of range: {op.rank}")
    if len(op.shape) < op.rank:
        raise GraphBreak("shape shorter than rank")
    for s in op.shape[: op.rank]:
        if s < 0:
            raise GraphBreak("negative shape")


def _normalize_layout(op: OpDescriptor) -> OpDescriptor:
    """Frontend "specialization": rewrite layout based on rank."""
    if op.op_type == OP_CONV and op.rank == 4:
        # Channels-last only meaningful for rank 4 conv.
        return op
    if op.layout == LAYOUT_CHANNELS_LAST and op.rank != 4:
        # Frontend fallback: collapse to contiguous.
        return dataclasses.replace(op, layout=LAYOUT_CONTIGUOUS)
    return op


def _should_break(op: OpDescriptor) -> Optional[str]:
    """Decide whether the frontend should refuse to compile this.

    Mimics Dynamo's "graph break" decisions. Each branch here is real
    Python coverage that we want our fuzzer to drive.
    """
    if op.op_type == OP_REDUCE and op.reduce_dim < 0:
        return "negative reduce_dim"
    if op.op_type == OP_REDUCE and op.reduce_dim >= op.rank:
        return "reduce_dim >= rank"
    if op.op_type == OP_MATMUL and op.rank == 0:
        return "matmul on scalar"
    if op.op_type == OP_CONV and op.rank == 0:
        return "conv on scalar"
    if op.dtype == DTYPE_FP64 and op.op_type == OP_CONV:
        # Pretend we don't have an fp64 conv kernel.
        return "no fp64 conv kernel"
    if op.fuse_hint and op.op_type == OP_REDUCE and op.rank > 4:
        return "fusion hint with high-rank reduce"
    return None


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------

def compile(op: OpDescriptor, out_cap: int = 128) -> CompileResult:  # noqa: A001
    """Compile an op. This is what fuzzer seeds call.

    Pipeline:
        validate -> normalise -> graph_break? -> backend.mc_compile
    """
    lib = load_backend()
    _validate_op(op)
    op = _normalize_layout(op)
    reason = _should_break(op)
    if reason is not None:
        return CompileResult(
            status=STATUS_OK,
            status_name="ok",
            output="",
            graph_broke=True,
            break_reason=reason,
        )

    c_op = op.to_c()
    out_buf = ctypes.create_string_buffer(out_cap)
    status = lib.mc_compile(ctypes.byref(c_op), out_buf, out_cap)
    name = lib.mc_status_str(status).decode("ascii")
    return CompileResult(
        status=status,
        status_name=name,
        output=out_buf.value.decode("ascii", errors="replace"),
        graph_broke=False,
        break_reason="",
    )


__all__ = [
    "OP_MATMUL", "OP_CONV", "OP_REDUCE", "OP_POINTWISE",
    "DTYPE_I8", "DTYPE_I16", "DTYPE_I32", "DTYPE_I64",
    "DTYPE_FP16", "DTYPE_BF16", "DTYPE_FP32", "DTYPE_FP64",
    "LAYOUT_CONTIGUOUS", "LAYOUT_CHANNELS_LAST", "LAYOUT_STRIDED",
    "OpDescriptor", "CompileResult", "GraphBreak",
    "compile", "load_backend",
]
