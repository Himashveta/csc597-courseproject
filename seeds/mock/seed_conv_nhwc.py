"""Seed: rank-4 conv, channels-last fp16, winograd path."""
from mock_compiler import OpDescriptor, GraphBreak
from mock_compiler import OP_MATMUL, OP_CONV, OP_REDUCE, OP_POINTWISE
from mock_compiler import DTYPE_I8, DTYPE_I16, DTYPE_I32, DTYPE_I64
from mock_compiler import DTYPE_FP16, DTYPE_BF16, DTYPE_FP32, DTYPE_FP64
from mock_compiler import LAYOUT_CONTIGUOUS, LAYOUT_CHANNELS_LAST, LAYOUT_STRIDED
from mock_compiler import compile as _mc_compile

def run_seed(op):
    try:
        _mc_compile(op)
    except GraphBreak:
        pass

run_seed(OpDescriptor(op_type=OP_CONV, dtype=DTYPE_FP16,
                     layout=LAYOUT_CHANNELS_LAST, rank=4,
                     shape=[1, 3, 32, 32], fuse_hint=1))
