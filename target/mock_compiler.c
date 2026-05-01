/*
 * mock_compiler.c
 *
 * A miniature C "compiler backend" that mimics the kind of
 * decision branches found in real DL compiler backends (TorchInductor,
 * TVM, XLA). It exposes a single entry point, mc_compile, which takes
 * a flat operator descriptor and returns a status code.
 *
 * The point is not to be a real compiler. The point is to provide a
 * realistic branch-rich C target that:
 *   1. Has dozens of distinct paths controlled by input fields.
 *   2. Contains a few intentional latent bugs so the oracle has
 *      something to find (NULL deref, divide-by-zero, OOB write).
 *   3. Compiles cleanly with --coverage so gcov can record branches.
 *
 * This file is the "C backend" half of a Python+C fuzzing target.
 * The Python frontend (mock_compiler.py) does its own dispatch and
 * then calls into here.
 */

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mock_compiler.h"
#include "mc_probe.h"

/* Internal helpers ------------------------------------------------- */

static int is_pow2(uint64_t x) {
    return x != 0 && (x & (x - 1)) == 0;
}

static int dtype_is_float(int dtype) {
    return dtype == MC_DTYPE_FP16 || dtype == MC_DTYPE_BF16 ||
           dtype == MC_DTYPE_FP32 || dtype == MC_DTYPE_FP64;
}

static int dtype_is_int(int dtype) {
    return dtype == MC_DTYPE_I8 || dtype == MC_DTYPE_I16 ||
           dtype == MC_DTYPE_I32 || dtype == MC_DTYPE_I64;
}

/* Lowering decisions ---------------------------------------------- */

static int select_kernel(const mc_op_t *op) {
    /* Branchy selection logic. Each predicate emits a branch-variable
     * probe when MC_TRACE_FD is set in the environment; otherwise the
     * macros are no-ops. Probe IDs 1-99 reserved for select_kernel.
     */
    MC_PROBE_I1(1, op->op_type);
    if (op->op_type == MC_OP_MATMUL) {
        MC_PROBE_I1(2, op->rank);
        if (op->rank != 2 && op->rank != 3) {
            return MC_KERNEL_FALLBACK;
        }
        MC_PROBE_I2(3, op->shape[0], op->shape[1]);
        if (op->shape[0] == 0 || op->shape[1] == 0) {
            return MC_KERNEL_EMPTY;
        }
        MC_PROBE_I1(4, op->dtype);
        if (op->dtype == MC_DTYPE_FP16 || op->dtype == MC_DTYPE_BF16) {
            MC_PROBE_I2(5, op->shape[0], op->shape[1]);
            if (op->shape[0] % 8 == 0 && op->shape[1] % 8 == 0) {
                return MC_KERNEL_TENSOR_CORE;
            }
            return MC_KERNEL_VECTOR_HALF;
        }
        if (op->dtype == MC_DTYPE_FP32) {
            MC_PROBE_I2(6, op->shape[0], op->shape[1]);
            if (op->shape[0] >= 256 && op->shape[1] >= 256) {
                return MC_KERNEL_BLOCKED_GEMM;
            }
            return MC_KERNEL_NAIVE_GEMM;
        }
        return MC_KERNEL_FALLBACK;
    }

    if (op->op_type == MC_OP_CONV) {
        MC_PROBE_I1(10, op->rank);
        if (op->rank != 4) {
            return MC_KERNEL_FALLBACK;
        }
        MC_PROBE_I1(11, op->layout);
        if (op->layout == MC_LAYOUT_CHANNELS_LAST) {
            MC_PROBE_I1(12, op->dtype);
            if (op->dtype == MC_DTYPE_FP16) {
                return MC_KERNEL_NHWC_FP16_WINOGRAD;
            }
            return MC_KERNEL_NHWC_DIRECT;
        }
        MC_PROBE_I2(13, op->shape[2], op->shape[3]);
        if (op->shape[2] <= 3 && op->shape[3] <= 3) {
            return MC_KERNEL_IM2COL_SMALL;
        }
        return MC_KERNEL_IM2COL_GENERAL;
    }

    if (op->op_type == MC_OP_REDUCE) {
        MC_PROBE_I2(20, op->reduce_dim, op->rank);
        if (op->reduce_dim >= op->rank) {
            return MC_KERNEL_FALLBACK;
        }
        MC_PROBE_I1(21, op->shape[op->reduce_dim]);
        if (op->shape[op->reduce_dim] >= 1024) {
            return MC_KERNEL_REDUCE_TREE;
        }
        return MC_KERNEL_REDUCE_LINEAR;
    }

    if (op->op_type == MC_OP_POINTWISE) {
        MC_PROBE_I1(30, op->dtype);
        if (dtype_is_int(op->dtype)) {
            return MC_KERNEL_POINTWISE_INT;
        }
        MC_PROBE_I2(31, op->rank, op->shape[0]);
        if (op->rank >= 1 && is_pow2((uint64_t)op->shape[0])) {
            /* Aligned float pointwise gets the float kernel either way,
             * but the alignment branch is real coverage. */
            return MC_KERNEL_POINTWISE_FLOAT;
        }
        return MC_KERNEL_POINTWISE_FLOAT;
    }

    return MC_KERNEL_FALLBACK;
}

/* Fusion / scheduling -------------------------------------------- */

static int decide_fusion(const mc_op_t *op, int kernel) {
    /* Whether the kernel can be fused with a downstream pointwise op.
     * Probe IDs 100-199. */
    MC_PROBE_I2(100, kernel, op->fuse_hint);
    if (kernel == MC_KERNEL_FALLBACK || kernel == MC_KERNEL_EMPTY) {
        return 0;
    }
    if (op->fuse_hint == 0) {
        return 0;
    }
    MC_PROBE_I2(101, kernel, op->dtype);
    if (kernel == MC_KERNEL_TENSOR_CORE && op->dtype == MC_DTYPE_FP16) {
        return 1;
    }
    if (kernel == MC_KERNEL_NAIVE_GEMM) {
        return 0;  /* not worth fusing */
    }
    if (kernel == MC_KERNEL_NHWC_DIRECT || kernel == MC_KERNEL_NHWC_FP16_WINOGRAD) {
        return 1;
    }
    MC_PROBE_I3(102, kernel, op->reduce_dim, op->rank);
    if (kernel == MC_KERNEL_REDUCE_TREE && op->reduce_dim == op->rank - 1) {
        return 1;
    }
    return 0;
}

/* Codegen --------------------------------------------------------- */

static int codegen(const mc_op_t *op, int kernel, int fused, char *out_buf,
                   size_t out_cap) {
    /* "Generate code" for the kernel. We don't actually emit any code;
     * we write a tag string into out_buf so callers can verify the
     * dispatch path. The bug surface here is the buffer-handling code. */
    if (out_buf == NULL || out_cap == 0) {
        return MC_STATUS_INVALID_ARG;
    }

    const char *kname = "unknown";
    switch (kernel) {
        case MC_KERNEL_TENSOR_CORE:        kname = "tc";      break;
        case MC_KERNEL_VECTOR_HALF:        kname = "vechalf"; break;
        case MC_KERNEL_BLOCKED_GEMM:       kname = "blkgemm"; break;
        case MC_KERNEL_NAIVE_GEMM:         kname = "naive";   break;
        case MC_KERNEL_NHWC_DIRECT:        kname = "nhwc";    break;
        case MC_KERNEL_NHWC_FP16_WINOGRAD: kname = "wino16";  break;
        case MC_KERNEL_IM2COL_SMALL:       kname = "im2col_s";break;
        case MC_KERNEL_IM2COL_GENERAL:     kname = "im2col";  break;
        case MC_KERNEL_REDUCE_TREE:        kname = "redtree"; break;
        case MC_KERNEL_REDUCE_LINEAR:      kname = "redlin";  break;
        case MC_KERNEL_POINTWISE_INT:      kname = "pwint";   break;
        case MC_KERNEL_POINTWISE_FLOAT:    kname = "pwflt";   break;
        case MC_KERNEL_EMPTY:              kname = "empty";   break;
        case MC_KERNEL_FALLBACK:           kname = "fallback";break;
    }

    int n = snprintf(out_buf, out_cap, "%s%s/d%d/r%d",
                     kname, fused ? "+pw" : "", op->dtype, op->rank);
    if (n < 0 || (size_t)n >= out_cap) {
        return MC_STATUS_OUTPUT_TRUNCATED;
    }

    /* Latent bug #1: divide by zero when reduce kernel is selected
     * with a zero-extent reduce dim. select_kernel forwards through
     * if rank > reduce_dim, but doesn't guard the extent itself. */
    if (kernel == MC_KERNEL_REDUCE_LINEAR) {
        int rd = op->shape[op->reduce_dim];
        int normalized = 100 / rd;  /* boom on rd == 0 */
        (void)normalized;
    }

    /* Latent bug #2: out-of-bounds write when a NHWC fp16 winograd
     * kernel is asked to run with rank-4 shape but channel count > 64.
     * The "tile buffer" is a fixed 64 entries.
     */
    if (kernel == MC_KERNEL_NHWC_FP16_WINOGRAD && op->shape[3] > 64) {
        static int tile_buf[64];
        int acc = 0;
        for (int i = 0; i <= op->shape[3]; ++i) {
            tile_buf[i] = i;          /* OOB write on i == 64 */
            acc += tile_buf[i];        /* OOB read */
        }
        /* Force the compiler to keep the side-effect. */
        if (acc < 0) return MC_STATUS_INVALID_ARG;
    }

    /* Latent bug #3 (cross-layer): the IM2COL_GENERAL kernel assumes
     * the frontend has already normalised the layout to contiguous,
     * but the frontend's _normalize_layout only collapses
     * channels-last -> contiguous when rank != 4. For a rank-4 conv
     * with layout=STRIDED (not channels-last, not contiguous), the
     * frontend leaves the layout as STRIDED and the backend reaches
     * IM2COL_GENERAL with op->shape[2] potentially zero — which here
     * produces a signed-int shift overflow. UBSan catches it.
     *
     * Reachability requires:
     *   - frontend: rank == 4, layout == STRIDED (NOT channels-last)
     *               -> _normalize_layout leaves layout untouched
     *               -> backend dispatch to IM2COL_GENERAL because
     *                  shape[2] > 3 OR shape[3] > 3
     *   - backend: shape[2] >= 32, which exercises a shift expression
     *
     * A C-only fuzzer doesn't see the frontend layout decision; a
     * Python-only fuzzer doesn't see this kernel branch.
     */
    if (kernel == MC_KERNEL_IM2COL_GENERAL && op->layout == MC_LAYOUT_STRIDED) {
        int shift = op->shape[2];
        int probe = 1 << shift;       /* UB on shift >= 32 */
        if (probe < 0) return MC_STATUS_INVALID_ARG;
    }

    return MC_STATUS_OK;
}

/* Public entry point --------------------------------------------- */

int mc_compile(const mc_op_t *op, char *out_buf, size_t out_cap) {
    if (op == NULL) {
        return MC_STATUS_NULL_OP;
    }
    MC_PROBE_I3(200, op->op_type, op->dtype, op->rank);
    if (op->rank < 0 || op->rank > MC_MAX_RANK) {
        return MC_STATUS_INVALID_RANK;
    }
    if (!dtype_is_float(op->dtype) && !dtype_is_int(op->dtype)) {
        return MC_STATUS_INVALID_DTYPE;
    }
    /* Latent bug #3: layout field is read without bounds-checking,
     * so MC_LAYOUT_CHANNELS_LAST followed by rank==1 deref to
     * shape[2..3] in the kernel selector reads garbage. We deliberately
     * don't check (op->layout requires rank>=2). */

    int kernel = select_kernel(op);
    int fused = decide_fusion(op, kernel);
    return codegen(op, kernel, fused, out_buf, out_cap);
}

/* A small introspection helper used by the harness/tests. */
const char *mc_status_str(int status) {
    switch (status) {
        case MC_STATUS_OK:                return "ok";
        case MC_STATUS_NULL_OP:           return "null_op";
        case MC_STATUS_INVALID_RANK:      return "invalid_rank";
        case MC_STATUS_INVALID_DTYPE:     return "invalid_dtype";
        case MC_STATUS_INVALID_ARG:       return "invalid_arg";
        case MC_STATUS_OUTPUT_TRUNCATED:  return "truncated";
        default:                          return "unknown";
    }
}
