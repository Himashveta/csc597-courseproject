/*
 * mock_compiler.h
 *
 * Public interface to the mock compiler backend.
 */

#ifndef MOCK_COMPILER_H
#define MOCK_COMPILER_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MC_MAX_RANK 6

/* Op types. */
enum {
    MC_OP_MATMUL    = 1,
    MC_OP_CONV      = 2,
    MC_OP_REDUCE    = 3,
    MC_OP_POINTWISE = 4,
};

/* Dtypes. */
enum {
    MC_DTYPE_I8   = 1,
    MC_DTYPE_I16  = 2,
    MC_DTYPE_I32  = 3,
    MC_DTYPE_I64  = 4,
    MC_DTYPE_FP16 = 5,
    MC_DTYPE_BF16 = 6,
    MC_DTYPE_FP32 = 7,
    MC_DTYPE_FP64 = 8,
};

/* Layouts. */
enum {
    MC_LAYOUT_CONTIGUOUS    = 1,
    MC_LAYOUT_CHANNELS_LAST = 2,
    MC_LAYOUT_STRIDED       = 3,
};

/* Kernel ids returned by the selector. */
enum {
    MC_KERNEL_FALLBACK            = 0,
    MC_KERNEL_TENSOR_CORE         = 1,
    MC_KERNEL_VECTOR_HALF         = 2,
    MC_KERNEL_BLOCKED_GEMM        = 3,
    MC_KERNEL_NAIVE_GEMM          = 4,
    MC_KERNEL_NHWC_DIRECT         = 5,
    MC_KERNEL_NHWC_FP16_WINOGRAD  = 6,
    MC_KERNEL_IM2COL_SMALL        = 7,
    MC_KERNEL_IM2COL_GENERAL      = 8,
    MC_KERNEL_REDUCE_TREE         = 9,
    MC_KERNEL_REDUCE_LINEAR       = 10,
    MC_KERNEL_POINTWISE_INT       = 11,
    MC_KERNEL_POINTWISE_FLOAT     = 12,
    MC_KERNEL_EMPTY               = 13,
};

/* Status codes returned by mc_compile. */
enum {
    MC_STATUS_OK               = 0,
    MC_STATUS_NULL_OP          = -1,
    MC_STATUS_INVALID_RANK     = -2,
    MC_STATUS_INVALID_DTYPE    = -3,
    MC_STATUS_INVALID_ARG      = -4,
    MC_STATUS_OUTPUT_TRUNCATED = -5,
};

typedef struct mc_op_s {
    int op_type;
    int dtype;
    int layout;
    int rank;
    int shape[MC_MAX_RANK];
    int reduce_dim;   /* used only for MC_OP_REDUCE */
    int fuse_hint;    /* 0/1 */
} mc_op_t;

int mc_compile(const mc_op_t *op, char *out_buf, size_t out_cap);
const char *mc_status_str(int status);

#ifdef __cplusplus
}
#endif

#endif  /* MOCK_COMPILER_H */
