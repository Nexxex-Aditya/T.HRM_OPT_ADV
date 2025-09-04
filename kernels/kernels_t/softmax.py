import torch
import triton
import triton.language as tl
import time

@triton.jit
def softmax_kernel(X_ptr, Y_ptr,
                   stride_xm, stride_xn,
                   stride_ym, stride_yn,
                   M, N, BLOCK_SIZE: tl.constexpr):

    row = tl.program_id(0)
    x_ptrs = X_ptr + row * stride_xm + tl.arange(0, BLOCK_SIZE) * stride_xn
    y_ptrs = Y_ptr + row * stride_ym + tl.arange(0, BLOCK_SIZE) * stride_yn

    x = tl.load(x_ptrs, mask=tl.arange(0, BLOCK_SIZE) < N, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    exp_sum = tl.sum(exp_x, axis=0)
    y = exp_x / exp_sum
    tl.store(y_ptrs, y, mask=tl.arange(0, BLOCK_SIZE) < N)

def softmax_triton(X, BLOCK_SIZE=128):
    M, N = X.shape
    Y = torch.empty_like(X)
    grid = (M,)
    softmax_kernel[grid](
        X, Y,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        M, N, BLOCK_SIZE
    )
    return Y
