import torch
import triton
import triton.language as tl
import time

@triton.jit
def layernorm_kernel(X_ptr, Y_ptr, W_ptr, B_ptr,
                     stride, N, eps: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    row = tl.program_id(0)
    X_row = X_ptr + row * stride
    Y_row = Y_ptr + row * stride

    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)

    for i in range(0, N, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_row + offs, mask=mask, other=0.)
        mean += tl.sum(x, axis=0)
        var += tl.sum(x * x, axis=0)

    mean = mean / N
    var = var / N - mean * mean
    rstd = 1 / tl.sqrt(var + eps)

    for i in range(0, N, BLOCK_SIZE):
        offs = i + tl.arange(0, BLOCK_SIZE)
        mask = offs < N
        x = tl.load(X_row + offs, mask=mask, other=0.)
        w = tl.load(W_ptr + offs, mask=mask, other=1.)
        b = tl.load(B_ptr + offs, mask=mask, other=0.)
        y = (x - mean) * rstd * w + b
        tl.store(Y_row + offs, y, mask=mask)

def layernorm_triton(X, weight, bias, eps=1e-5, BLOCK_SIZE=128):
    M, N = X.shape
    Y = torch.empty_like(X)
    grid = (M,)
    layernorm_kernel[grid](
        X, Y, weight, bias,
        X.stride(0), N,
        eps, BLOCK_SIZE
    )
    return Y
