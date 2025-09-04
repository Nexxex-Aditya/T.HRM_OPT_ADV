import triton
import triton.language as tl
import torch

@triton.jit
def asin_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.libdevice.asin(x)
    tl.store(Y_ptr + offs, y, mask=mask)

def asin_triton(X, BLOCK_SIZE=1024):
    Y = torch.empty_like(X)
    N = X.numel()
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    asin_kernel[grid](X, Y, N, BLOCK_SIZE)
    return Y
