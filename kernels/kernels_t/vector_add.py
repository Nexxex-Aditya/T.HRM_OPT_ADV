import triton
import triton.language as tl
import torch

@triton.jit
def vector_add_kernel(X_ptr, Y_ptr, Z_ptr, N,
                      BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask)
    y = tl.load(Y_ptr + offs, mask=mask)
    z = x + y
    tl.store(Z_ptr + offs, z, mask=mask)

def vector_add(X, Y, BLOCK_SIZE=1024):
    assert X.shape == Y.shape
    N = X.numel()
    Z = torch.empty_like(X)
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    vector_add_kernel[grid](X, Y, Z, N, BLOCK_SIZE)
    return Z
