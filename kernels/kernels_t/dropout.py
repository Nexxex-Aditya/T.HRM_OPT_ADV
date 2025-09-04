import triton
import triton.language as tl
import torch

@triton.jit
def dropout_kernel(X_ptr, Y_ptr, mask_ptr,
                   N_ELEMENTS, p: tl.constexpr, seed: tl.constexpr,
                   BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEMENTS

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)

    # Philox-based RNG (stateless)
    random = tl.rand(seed, offs)

    keep = random > p
    y = x * keep / (1 - p)

    tl.store(Y_ptr + offs, y, mask=mask)
    tl.store(mask_ptr + offs, keep, mask=mask)

def dropout_triton(X, p=0.5, seed=0, BLOCK_SIZE=1024):
    Y = torch.empty_like(X)
    mask = torch.empty_like(X, dtype=torch.int32)
    N = X.numel()
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    dropout_kernel[grid](X, Y, mask, N, p, seed, BLOCK_SIZE)
    return Y, mask
