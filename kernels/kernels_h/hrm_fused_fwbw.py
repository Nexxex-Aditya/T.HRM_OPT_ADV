import triton
import triton.language as tl
import torch

@triton.jit
def hrm_fwbw_kernel(X_ptr, W_ptr, dY_ptr, Y_ptr, dX_ptr, dW_ptr,
                    N: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # forward pass
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    w = tl.load(W_ptr + offs, mask=mask, other=1.0)
    y = tl.tanh(x * w)
    tl.store(Y_ptr + offs, y, mask=mask)

    # backward pass (dy/dx and dy/dw)
    dy = tl.load(dY_ptr + offs, mask=mask, other=0.0)
    grad = (1 - y * y) * dy
    dx = grad * w
    dw = grad * x

    tl.store(dX_ptr + offs, dx, mask=mask)
    tl.store(dW_ptr + offs, dw, mask=mask)

def hrm_fused_fwbw(X, W, dY, BLOCK_SIZE=128):
    N = X.numel()
    Y = torch.empty_like(X)
    dX = torch.empty_like(X)
    dW = torch.empty_like(W)

    grid = (triton.cdiv(N, BLOCK_SIZE),)
    hrm_fwbw_kernel[grid](X, W, dY, Y, dX, dW, N, BLOCK_SIZE)
    return Y, dX, dW
