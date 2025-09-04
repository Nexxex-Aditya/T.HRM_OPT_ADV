import triton
import triton.language as tl
import torch

@triton.jit
def block_scaled_matmul_kernel(A_ptr, B_ptr, Scales_ptr, C_ptr,
                               M, N, K,
                               stride_am, stride_ak,
                               stride_bk, stride_bn,
                               stride_cm, stride_cn,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=offs_m[:, None] < M, other=0.)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=offs_n[None, :] < N, other=0.)
        scale = tl.load(Scales_ptr + k // BLOCK_K)
        acc += tl.dot(a, b) * scale

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def block_scaled_matmul(A, B, scales, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    M, K = A.shape
    K_, N = B.shape
    assert K == K_
    C = torch.empty((M, N), device='cuda', dtype=A.dtype)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    block_scaled_matmul_kernel[grid](
        A, B, scales, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C
