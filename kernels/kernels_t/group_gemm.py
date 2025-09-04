import triton
import triton.language as tl
import torch

@triton.jit
def group_gemm_kernel(A_ptr, B_ptr, C_ptr,
                      M, N, K, batch,
                      stride_ab, stride_am, stride_ak,
                      stride_bb, stride_bk, stride_bn,
                      stride_cb, stride_cm, stride_cn,
                      BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

def group_gemm(A, B, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32):
    batch, M, K = A.shape
    _, K_, N = B.shape
    assert K == K_
    C = torch.empty((batch, M, N), device='cuda', dtype=A.dtype)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), batch)
    group_gemm_kernel[grid](
        A, B, C,
        M, N, K, batch,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
        BLOCK_M, BLOCK_N, BLOCK_K
    )
    return C
