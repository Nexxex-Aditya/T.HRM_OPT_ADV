import triton
import triton.language as tl
import torch

@triton.jit
def fused_attention_kernel(Q_ptr, K_ptr, V_ptr, Out_ptr,
                           stride_qm, stride_qk,
                           stride_km, stride_kn,
                           stride_vm, stride_vn,
                           stride_om, stride_on,
                           M, N, D,
                           BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr):

    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    q_ptrs = Q_ptr + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    k_ptrs = K_ptr + offs_n[None, :] * stride_kn + offs_d[:, None] * stride_km
    v_ptrs = V_ptr + offs_n[:, None] * stride_vm + offs_d[None, :] * stride_vn

    q = tl.load(q_ptrs, mask=offs_m[:, None] < M, other=0.0)
    k = tl.load(k_ptrs, mask=offs_n[None, :] < N, other=0.0)

    # QK^T scaled
    scores = tl.dot(q, tl.trans(k)) / tl.sqrt(float(D))

    # Softmax along N
    scores = scores - tl.max(scores, axis=1)[:, None]
    exp_scores = tl.exp(scores)
    denom = tl.sum(exp_scores, axis=1)[:, None]
    probs = exp_scores / denom

    v = tl.load(v_ptrs, mask=offs_n[:, None] < N, other=0.0)
    out = tl.dot(probs, v)

    o_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_d[None, :] * stride_on
    tl.store(o_ptrs, out, mask=offs_m[:, None] < M)

def fused_attention(Q, K, V, BLOCK_M=64, BLOCK_N=64, BLOCK_D=32):
    M, D = Q.shape
    N, D_ = K.shape
    assert D == D_
    O = torch.empty((M, D), device='cuda', dtype=Q.dtype)

    grid = (triton.cdiv(M, BLOCK_M),)
    fused_attention_kernel[grid](
        Q, K, V, O,
        Q.stride(0), Q.stride(1),
        K.stride(0), K.stride(1),
        V.stride(0), V.stride(1),
        O.stride(0), O.stride(1),
        M, N, D,
        BLOCK_M, BLOCK_N, BLOCK_D
    )
    return O
