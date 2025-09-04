import triton
import triton.language as tl
import torch

@triton.jit
def hrm_heads_kernel(State_ptr, Weights_ptr, Out_ptr,
                     STATE_SIZE, N_HEADS: tl.constexpr, BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < STATE_SIZE

    # determine which head this block belongs to
    head_size = STATE_SIZE // N_HEADS
    head_id = pid % N_HEADS
    head_offset = head_id * head_size

    state = tl.load(State_ptr + head_offset + offs, mask=mask, other=0.0)
    w = tl.load(Weights_ptr + head_offset + offs, mask=mask, other=1.0)

    # parallel recurrence inside each head
    state = tl.relu(state * w)

    tl.store(Out_ptr + head_offset + offs, state, mask=mask)

def hrm_heads(State, Weights, n_heads=8, BLOCK_SIZE=128):
    STATE_SIZE = State.numel()
    Out = torch.empty_like(State)
    grid = (triton.cdiv(STATE_SIZE, BLOCK_SIZE),)
    hrm_heads_kernel[grid](State, Weights, Out,
                           STATE_SIZE, n_heads, BLOCK_SIZE)
    return Out
