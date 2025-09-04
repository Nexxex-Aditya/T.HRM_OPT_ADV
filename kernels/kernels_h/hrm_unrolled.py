import triton
import triton.language as tl
import torch

@triton.jit
def hrm_unrolled_kernel(State_ptr, Weights_ptr, Out_ptr,
                        N_STEPS: tl.constexpr, STATE_SIZE: tl.constexpr,
                        BLOCK_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < STATE_SIZE

    # load initial state
    state = tl.load(State_ptr + offs, mask=mask, other=0.0)

    # unroll multiple reasoning steps
    for _ in tl.static_range(N_STEPS):
        w = tl.load(Weights_ptr + offs, mask=mask, other=1.0)
        state = tl.tanh(state * w)   # replace with HRM update

    tl.store(Out_ptr + offs, state, mask=mask)

def hrm_unrolled(State, Weights, n_steps=8, BLOCK_SIZE=128):
    Out = torch.empty_like(State)
    STATE_SIZE = State.numel()
    grid = (triton.cdiv(STATE_SIZE, BLOCK_SIZE),)
    hrm_unrolled_kernel[grid](State, Weights, Out,
                              n_steps, STATE_SIZE, BLOCK_SIZE)
    return Out
