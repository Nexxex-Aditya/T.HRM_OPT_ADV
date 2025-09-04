import triton
import triton.language as tl
import torch

@triton.jit
def hrm_chunk_kernel(State_ptr, W_ptr, Out_ptr,
                     CHUNK_SIZE: tl.constexpr, STATE_SIZE: tl.constexpr):

    pid = tl.program_id(0)
    offs = pid * STATE_SIZE + tl.arange(0, STATE_SIZE)
    mask = offs < STATE_SIZE

    # load local state
    state = tl.load(State_ptr + offs, mask=mask, other=0.0)

    # run CHUNK_SIZE steps locally
    for _ in tl.static_range(CHUNK_SIZE):
        w = tl.load(W_ptr + offs, mask=mask, other=1.0)
        state = tl.sigmoid(state * w)

    tl.store(Out_ptr + offs, state, mask=mask)

def hrm_chunks(State, Weights, chunk_size=16):
    STATE_SIZE = State.numel()
    Out = torch.empty_like(State)
    grid = (1,)  # one kernel per chunk
    hrm_chunk_kernel[grid](State, Weights, Out,
                           chunk_size, STATE_SIZE)
    return Out
