import torch
from kernels_t.block_scaled_matmul import block_scaled_matmul

def hrm_mixed_block(State, Weights, Scales):
    State_fp16 = State.half()
    W_fp16 = Weights.half()
    Out = block_scaled_matmul(State_fp16, W_fp16, Scales)
    return Out.float()  # back to FP32 for stability
