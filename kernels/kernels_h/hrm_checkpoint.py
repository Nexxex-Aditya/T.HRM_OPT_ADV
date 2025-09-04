# kernels/hrm_checkpoint.py
import torch
from torch.utils.checkpoint import checkpoint

def hrm_checkpointed(fn, *args):
    # wrap HRM step in pytorch checkpointing
    # saves memory by recomputing forward pass during backward
    return checkpoint(fn, *args)
