# demo_hrm_training.py
import torch
import torch.nn as nn

from kernels.kernels_h.hrm_unrolled import hrm_unrolled
from kernels.kernels_h.hrm_heads import hrm_heads
from kernels.kernels_h.hrm_fused_fwbw import hrm_fused_fwbw
from kernels.kernels_h.hrm_checkpoint import hrm_checkpointed
from kernels.kernels_h.hrm_pipeline import hrm_pipeline

# kernels/kernels_h/hrm_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# real HRM Step block (High-level + Low-level reasoning)
class HRMStep(nn.Module):
    def __init__(self, dim, n_heads=4, ff_mult=4):
        super().__init__()
        self.dim = dim

        # high level reasoning (controller)
        self.h_norm = nn.LayerNorm(dim)
        self.h_attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True)

        # low level reasoning (executor)
        self.l_norm = nn.LayerNorm(dim)
        self.l_ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            nn.GELU(),
            nn.Linear(ff_mult * dim, dim)
        )

        # gating between H and L
        self.gate = nn.Parameter(torch.zeros(1, dim))

    def forward(self, x):
        # high level reasoning
        h_in = self.h_norm(x)
        h_out, _ = self.h_attn(h_in, h_in, h_in)

        # low level reasoning
        l_in = self.l_norm(h_out + x)
        l_out = self.l_ff(l_in)

        # gated update
        gate = torch.sigmoid(self.gate)
        return x + gate * (h_out + l_out)

# HRM model with unrolled + checkpointed steps
class HRMModel(nn.Module):
    def __init__(self, dim, steps=8):
        super().__init__()
        self.step = HRMStep(dim)
        self.steps = steps

    def forward(self, x):
        for _ in range(self.steps):
            x = hrm_checkpointed(self.step, x)  # memory efficient unrolled
        return x

# training 
def train_demo():
    device = "cuda:0"
    model = HRMModel(dim=512, steps=8).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    data = torch.randn(32, 512, device=device)
    target = torch.randn(32, 512, device=device)

    for step in range(5):
        optim.zero_grad()
        out = model(data)
        loss = ((out - target) ** 2).mean()
        loss.backward()
        optim.step()
        print(f"Step {step}: loss={loss.item():.4f}")

if __name__ == "__main__":
    train_demo()