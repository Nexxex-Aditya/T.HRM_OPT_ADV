import torch

def hrm_multi_gpu_heads(State, Weights, n_heads=2):
    # split HRM state into heads and distribute across GPUs
    # each GPU computes one subset of heads
    parts = State.chunk(n_heads, dim=-1)
    w_parts = Weights.chunk(n_heads, dim=-1)

    results = []
    for i, (s, w) in enumerate(zip(parts, w_parts)):
        device = torch.device(f"cuda:{i}")
        results.append((s.to(device) * w.to(device)).tanh().to("cuda:0"))

    return torch.cat(results, dim=-1)
