import torch
import time
import psutil
import os

from kernels.kernels_h.hrm_unrolled import hrm_unrolled
from kernels.kernels_h.hrm_heads import hrm_heads
from kernels.kernels_h.hrm_pipeline import hrm_pipeline
from kernels.kernels_h.hrm_cuda_graphs import hrm_with_cuda_graph
from kernels.kernels_h.hrm_multi_gpu_heads import hrm_multi_gpu_heads

def get_vram():
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024**2)

def run_naive(model, data, steps=8):
    start = time.time()
    x = data
    for _ in range(steps):
        x = model(x)
    torch.cuda.synchronize()
    return time.time() - start, x

def run_triton_unrolled(data, weights, steps=8):
    start = time.time()
    out = hrm_unrolled(data, weights, steps)
    torch.cuda.synchronize()
    return time.time() - start, out

def run_heads(data, weights, n_heads=8):
    start = time.time()
    out = hrm_heads(data, weights, n_heads)
    torch.cuda.synchronize()
    return time.time() - start, out

def run_pipeline(H_module, L_module, data):
    start = time.time()
    out = hrm_pipeline(H_module, L_module, data)
    torch.cuda.synchronize()
    return time.time() - start, out

def run_cuda_graph(model, data, steps=8):
    g, static_in, static_out = hrm_with_cuda_graph(model, data, steps)
    start = time.time()
    g.replay()
    torch.cuda.synchronize()
    return time.time() - start, static_out

def run_multi_gpu_heads(data, weights, n_heads=2):
    start = time.time()
    out = hrm_multi_gpu_heads(data, weights, n_heads)
    torch.cuda.synchronize()
    return time.time() - start, out

def bench_all():
    device = "cuda:0"
    torch.cuda.reset_peak_memory_stats(device)

    batch, dim = 32, 512
    steps = 8

    data = torch.randn(batch, dim, device=device)
    weights = torch.ones_like(data)

    # naive HRM (Pytorch)
    naive_model = torch.nn.Linear(dim, dim).to(device)
    t, out = run_naive(naive_model, data, steps)
    print(f"Naive HRM: {t*1e3:.2f} ms | VRAM {get_vram():.1f} MB")

    # triton unrolled
    t, out = run_triton_unrolled(data, weights, steps)
    print(f"Triton Unrolled: {t*1e3:.2f} ms | VRAM {get_vram():.1f} MB")

    # heads
    t, out = run_heads(data, weights, n_heads=8)
    print(f"Head-Parallel: {t*1e3:.2f} ms | VRAM {get_vram():.1f} MB")

    # pipeline (simulate H+L)
    H = torch.nn.Linear(dim, dim).to("cuda:0")
    L = torch.nn.Linear(dim, dim).to("cuda:1") if torch.cuda.device_count() > 1 else H
    t, out = run_pipeline(H, L, data)
    print(f"Pipeline HRM: {t*1e3:.2f} ms | VRAM {get_vram():.1f} MB")

    # CUDA Graphs
    t, out = run_cuda_graph(naive_model, data, steps)
    print(f"CUDA Graph HRM: {t*1e3:.2f} ms | VRAM {get_vram():.1f} MB")

    # multi GPU heads
    if torch.cuda.device_count() > 1:
        t, out = run_multi_gpu_heads(data, weights, n_heads=2)
        print(f"Multi-GPU Heads: {t*1e3:.2f} ms | VRAM {get_vram():.1f} MB")
    else:
        print("Multi-GPU Heads skipped (need >=2 GPUs)")

if __name__ == "__main__":
    bench_all()
