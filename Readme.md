# Triton HRM Optimization Project
***Note:- This is version 1 a simple foundational implementation for learning purposes. The next update will deliver a more polished, optimized, and plug-in-ready project that can be used in real workflows.***

## Note:- We need very specific dependency versions to work with Triton.
## You need cuda ( NVIDIA GPU ) and linux enviroment to work with Triton. For Windows users try to use with WSL2.
- Make sure WSL2 is enabled and you have Ubuntu 22.04 installed.
- NVIDIA driver with WSL support (normal game read drivers already include this).

This project provides a modular Triton kernel library and HRM specific optimizations to accelerate the Hierarchical Reasoning Model (HRM) by reducing sequential bottlenecks.

It includes:

Core Triton kernels (MatMul, Softmax, Dropout, Attention, etc.)

HRM-optimized kernels (loop unrolling, head-parallelism, speculative chunking, fused forward+backward)

Advanced integrations (CUDA Graphs, Mixed Precision + Block Scaling, Pipeline Parallelism, Checkpointing, Multi-GPU heads)

A full benchmark suite comparing naive PyTorch HRM vs Triton-accelerated HRM.

T_HRM_OPT_ADV/
│── README.md
│── requirements.txt
│── utils.py
│
├── kernels/                     # Core + HRM-specific Triton kernels
│       kernels_t/               # Core Triton kernels
│           ├── vector_add.py
│           ├── matmul.py
│           ├── group_gemm.py
│           ├── block_scaled_matmul.py
│           ├── layernorm.py
│           ├── softmax.py
│           ├── dropout.py
│           ├── attention.py
│           ├── libdevice_ops.py
│       kernels_h/                # Core HRM-specific Triton kernels
│           ├── hrm_unrolled.py
│           ├── hrm_heads.py
│           ├── hrm_chunks.py
│           ├── hrm_fused_fwbw.py
│           ├── hrm_cuda_graphs.py
│           ├── hrm_mixed_block.py
│           ├── hrm_pipeline.py
│           ├── hrm_checkpoint.py
│           ├── hrm_multi_gpu_heads.py
├── benchmarks/
│   ├── bench_hrm.py          # Benchmark suite
│
├── training_hrm_training.py      # Example training loop with something closer to HRM block design


## Baseline Kernels

- MatMul & Group GEMM
- Block-Scaled MatMul
- LayerNorm / RMSNorm
- Fused Softmax
- Low-Memory Dropout
- Fused Attention
- Vector Add (utility)
- Libdevice Ops (asin)

## HRM-Specific Optimizations
- Loop Unrolling (tl.static_range)
- Head-Parallelism (FlashRNN-style block-diagonal heads)
- Speculative Chunk Execution (parallel chunks)
- Fused Forward + Backward (custom Triton autograd)

## Advanced Integrations
- CUDA Graphs (remove Python overhead)
- Mixed Precision + Block Scaling
- Pipeline Parallelism (GPipe-style across GPUs)
- Checkpointing + Custom Backprop
- Multi-GPU Head Parallelism

Clone the repo and install dependencies: You can do it by yourself 
Running the benchmark and HRM training : If you trust this work please run it at your own risk. 🫡

## The benchmarks/bench_hrm.py script compares:

- Naive PyTorch HRM
- Triton Unrolled HRM
- Head-Parallel HRM
- Pipeline HRM
- CUDA Graph HRM
- Multi-GPU HRM (if available)

Metrics:

- Step Time (ms)
- VRAM Usage (MB)
- Tokens/sec (extendable)

## Next Steps

Extend kernels to integrate FlashAttention2 primitives

Explore HRM decoding

Add distributed benchmarks (multi node HRM)


# Built with by an confused guy 😕