M, N, K = 512, 512, 512
A = torch.randn((M, K), device='cuda', dtype=torch.float16)
B = torch.randn((K, N), device='cuda', dtype=torch.float16)

triton_time = benchmark(matmul_triton, A, B)
torch_time = benchmark(lambda A, B: torch.matmul(A, B), A, B)

print(f"Triton MatMul: {triton_time*1e3:.3f} ms, Torch MatMul: {torch_time*1e3:.3f} ms")
