import torch

def hrm_with_cuda_graph(model, inputs, steps=16):
    # capture HRM loop execution inside CUDA graph
    static_input = inputs.clone()
    static_output = torch.empty_like(inputs)

    g = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    with torch.cuda.graph(g, stream=stream):
        state = static_input
        for _ in range(steps):
            state = model(state)
        static_output.copy_(state)

    return g, static_input, static_output
