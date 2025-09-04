import torch

def hrm_pipeline(H_module, L_module, x, microbatches=4):
    # Simple GPipe like pipeline: split input into microbatches,
    # run H and L on different devices/streams.

    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")

    micro_chunks = x.chunk(microbatches, dim=0)
    outputs = []

    stream0 = torch.cuda.Stream(device=device0)
    stream1 = torch.cuda.Stream(device=device1)

    for chunk in micro_chunks:
        with torch.cuda.stream(stream0):
            h_out = H_module(chunk.to(device0))
        with torch.cuda.stream(stream1):
            l_out = L_module(h_out.to(device1))
        outputs.append(l_out)

    torch.cuda.synchronize()
    return torch.cat(outputs, dim=0)
