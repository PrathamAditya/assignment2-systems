# import torch
# from torch import nn

# x = torch.randn((4, 512, 2560), requires_grad=True)

# class RMSNorm(nn.Module):
#     def __init__(
#     self,
#     hidden_size: int,
#     eps: float = 1e-5,
#     device=None,
#     ):
#         super().__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
#         self.eps = eps
#     def forward(self, x):
#         rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
#         x = x * rms
#         return self.weight * x
# def pack_hook(t):
#     shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
#     print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
#     return t
# def unpack_hook(t):
#     shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
#     print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
#     return t
    
# ln = torch.compile(RMSNorm(x.shape[-1]))

# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     y = ln(x)
#     y.sum().backward()

##################################################################################################################################3

import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock

# num_layers for this model is 32
d_model = 2560
d_ff = 10240
num_heads = 16
context_length = 2048

block = TransformerBlock(
    d_model=d_model,
    d_ff=d_ff,
    num_heads=num_heads,
    positional_encoder=RotaryEmbedding(
        dim=d_model // num_heads,
        context_length=context_length,
    ),
)

# Fuse as much as torch.compile will allow
# block = torch.compile(block, fullgraph=True)

x = torch.randn(
    (4, context_length, d_model),
    requires_grad=True,
)

# Track total memory saved for backward
total_size_bytes = 0


def pack_hook(t):
    # Skip parameters to avoid double counting parameter memory
    if isinstance(t, torch.nn.Parameter):
        return t

    global total_size_bytes

    shape = t.shape
    dtype = t.dtype
    grad_fn = t.grad_fn

    total_size_bytes += t.numel() * t.element_size()

    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")

    return t


def unpack_hook(t):
    shape = t.shape
    dtype = t.dtype
    grad_fn = t.grad_fn

    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")

    return t


# Run forward pass while tracking tensors saved for backward
# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     y = block(x)
# print(
#     f"Total size of saved tensors in single TransformerBlock: "
#     f"{total_size_bytes / (1024 ** 2):.2f} MiB"
# )

# def four_blocks(x):
#     x = block(x)
#     x = block(x)
#     x = block(x)
#     x = block(x)
#     return x
# with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
#     y = four_blocks(x)
# print(f"Total size of saved tensors in four TransformerBlocks: {total_size_bytes /
# (1024**2):.2f} MiB")

from torch.utils.checkpoint import checkpoint
def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
# checkpoint throws out all the saved tensors until the backward pass
# when getting to the checkpointed block in the backward pass,
# it reruns a forward pass to produce the saved tensors,
# then completes normal backward pass.
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)
print(f"Total size of saved tensors in four TransformerBlocks with checkpointing:{total_size_bytes / (1024**2):.2f} MiB")