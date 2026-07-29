import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import cs336_basics.model as model_module
import cs336_basics.nn_utils as nn_utils

class DDP(torch.nn.Module):
    """
    Custom Distributed Data Parallel wrapper that overlaps gradient 
    communication with backprop computation.
    """
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        # 1. Broadcast the initial weights from rank 0 to all other processes
        # This guarantees all GPUs start with the exact same model weights.
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # 2. Register a hook on every parameter that requires a gradient
        for param in self.module.parameters():
            if param.requires_grad:
                # This hook fires automatically during loss.backward() the moment 
                # this specific parameter's gradient is fully accumulated.
                param.register_post_accumulate_grad_hook(self._async_all_reduce_hook)

    def _async_all_reduce_hook(self, param: torch.Tensor):
        """Hook to launch an asynchronous all_reduce on the gradient."""
        with torch.no_grad():
            # Divide by world_size to get the average (doing it before summing is mathematically identical)
            param.grad.div_(self.world_size)
            
            # Launch the all_reduce asynchronously so the backward pass can continue
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            
            # Store the handle so we can block/wait for it later
            self.handles.append(handle)

    def forward(self, *args, **kwargs):
        # Simply pass through to the underlying module
        return self.module(*args, **kwargs)
        
    def wait(self):
        """Blocks execution until all asynchronous all_reduce operations are complete."""
        for handle in self.handles:
            if handle is not None:
                handle.wait()
        # Clear handles for the next training iteration
        self.handles.clear()


def get_ddp_(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles parameter broadcasting 
    and gradient synchronization for distributed data parallel training.
    """
    return DDP(module)

def ddp_on_after_backward_(ddp_model, optimizer):
    ddp_model.wait()


def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size, x_chunks, y_chunks):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    torch.cuda.synchronize(device)
    
    x = x_chunks[rank].to(device=device)
    y = y_chunks[rank].to(device=device)

    # Model size small
    d_model = 768
    d_ff = 3072
    num_layers = 12
    num_heads = 12

    # model hyperparameters
    base_model = model_module.BasicsTransformerLM(
        d_model, num_layers, num_heads, d_ff, 
        rope_theta=10_000.0, vocab_size=10000, context_length=128
    ).to(device=device)
    
    # 1. Wrap the module. (Initialization broadcast happens here)
    model = get_ddp_(base_model)
    
    optimizer = torch.optim.AdamW(model.parameters())
    model.train()
    
    logits = model(x)
    loss = model_module.cross_entropy(logits, y)
    
    optimizer.zero_grad(set_to_none=True)
    
    # 2. Backward pass! 
    # Because of our hooks, parameters closest to the output layer will start communicating
    # over the network while the GPUs are still busy crunching gradients for the input layers.
    loss.backward()
    
    # 3. Synchronize.
    # Before the optimizer steps, we must wait to ensure all network traffic has finished
    # and all gradients are fully averaged.
    ddp_on_after_backward_(model, optimizer)
    
    optimizer.step()
    
    dist.destroy_process_group()


def main():
    B = 128
    S = 128
    
    x = torch.randint(0, 1000, (B, S), dtype=torch.long)
    y = torch.randint(0, 1000, (B, S), dtype=torch.long)

    world_size = torch.cuda.device_count()
    x_chunks = torch.split(x, split_size_or_sections=int(B/world_size), dim=0)
    y_chunks = torch.split(y, split_size_or_sections=int(B/world_size), dim=0)

    mp.spawn(fn=distributed_demo, args=(world_size, x_chunks, y_chunks), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()