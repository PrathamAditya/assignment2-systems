import torch
import torch.distributed as dist


class FSDP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None):
        # triggers the initialization logic of its parent (base) class
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []
        self.compute_dtype = compute_dtype 


    def forward(self, *inputs, **kwargs): 
        NotImplementedError
    def finish_gradient_synchronization(self):
        NotImplementedError