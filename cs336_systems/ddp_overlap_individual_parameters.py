import torch
import torch.distributed as dist
import cs336_basics.data as data
import cs336_basics.nn_utils as nn_utils
import cs336_basics.model as model_module
import cs336_basics.optimizer as optimizer
from cs336_basics.toymodel import ToyModel


class DDPOIP(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        self.handles = []

        # Broadcasting initial parameters
        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

        # Register a hook in every parameter that requires a gradient
        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(self._async_all_reduce_hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)
    
    def finish_gradient_synchronization(self):
        """Blocks execution until all asynchronous all_reduce operations are complete."""
        for handle in self.handles:
            if handle is not None:
                handle.wait()
        self.handles.clear()

    def _async_all_reduce_hook(self, param: torch.Tensor):
        with torch.no_grad():
            # Divide by world_size to get the average (doing it before summing is mathematically identical)
            param.grad.div_(self.world_size)
            
            # Launch the all_reduce asynchronously so the backward pass can continue
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            
            # Store the handle so we can block/wait for it later
            self.handles.append(handle)



def get_ddp_(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles parameter broadcasting 
    and gradient synchronization for distributed data parallel training.
    """
    return DDPOIP(module)

def ddp_on_after_backward_(ddp_model):
    ddp_model.finish_gradient_synchronization()


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = ToyModel(4, 10).to(device)
# ddp_model = DDPOIP(model)
# for _ in range(10):
#     x, y = data.get_batch()
#     logits = ddp_model(x)
#     loss = model_module.cross_entropy(logits, y)
#     loss.backward()
#     ddp_model.finish_gradient_synchronization()
#     optimizer.step()
        