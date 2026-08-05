import torch
import math
from typing import Type, Any
from torch.optim import Optimizer
import torch.distributed as dist

class OptimizerStateSharding(Optimizer):

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any): 
        # 1. Distributed Setup
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.group_sharding_meta = []
        self._inner_optimizer_created = False
        
        # 2. Call PyTorch's native Optimizer constructor.
        # This will automatically format the params, apply the kwargs as defaults, 
        # and call our overridden `add_param_group()` for every group in `params`.
        super().__init__(params, kwargs)
        
        # 3. By this point, `super().__init__` has populated `self.param_groups`
        # with our newly sharded parameters. We can now initialize the inner 
        # Adam/SGD optimizer by directly feeding it these configured groups!
        self.optimizer = optimizer_cls(self.param_groups)
        
        # Flag that the inner optimizer exists so future dynamic param groups know to update it
        self._inner_optimizer_created = True

    def step(self, closure=None, **kwargs): 
        # 1. Perform local step
        loss = self.optimizer.step(closure=closure, **kwargs)

        # 2. Synchronize all groups
        if self.world_size > 1:
            with torch.no_grad():
                for group_meta in self.group_sharding_meta:
                    current_offset = group_meta['offset']
                    
                    for i, param in enumerate(group_meta['global_params']):
                        src_rank = i // current_offset
                        dist.broadcast(param.data, src=src_rank)
                        
        return loss

    def add_param_group(self, param_group: dict[str, Any]):
        # 1. Lock length and deterministic order
        params_list = list(param_group['params'])
        param_count = len(params_list)
        
        # 2. Calculate sharding math specifically for THIS group
        offset = math.ceil(param_count / self.world_size)
        start_index = self.rank * offset
        end_index = min((self.rank + 1) * offset, param_count)
        
        # 3. Overwrite the 'params' key with only the local subset
        param_group['params'] = params_list[start_index:end_index]
        
        # 4. Let the PyTorch superclass handle validation and append to self.param_groups
        super().add_param_group(param_group)
        
        # 5. Track global metadata for the step() broadcast loop
        self.group_sharding_meta.append({
            'global_params': params_list,
            'offset': offset
        })
        
        # 6. If a group is dynamically added AFTER __init__ has finished, 
        # we must explicitly pass it down to the inner optimizer as well.
        if getattr(self, '_inner_optimizer_created', False):
            self.optimizer.add_param_group(param_group)