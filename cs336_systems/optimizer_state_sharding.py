import torch
import math
from typing import Type, Any
from torch.optim import Optimizer
import torch.distributed as dist

class OptimizerStateSharding(Optimizer):

    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any): 
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.group_sharding_meta = []
        self._inner_optimizer_created = False
        super().__init__(params, kwargs)
        self.optimizer = optimizer_cls(self.param_groups)
        self._inner_optimizer_created = True

    def step(self, closure=None, **kwargs): 
        loss = self.optimizer.step(closure=closure, **kwargs)

        if self.world_size > 1:
            with torch.no_grad():
                for group_meta in self.group_sharding_meta:
                    current_offset = group_meta['offset']
                    
                    for i, param in enumerate(group_meta['global_params']):
                        src_rank = i // current_offset
                        dist.broadcast(param.data, src=src_rank)
                        
        return loss

    def add_param_group(self, param_group: dict[str, Any]):

        params_list = list(param_group['params'])
        param_count = len(params_list)
        offset = math.ceil(param_count / self.world_size)
        start_index = self.rank * offset
        end_index = min((self.rank + 1) * offset, param_count)
        
        param_group['params'] = params_list[start_index:end_index]
        super().add_param_group(param_group)

        self.group_sharding_meta.append({
            'global_params': params_list,
            'offset': offset
        })

        if getattr(self, '_inner_optimizer_created', False):
            self.optimizer.add_param_group(param_group)