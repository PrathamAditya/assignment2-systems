import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

class NaiveDDP(torch.nn.Module):
    """
    Naive Distributed Data Parallel wrapper.
    Waits for the entire backward pass to finish before synchronizing gradients.
    """
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()

        with torch.no_grad():
            for param in self.module.parameters():
                dist.broadcast(param.data, src=0)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
        
    def all_reduce_grads(self):
        """
        Synchronously averages gradients across all ranks.
        Called manually after loss.backward() completes.
        """
        print("working")
        with torch.no_grad():
            for param in self.module.parameters():
                if param.requires_grad and param.grad is not None:
                    param.grad.div_(self.world_size)

                    dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

def get_ddp_(module: torch.nn.Module) -> torch.nn.Module:
    return NaiveDDP(module)


# def setup(rank, world_size):
#     os.environ["MASTER_ADDR"] = "localhost"
#     os.environ["MASTER_PORT"] = "29500"
#     torch.cuda.set_device(rank)
#     dist.init_process_group("nccl", rank=rank, world_size=world_size)


# def distributed_demo(rank, world_size, x_chunks, y_chunks, warm_up: int, reps: int):
#     setup(rank, world_size)
#     device = torch.device(f"cuda:{rank}")
#     torch.cuda.synchronize(device)
#     time_list_full_pass = []
#     time_list_collective_communication = []
    
#     x = x_chunks[rank].to(device=device)
#     y = y_chunks[rank].to(device=device)

#     d_model = 768
#     d_ff = 3072
#     num_layers = 12
#     num_heads = 12

#     base_model = model_module.BasicsTransformerLM(
#         d_model, num_layers, num_heads, d_ff, 
#         rope_theta=10_000.0, vocab_size=10000, context_length=512
#     ).to(device=device)
    
#     model = get_ddp_(base_model)
#     optimizer = torch.optim.AdamW(model.parameters())
#     model.train()

#     # --- WARM UP ---
#     for _ in range(warm_up):
#         logits = model(x)
#         loss = model_module.cross_entropy(logits, y)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
        
#         # Naive synchronization happens AFTER backward is totally done
#         model.all_reduce_grads() 
#         optimizer.step()

#     # --- BENCHMARKING ---
#     for _ in range(reps):
#         torch.cuda.synchronize()
#         start_time_full = timeit.default_timer()
        
#         logits = model(x)
#         loss = model_module.cross_entropy(logits, y)
#         optimizer.zero_grad(set_to_none=True)
#         loss.backward()
        
#         # Start timing collective communication
#         torch.cuda.synchronize()
#         start_time_c_communication = timeit.default_timer()
        
#         # Synchronous gradient aggregation
#         model.all_reduce_grads()
        
#         # End timing collective communication
#         torch.cuda.synchronize()
#         end_time_c_communication = timeit.default_timer()
        
#         optimizer.step()
        
#         torch.cuda.synchronize()
#         end_time_full = timeit.default_timer()
        
#         if rank == 0: 
#             time_list_full_pass.append(end_time_full - start_time_full)
#             time_list_collective_communication.append(end_time_c_communication - start_time_c_communication)

#     if rank == 0:
#         avg_time = statistics.mean(time_list_full_pass)
#         avg_time_c = statistics.mean(time_list_collective_communication)
#         print(f"Mean Full Time: {avg_time * 1000:.3f} ms")
#         print(f"Mean Comm Time: {avg_time_c * 1000:.3f} ms")

#     dist.destroy_process_group()


# def main():
    # B = 2
    # S = 128
    
    # x = torch.randint(0, 1000, (B, S), dtype=torch.long)
    # y = torch.randint(0, 1000, (B, S), dtype=torch.long)

    # world_size = torch.cuda.device_count()
    # x_chunks = torch.split(x, split_size_or_sections=int(B/world_size), dim=0)
    # y_chunks = torch.split(y, split_size_or_sections=int(B/world_size), dim=0)

    # mp.spawn(fn=distributed_demo, args=(world_size, x_chunks, y_chunks, 5, 10), nprocs=world_size, join=True)

# if __name__ == "__main__":
#     main()