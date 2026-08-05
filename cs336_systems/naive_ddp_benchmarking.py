import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import statistics
import cs336_basics.model as model_module
import cs336_basics.nn_utils as nn_utils
import cs336_systems.naive_ddp as naive_ddp
import torch.cuda.nvtx as nvtx

TIME_LISTS = {}

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size, x_chunks, y_chunks, warm_up: int, reps: int):
    setup(rank, world_size)
    print(rank)
    device = torch.device(f"cuda:{rank}")
    time_list_full_pass = []
    time_list_collective_communication = []
    x = x_chunks[rank].to(device=device)
    y = y_chunks[rank].to(device=device)

    # Model size small
    # d_model = 768
    # d_ff = 3072
    # num_layers = 12
    # num_heads = 12

    # Model size XL   
    d_model = 2560
    d_ff = 10240
    num_layers = 32
    num_heads = 32

#  vocab_size: int,
#         context_length: int,
#         d_model: int,
#         num_layers: int,
#         num_heads: int,
#         d_ff: int,
#         group_size: int,
#         rope_theta: float | None = 10_000.0,
    # model hyperparameters
    base_model = model_module.BasicsTransformerLM(
        d_model=d_model, num_layers=num_layers, num_heads=num_heads, d_ff=d_ff, 
        rope_theta=10_000.0, vocab_size=10000, context_length=512,group_size=world_size
    ).to(device=device)
    
    model = naive_ddp.get_ddp_(base_model)
    optimizer = torch.optim.AdamW(model.parameters())
    model.train()
    # print(type(model))
    # print(hasattr(model, "all_reduce_grads"))
    # return
    with nvtx.range("Warmup"):
        for _ in range(warm_up):
            logits = model(x)
            loss = nn_utils.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            model.all_reduce_grads()
            optimizer.step()
    for _ in range(reps):
        with nvtx.range("Training Step"):
            torch.cuda.synchronize(device)
            start_time_full = timeit.default_timer()
            logits = model(x)
            loss = nn_utils.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.cuda.synchronize(device)
            start_time_c_communication = timeit.default_timer()
            with nvtx.range("Reduce All"):
                model.all_reduce_grads()
            torch.cuda.synchronize(device)
            end_time_c_communication = timeit.default_timer()
            optimizer.step()
            torch.cuda.synchronize(device)
            end_time_full = timeit.default_timer()
            time_list_full_pass.append(end_time_full-start_time_full)
            time_list_collective_communication.append(end_time_c_communication - start_time_c_communication)

    print(f"Stats for: {rank} ###################")
    avg_time = statistics.mean(time_list_full_pass)
    avg_time_c = statistics.mean(time_list_collective_communication)
    print(f"Mean Time: {avg_time * 1000:.3f} ms")
    print(f"Mean Time CC: {avg_time_c * 1000:.3f} ms")
    print(f"Ratio: {(avg_time_c/avg_time)}")
    print("########################################")
    dist.destroy_process_group()

def main():
    B = 4
    S = 512
    
    x = torch.randint(0, 1000, (B, S), dtype=torch.long)
    y = torch.randint(0, 1000, (B, S), dtype=torch.long)

    world_size = torch.cuda.device_count()
    x_chunks = torch.split(x, split_size_or_sections=int(B/world_size), dim=0)
    y_chunks = torch.split(y, split_size_or_sections=int(B/world_size), dim=0)
    mp.spawn(fn=distributed_demo, args=(world_size, x_chunks, y_chunks, 5, 10), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()