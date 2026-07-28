import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import timeit
import statistics

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    torch.cuda.set_device(0)
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def distributed_demo(rank, world_size, total_elements, warm_up: int, reps: int):
    time_list = []
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    # data = torch.randint(0, 5, (total_elements,), dtype=torch.float32)
    data = torch.randint(0, 5, (total_elements,), dtype=torch.float32, device=device)

    
    for _ in range(warm_up):
        dist.all_reduce(data, async_op=False)

    for _ in range(reps):
        torch.cuda.synchronize(device)
        start_time = timeit.default_timer()
        dist.all_reduce(data, async_op=False)
        torch.cuda.synchronize(device)
        end_time = timeit.default_timer()
        if rank == 0: time_list.append(end_time-start_time)

    if rank == 0:
        size_mb = (total_elements * 4) / (1024 * 1024)
        avg_time = statistics.mean(time_list)
        print(f"Payload: {size_mb:.1f} MB | Mean Time: {avg_time * 1000:.3f} ms")

    dist.destroy_process_group()
    


def main():
    sizes_mb = [1, 10, 100, 1000]
    world_size = torch.cuda.device_count()
    warm_up = 5
    reps = 100
    for size in sizes_mb:
        total_elements = int(size) * 1024 * 1024 // 4
        mp.spawn(fn=distributed_demo, args=(world_size, total_elements, warm_up, reps), nprocs=world_size, join=True)
if __name__ == "__main__":
    main()