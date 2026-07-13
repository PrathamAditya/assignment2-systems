import cs336_basics.model as model_module
import cs336_basics.nn_utils as nn_utils
from cs336_basics.adamw import AdamW
import timeit
import torch
import statistics
from contextlib import nullcontext
from pathlib import Path
import torch.cuda.nvtx as nvtx

# def backward(d, s):

#     Q = torch.rand(8, s, d, requires_grad=True).to(device=device)
#     K = torch.rand(8, s, d, requires_grad=True).to(device=device)
#     V = torch.rand(8, s, d, requires_grad=True).to(device=device)
#     loss = output.sum()
#     print(loss)
#     loss.backward()
    
def forward(Q, K, V, device):
    return 

def cal_memory():
    snapshot = torch.cuda.memory.memory_snapshot()
    print(snapshot)
    # total_allocated_bytes = 0

    # for segment in snapshot.get("segments", []):
    #     for block in segment.get("blocks", []):
    #         if block.get("state") == "active_allocated":
    #             total_allocated_bytes += block.get("size", 0)

    # print(f"Total allocated memory via snapshot: {total_allocated_bytes / (1024**2):.2f} MB")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sequence_length = [256, 1024, 4096, 8192, 16384]
    d_model = [16, 32, 64, 128]

    sequence_length = [256, 1024, 4096]
    sequence_length = [256, 1024, 4096, 8192, 16384]
    d_model = [16, 32, 64, 128]
    
    # ###FIRST OOM###
    # sequence_length = [4096]
    # d_model = [128]
    time = {}

    for s_l in sequence_length:
        for d_m in d_model:
            # warm_up
            # Q = torch.rand(8, s_l, d_m, requires_grad=True).to(device=device)
            # K = torch.rand(8, s_l, d_m, requires_grad=True).to(device=device)
            # V = torch.rand(8, s_l, d_m, requires_grad=True).to(device=device)
            
            print("Warm started")
            for i in range(5):
                Q = torch.rand(8, s_l, d_m, requires_grad=True).to(device=device)
                K = torch.rand(8, s_l, d_m, requires_grad=True).to(device=device)
                V = torch.rand(8, s_l, d_m, requires_grad=True).to(device=device)
                output = model_module.scaled_dot_product_attention(Q, K , V).to(device=device)
                loss = output.sum()
                loss.backward()
            print("Warm completed")
            # time 
            torch.cuda.synchronize()
            temp_forward_time_list = []
            temp_backward_time_list = []
            for i in range(100):
                Q = torch.rand(8, s_l, d_m, requires_grad=True).to(device=device)
                K = torch.rand(8, s_l, d_m, requires_grad=True).to(device=device)
                V = torch.rand(8, s_l, d_m, requires_grad=True).to(device=device)
                # optimizer.zero_grad(set_to_none = True)
                torch.cuda.synchronize()
                time_start = timeit.default_timer()
                output = model_module.scaled_dot_product_attention(Q, K , V).to(device=device)
                loss = output.sum()
                if i == 99: print(f"{d_m}_{s_l} Forward: {torch.cuda.memory_allocated(device=device) / (1024**2): 4f}MiB")
                torch.cuda.synchronize()
                time_end = timeit.default_timer()
                temp_forward_time_list.append(time_end-time_start)
                torch.cuda.synchronize()
                time_start = timeit.default_timer()
                loss.backward()
                if i == 99: print(f"{d_m}_{s_l} Backward: {torch.cuda.memory_allocated(device=device) / (1024**2): 4f}MiB")
                torch.cuda.synchronize()
                time_end = timeit.default_timer()
                temp_backward_time_list.append(time_end-time_start)
                
            time[f"{s_l}_{d_m}_forward"]=statistics.mean(temp_forward_time_list)
            time[f"{s_l}_{d_m}_backward"]=statistics.mean(temp_backward_time_list)
            ###FOR LOCAL RUN###
            memory_snapshot_filename = Path(f"../memory_profiling/memory_snapshot_pytorch_attention_forward_{s_l}_{d_m}.pickle")
            memory_snapshot_filename.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(memory_snapshot_filename)
            torch.cuda.memory._record_memory_history(enabled=None)
            print(time_end-time_start)
    print(time)
if __name__ == "__main__":
    main()
