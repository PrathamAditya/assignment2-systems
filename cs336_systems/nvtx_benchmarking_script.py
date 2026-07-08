import cs336_basics.model as model_module
import cs336_basics.nn_utils as nn_utils
import timeit
import torch
import statistics
import torch.cuda.nvtx as nvtx


def method(d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, 
           warmup_steps: int , steps: int, which_type: str,vocab_size = 10000, context_length = 512,):
    data_type = torch.long
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randint(0, vocab_size, (batch_size, context_length), dtype=data_type).to(device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long).to(device=device)
    model = model_module.BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff
                              , rope_theta).to(device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters())
    times = []
    counter = 0
    if which_type == "f":
        with nvtx.range("warmup"):
            while counter < warmup_steps:
                logits = model(x)
                counter+=1

        while counter < warmup_steps + steps:
            with nvtx.range("training step"):   
                torch.cuda.synchronize()
                time_start = timeit.default_timer()
                with nvtx.range("forward"):
                    logits = model(x)
                torch.cuda.synchronize()
                time_end = timeit.default_timer()
                times.append(time_end-time_start)
                counter+=1
    elif which_type == "fb":
        with nvtx.range("warmup"):
            while counter < warmup_steps:
                logits = model(x)
                loss = nn_utils.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                counter+=1  
        while counter < warmup_steps + steps:
            with nvtx.range("training step"): 
                torch.cuda.synchronize()
                time_start = timeit.default_timer()
                with nvtx.range("forward"):
                    logits = model(x)
                loss = nn_utils.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none = True)
                with nvtx.range("backward"):
                    loss.backward()
                torch.cuda.synchronize()
                time_end = timeit.default_timer()
                times.append(time_end-time_start)
                counter+=1
    elif which_type == "fbo":
        with nvtx.range("warmup"):
            while counter < warmup_steps:
                logits = model(x)
                loss = nn_utils.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none = True)
                loss.backward()
                optimizer.step()
                counter+=1

        while counter < warmup_steps + steps:
            with nvtx.range("training step"):     
                torch.cuda.synchronize()
                time_start = timeit.default_timer()
                with nvtx.range("forward"):
                    logits = model(x)
                loss = nn_utils.cross_entropy(logits, y)
                optimizer.zero_grad(set_to_none = True)
                with nvtx.range("backward"):
                    loss.backward()
                with nvtx.range("optimizer"):
                    optimizer.step()
                torch.cuda.synchronize()
                time_end = timeit.default_timer()
                times.append(time_end-time_start)
                counter+=1

    if len(times) == 1:
        print(f"Mean time: {times[0]:.6f} s")
        print(f"SD time:   {times[0]:.6f} s")
    else:
        print(f"Mean time: {statistics.mean(times):.6f} s")
        print(f"SD time:   {statistics.stdev(times):.6f} s")
