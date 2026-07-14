import cs336_basics.model as model_module
import cs336_basics.nn_utils as nn_utils
from cs336_basics.adamw import AdamW
import timeit
import torch
import statistics
from contextlib import nullcontext
from pathlib import Path


def method(d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, 
           warmup_steps: int , steps: int, which_type: str, which_data_type: str, memory_profiling: bool,
           group_size: int, vocab_size = 10000, context_length = 512,):
    data_type = torch.long
    batch_size = 4
    # batch_size = 1
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randint(0, vocab_size, (batch_size, context_length), dtype=data_type).to(device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), dtype=data_type).to(device=device)
    model = model_module.BasicsTransformerLM(vocab_size = vocab_size, context_length=context_length, d_model = d_model,
                                             num_layers = num_layers, num_heads=num_heads, d_ff=d_ff
                              , rope_theta=rope_theta, group_size=group_size).to(device=device)
    
    # model.train()
    ###torch.compile###
    model_opt = torch.compile(model)
    model_opt.train()
    
    ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16) if which_data_type == "b" else nullcontext())
    optimizer = AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,)
    times = []
    counter = 0
    if which_type == "f":
        while counter < warmup_steps:
            with ctx, torch.inference_mode():
                # logits = model(x)
                logits = model_opt(x)
                counter+=1
        if memory_profiling == True:
            # torch.cuda.synchronize()
            torch.cuda.memory._record_memory_history(max_entries=100000)
            with ctx, torch.inference_mode():
                # logits = model(x)
                logits = model_opt(x)
        else:
            while counter < warmup_steps + steps:
                torch.cuda.synchronize()
                time_start = timeit.default_timer()
                with ctx, torch.inference_mode():
                    # logits = model(x)
                    logits = model_opt(x)
                torch.cuda.synchronize()
                time_end = timeit.default_timer()
                times.append(time_end-time_start)
                counter+=1
    elif which_type == "fb":
        while counter < warmup_steps:
            optimizer.zero_grad(set_to_none = True)
            with ctx:
                # logits = model(x)
                logits = model_opt(x)
                loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
            counter+=1
        if memory_profiling == True:
            # torch.cuda.synchronize()
            torch.cuda.memory._record_memory_history(max_entries=100000)
            optimizer.zero_grad(set_to_none = True)
            with ctx:
                # logits = model(x)
                logits = model_opt(x)
                loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
        else:
            while counter < warmup_steps + steps:
                torch.cuda.synchronize()
                time_start = timeit.default_timer()
                optimizer.zero_grad(set_to_none = True)
                with ctx:
                    # logits = model(x)
                    logits = model_opt(x)
                    loss = nn_utils.cross_entropy(logits, y)
                loss.backward()
                torch.cuda.synchronize()
                time_end = timeit.default_timer()
                times.append(time_end-time_start)
                counter+=1
    elif which_type == "fbo":
        while counter < warmup_steps:
            optimizer.zero_grad(set_to_none = True)
            with ctx:
                # logits = model(x)
                logits = model_opt(x)
                loss = nn_utils.cross_entropy(logits, y) 
            loss.backward()
            optimizer.step()
            counter+=1
        if memory_profiling == True: 
            # torch.cuda.synchronize()
            torch.cuda.memory._record_memory_history(max_entries=100000)
            optimizer.zero_grad(set_to_none = True)
            with ctx:
                # logits = model(x)
                logits = model_opt(x)
                loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
        else:
            while counter < warmup_steps + steps:
                torch.cuda.synchronize()
                time_start = timeit.default_timer()
                optimizer.zero_grad(set_to_none = True)
                with ctx:
                    # logits = model(x)
                    logits = model_opt(x)
                    loss = nn_utils.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                time_end = timeit.default_timer()
                times.append(time_end-time_start)
                counter+=1
    torch.cuda.synchronize()

    ##For Modal##
    memory_snapshot_filename = (f"/memory/memory_snapshot_gradient_checkpointing_{d_model}_{context_length}_{batch_size}_{which_type}_{which_data_type}_{group_size}.pickle")
    if memory_profiling == True: torch.cuda.memory._dump_snapshot(memory_snapshot_filename)

    # ## FOR Local Run ##
    # memory_snapshot_filename = Path(f"../memory_profiling/memory_snapshot_{d_model}_{context_length}_{batch_size}_{which_type}_{which_data_type}.pickle")
    # memory_snapshot_filename.parent.mkdir(parents=True, exist_ok=True)
    # if memory_profiling == True: torch.cuda.memory._dump_snapshot(memory_snapshot_filename)
    # ###
    torch.cuda.memory._record_memory_history(enabled=None)
    if not memory_profiling:
        if len(times) == 1:
            print(f"Mean time: {times[0]:.6f} s")
            print(f"SD time:   {times[0]:.6f} s")
        else:
            print(f"Mean time: {statistics.mean(times):.6f} s")
            print(f"SD time:   {statistics.stdev(times):.6f} s")
