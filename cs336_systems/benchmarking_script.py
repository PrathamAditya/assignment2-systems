import cs336_basics.model as model_module
import cs336_basics.nn_utils as nn_utils
import timeit
import torch


def method(d_model: int, num_layers: int, num_heads: int, d_ff: int, rope_theta: float, 
           warmup_steps: int , steps: int, which_type: str,vocab_size = 10000, context_length = 512, 
           ):
   
    # if vocab_size > 10000:
    #     data_type = torch.long
    # else:
    #     data_type = torch.int32
    data_type = torch.long
    batch_size = 4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randint(0, vocab_size, (batch_size, context_length), dtype=data_type).to(device=device)
    y = torch.randint(0, vocab_size, (batch_size, context_length), dtype=torch.long).to(device=device)
    model = model_module.BasicsTransformerLM(vocab_size, context_length, d_model, num_layers, num_heads, d_ff
                              , rope_theta).to(device=device)
    

    if which_type == "f":
        counter = 0
        while counter <= warmup_steps:
            result = model.forward(x)
            print(counter)
            counter+=1
        
        torch.cuda.synchronize()
        time_total = 0
        time__ = timeit.default_timer()
        while counter <= warmup_steps + steps:
            
            time_start = timeit.default_timer()
            result = model(x)
            result.backward()
            torch.cuda.synchronize()
            time_end = timeit.default_timer()
            time_total += (time_end-time_start)
            counter+=1
            print(counter)
            print(time_total)
        print(f"Total time: {time_total: 4f}")
    elif which_type == "fb":
        counter = 0
        while counter <= warmup_steps:
            result = model.forward(x)
            print(counter)
            counter+=1
        
        torch.cuda.synchronize()
        time_total = 0
        # time__ = timeit.default_timer()
        while counter <= warmup_steps + steps:
            
            time_start = timeit.default_timer()
            logits = model(x)
            loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
            # result.backward()
            torch.cuda.synchronize()
            time_end = timeit.default_timer()
            time_total += (time_end-time_start)
            counter+=1
            print(counter)
            print(time_total)
        print(f"Total time: {time_total: 4f}")
    elif which_type == "fbo":
        counter = 0
        while counter <= warmup_steps:
            result = model.forward(x)
            print(counter)
            counter+=1
        
        torch.cuda.synchronize()
        time_total = 0
        # time__ = timeit.default_timer()
        while counter <= warmup_steps + steps:
            
            time_start = timeit.default_timer()
            logits = model(x)
            loss = nn_utils.cross_entropy(logits, y)
            loss.backward()
            # result.backward()
            torch.cuda.synchronize()
            time_end = timeit.default_timer()
            time_total += (time_end-time_start)
            counter+=1
            print(counter)
            print(time_total)
        print(f"Total time: {time_total: 4f}")

        

# inialize model with hyperparameters
# generate rand batch with torch.rand
# warm
# cuda sync and measure time
