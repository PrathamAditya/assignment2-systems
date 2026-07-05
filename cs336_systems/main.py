import torch
print(torch.randint(0, 10000, (4, 512)).shape)
# from benchmarking_script import method

# method(120, 4, 2, 90, None,warmup_steps=5,steps=100)