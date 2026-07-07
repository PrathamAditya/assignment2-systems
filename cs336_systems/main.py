# import torch
# print(torch.randint(0, 10000, (4, 512)).shape)
from benchmarking_script import method
w_type = "fbo"
method(120, 4, 2, 90, None,warmup_steps=1,steps=10,which_type=w_type)