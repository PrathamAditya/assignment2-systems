import torch
from benchmarking_script import method
import math

# Size d_model d_ff num_layers num_heads
# small 768 3072 12 12
# medium 1024 4096 24 16
# large 1280 5120 36 20
# xl 2560 10240 32 32
# 10B 4608 12288 50 36
# 128 and 2048

def main():
    w_type = "fb"
    which_data_type = None
    # method(d_model=2560, d_ff = 10240, num_layers = 32, num_heads = 32, rope_theta=None, warmup_steps=5,
    #     steps=10,which_type=w_type, context_length=120,which_data_type = which_data_type)
    method(d_model=768, d_ff = 3072, num_layers = 12, num_heads = 12, rope_theta=None, warmup_steps=5,
        steps=1,which_type=w_type, context_length=128,which_data_type = which_data_type, memory_profiling = True
        , group_size=6)
    
def code_test():
    layers = 32
    group_size = math.ceil(math.sqrt(layers))
    counter = 0
    while counter < layers:
        for g in range(group_size):
            counter+=1
            if counter <= layers: print(f"{counter}, ")
            
        print("=============")
if __name__ == "__main__":
    main()
