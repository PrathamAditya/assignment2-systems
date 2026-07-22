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
    w_type = "f"
    which_data_type = None
    # method(d_model=2560, d_ff = 10240, num_layers = 32, num_heads = 32, rope_theta=None, warmup_steps=5,
    #     steps=10,which_type=w_type, context_length=120,which_data_type = which_data_type)
    method(d_model=768, d_ff = 3072, num_layers = 12, num_heads = 12, rope_theta=None, warmup_steps=5,
        steps=1,which_type=w_type, context_length=256,which_data_type = which_data_type, memory_profiling = False, group_size=1)
    
def code_test():
    layers = 32
    group_size = math.ceil(math.sqrt(layers))
    counter = 0
    while counter < layers:
        for g in range(group_size):
            counter+=1
            if counter <= layers: print(f"{counter}, ")
            
        print("=============")

def tensor_broadcasting():
    Bq = 4
    d = 3

    # O has shape (Bq, d)
    O = torch.tensor([
        [1., 2., 3.],
        [4., 5., 6.],
        [7., 8., 9.],
        [10., 11., 12.]
    ])

    # One scale factor per row
    scale = torch.tensor([10., 20., 30., 40.])

    print("O shape:", O.shape)
    print(O)

    print("\nscale shape:", scale.shape)
    print(scale)

    # View as (Bq, 1)
    scale_col = scale[:, None]

    print("\nscale_col shape:", scale_col.shape)
    print(scale_col)

    # Broadcast multiplication
    result = scale_col * O

    print("\nResult shape:", result.shape)
    print(result)
    
if __name__ == "__main__":
    tensor_broadcasting()
