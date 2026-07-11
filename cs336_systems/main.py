import torch
from benchmarking_script import method

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
    method(d_model=2560, d_ff = 10240, num_layers = 32, num_heads = 32, rope_theta=None, warmup_steps=5,
        steps=10,which_type=w_type, context_length=120,which_data_type = which_data_type)

if __name__ == "__main__":
    main()
