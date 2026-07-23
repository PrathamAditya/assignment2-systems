import torch
import math
import triton

@triton.jit
def flash_attention_forward_kernel(
    # Tensor pointers
    Q_ptr, K_ptr, V_ptr, O_ptr, L_ptr

    # Tensor metadata
    strides,

    # Tile sizes / compile-time constants
    bq, bk):
    return

class FlashAttention2Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V: torch.tensor, is_causal=False):
        _bq = 16
        _bk = 16

        if K.shape != V.shape:
            raise "K and V shape should match."
        
        B, S, D = Q.shape
        O = torch.empty(B, S, D)
        L = torch.empty(B, S)

        # This assignment implements self-attention, so Q, K, and V all have the same sequence length
        Nq = S   
        Nk = S
        d  = D

        Tq = math.ceil(Nq/_bq)
        Tk = math.ceil(Nk/_bk)