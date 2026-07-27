import triton
import math
import torch
import timeit
from cs336_systems.flash_attention_2_pytroch import FlashAttention2Pytorch
from cs336_systems.flash_attention_2_triton import FlashAttention2Triton

def main(name: str):
    # name = input("Forward: 1, Backward: 2 or Full: 3? ")
    output_dict = {}
    print(f"Your option is: {name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    precision = [torch.bfloat16, torch.float32]
    # precision = [torch.bfloat16]
    embedding_dimensions = [16]
    sequence_lengths = [128]
    # embedding_dimensions = [16, 32, 64, 128]
    # sequence_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]

    for typ in precision:
        for emb_dim in embedding_dimensions:
            for seq_length in sequence_lengths:
                print("###########################################################")
                key = f"Type: {typ}, emb_dim: {emb_dim}, seq_length: {seq_length}"
                print(key)
                dtype = typ
                shape = (1, seq_length, emb_dim)
                q_tensor = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
                k_tensor = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)
                v_tensor = torch.randn(shape, dtype=dtype, device=device, requires_grad=True)

                is_casual = True

                def ForwardPytorch():
                    print(FlashAttention2Pytorch.apply(q_tensor, k_tensor, v_tensor, is_casual))

                def ForwardTriton():
                    print(FlashAttention2Triton.apply(q_tensor, k_tensor, v_tensor, is_casual))

                y_pytorch = FlashAttention2Pytorch.apply(q_tensor, k_tensor, v_tensor, is_casual)
                loss_pytorch = y_pytorch.sum()

                y_triton = FlashAttention2Triton.apply(q_tensor, k_tensor, v_tensor, is_casual)
                loss_triton = y_triton.sum()

                def BenchmarkBackwardPytorch():
                    loss_pytorch.backward(retain_graph=False)
                    q_tensor.grad = None
                    k_tensor.grad = None
                    v_tensor.grad = None

                def BenchmarkBackwardTriton():
                    loss_triton.backward(retain_graph=False)
                    q_tensor.grad = None
                    k_tensor.grad = None
                    v_tensor.grad = None

                    # Full − Forward
                    # =
                    # ZeroGrad
                    # +
                    # Loss
                    # +
                    # Backward
                    # +
                    # OptimizerStep

                optimizer = torch.optim.AdamW([q_tensor, k_tensor, v_tensor], lr=1e-3)

                def FullPassPytorch():
                    optimizer.zero_grad()
                    y = FlashAttention2Pytorch.apply(q_tensor, k_tensor, v_tensor, is_casual)
                    loss = y.sum() 
                    loss.backward()
                    optimizer.step()
                    # return loss.item()

                def FullPassTriton():
                    optimizer.zero_grad()
                    y = FlashAttention2Triton.apply(q_tensor, k_tensor, v_tensor, is_casual)
                    loss = y.sum() 
                    loss.backward()
                    optimizer.step()
                    # return loss.item()

                def ForwardPytorchReturn():
                    return FlashAttention2Pytorch.apply(q_tensor, k_tensor, v_tensor, is_casual)

                def ForwardTritonReturn():
                    return FlashAttention2Triton.apply(q_tensor, k_tensor, v_tensor, is_casual)

                warm_up = 5
                rep = 10
                grad_to_none = None
                return_mode = "mean"
                value = ""
                if name == "1":
                    f_pytorch = triton.testing.do_bench(fn=ForwardPytorch, rep=rep, warmup=warm_up, 
                                                        grad_to_none=grad_to_none, return_mode=return_mode)
                    f_triton = triton.testing.do_bench(fn=ForwardTriton, rep=rep, warmup=warm_up, 
                                                       grad_to_none=grad_to_none, return_mode=return_mode)
                    value = f"f_pytorch: {f_pytorch} and f_triton: {f_triton}"
                    
                if name == "2":
                    b_pytorch = triton.testing.do_bench(fn=BenchmarkBackwardPytorch, rep=rep, warmup=warm_up, 
                                                        grad_to_none=grad_to_none, return_mode=return_mode)
                    b_triton = triton.testing.do_bench(fn=BenchmarkBackwardTriton, rep=rep, warmup=warm_up, 
                                                        grad_to_none=grad_to_none, return_mode=return_mode)
                    value = f"f_pytorch: {b_pytorch} and f_triton: {b_triton}"
                
                if name == "3":
                    fnb_pytorch = triton.testing.do_bench(fn=FullPassPytorch, rep=rep, warmup=warm_up, 
                                                          grad_to_none=grad_to_none, return_mode=return_mode)
                    fnb_triton = triton.testing.do_bench(fn=FullPassTriton, rep=rep, warmup=warm_up, 
                                                            grad_to_none=grad_to_none, return_mode=return_mode)
                    value = f"f_pytorch: {fnb_pytorch} and f_triton: {fnb_triton}"

                if name == "4":
                    O_pytorch = ForwardPytorch()
                    O_triton = ForwardTriton()

                    # 2. Check maximum absolute difference
                    max_diff = torch.max(torch.abs(O_pytorch - O_triton)).item()
                    print(f"Max absolute difference: {max_diff:.6f}")

                    # 3. Check element-wise closeness
                    # Note: standard tolerances for bfloat16 are rtol=1e-2, atol=1e-2
                    # for float16: rtol=1e-3, atol=1e-3
                    is_matching = torch.allclose(O_pytorch, O_triton, rtol=1e-2, atol=1e-2)
                    print(f"Outputs match: {is_matching}")

                    if not is_matching:
                        raise ValueError(f"Mismatch detected! Max difference is {max_diff}")

                print(value)
                output_dict[key] = value
    return output_dict
if __name__ == "__main__":
    main()