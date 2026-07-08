import torch
import torch.nn as nn
import cs336_basics.nn_utils as nn_utils

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()
    def forward(self, x):
        print(f"InputX: {x.dtype}")
        x = self.relu(self.fc1(x))
        print(f"After Relu: {x.dtype}")
        x = self.ln(x)
        print(f"After LayerNorm: {x.dtype}")
        x = self.fc2(x)
        print(f"After Linear: {x.dtype}")
        return x

vocab_size = 5
model = ToyModel(100, vocab_size).cuda()
x = torch.randn(2, 100, device="cuda")
y =torch.randint(low=0, high=vocab_size, size=(2,), device="cuda")
with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(x)
# print(f"Shape of logits: {logits.shape}")
# print(f"Shape of logits: {logits.dtype}")
loss = nn_utils.cross_entropy(logits, y)
loss.backward()
print(f"Loss: {loss.dtype}")
print("logits: ", logits.dtype)

print("=================================================")
for name, param in model.named_parameters():
    print(name, param.grad.dtype)
# for name, param in model.named_parameters():
#     print(name, param.grad.dtype)
# print(model.parameters())
# print(type(model.parameters()))