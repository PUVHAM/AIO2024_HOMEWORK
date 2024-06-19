import torch
import torch.nn as nn

class Softmax(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        exp_x = torch.exp(x)
        sum_exp_x = torch.sum(exp_x)
        return exp_x / sum_exp_x

class SoftmaxStable(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        c = max(x)
        exp_x = torch.exp(x - c)
        sum_exp_x = torch.sum(exp_x)
        return exp_x / sum_exp_x

#Testcases
data = torch.Tensor([1, 2, 3])
softmax = Softmax()
output = softmax(data)
print(output)

softmax_stable = SoftmaxStable()
output = softmax_stable(data)
print(output)