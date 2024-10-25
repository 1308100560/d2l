import torch
from torch import nn

print(torch.device('cpu'), torch.device('npu'), torch.device('npu:1'))
print(torch.npu.device_count())

def try_npu(i=0):  #@save
    """如果存在，则返回npu(i)，否则返回cpu()"""
    if torch.npu.device_count() >= i + 1:
        return torch.device(f'npu:{i}')
    return torch.device('cpu')

def try_all_npus():  #@save
    """返回所有可用的NPU，如果没有NPU，则返回[cpu(),]"""
    devices = [torch.device(f'npu:{i}') for i in range(torch.npu.device_count())]
    return devices if devices else [torch.device('cpu')]

# 测试
print(try_npu())
print(try_npu(10))
print(try_all_npus())

x = torch.tensor([1, 2, 3])
print(x.device)
X = torch.ones(2, 3, device=try_npu())
print(X)
Y = torch.rand(2, 3, device=try_npu(1))
print(Y)
Z = X.npu(1)
print(X)
print(Z)
print(Y + Z)
print(Z.npu(1) is Z)

net = nn.Sequential(nn.Linear(3, 1))
net = net.to(device=try_npu())
print(net(X))
print(net[0].weight.data.device)

breakpoint()
