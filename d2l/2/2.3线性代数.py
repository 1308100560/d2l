import torch
import numpy as np

x = torch.tensor(3.0)
y = torch.tensor(2.0)

x = torch.arange(4)

A = torch.arange(20).reshape(5, 4)

B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])

X = torch.arange(24).reshape(2, 3, 4)

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()  # 通过分配新内存，将A的一个副本分配给B

a = 2
X = torch.arange(24).reshape(2, 3, 4)

x = torch.arange(4, dtype=torch.float32)

A_sun_axis0 = A.sum(axis = 0)
A_sum_axis1 = A.sum(axis = 1)

sum_A = A.sum(axis=1, keepdims=True)

y = torch.ones(4, dtype = torch.float32)

B = torch.ones(4, 3)

u = np.array([3, -4])

breakpoint()