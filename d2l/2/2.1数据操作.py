import torch

x = torch.arange(12)

X = x.reshape(3, 4)

zeros = torch.zeros((2, 3, 4))

ones = torch.ones((2, 3, 4))

randn = torch.randn(3, 4)

x = torch.tensor([1.0, 2, 4, 8])
b = torch.tensor([2, 2, 2, 2])

X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

before = id(Y)
Y = Y + X

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

A = X.numpy()
B = torch.tensor(A)

a = torch.tensor([3.5])

