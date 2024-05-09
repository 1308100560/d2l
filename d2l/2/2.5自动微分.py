import torch

x = torch.arange(4.0)

x.requires_grad_(True) # 等价于x = torch.arange(4.0, requires_grad = True)
x.grad # 默认值是None

y = 2 * torch.dot(x, x)

y.backward()

# 在默认情况下，PyTorch会累积梯度，我们需要清除之前的值
x.grad.zero_()
y = x.sum()
y.backward()

# 对非标量调用backward需要传入一个gradient参数，该参数指定微分函数关于自变量的梯度
# 本例只想求偏导数的和，所以传入一个全1的梯度
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()

x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()

x.grad.zero_()
y.sum().backward()

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size = (), requires_grad = True)
d = f(a)
d.backward()


breakpoint()