import torch

# 数据集
voltage = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
current = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

# 模型参数
w = torch.randn(1, requires_grad = True)
b = torch.randn(1, requires_grad = True)

# 定义模型
def model(v):
    return v * w + b

# 定义损失函数
def loss(y_pres, y_true):
    return torch.mean((y_pres - y_true) ** 2)

# 学习率
lr = 0.01

# 训练过程
for epoch in range(1000):
    # 前向传播
    y_pred = model(voltage)
    # 损失函数
    l = loss(y_pred, current)
    # 反向传播
    l.backward()
    # 更新参数
    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        # 梯度清零
        w.grad.zero_()
        b.grad.zero_()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss {l.item()}')

# 输出结果
print("学得到的参数：")
print("权重 w:", w.item())
print("偏置 b:", b.item())