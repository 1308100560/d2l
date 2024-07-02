import torch
from torch import nn
from d2l import torch as d2l

n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5  # 训练集数量，测试集数量，特征维度，批量大小
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05  # 生成真实权重和偏置
# 生成训练数据和测试数据
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)
def init_params():
    # 初始化模型参数
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return w, b
def l2_penalty(w):
    # 定义L2范数惩罚
    return torch.sum(w.pow(2)) / 2
def train(lambd):
    # 定义训练函数
    w, b = init_params()  # 初始化模型参数
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss  # 定义线性网络和平方损失函数
    num_epochs, lr = 100, 0.01  # 训练次数和学习率
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])  # 绘制损失曲线
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 添加L2范围惩罚项到损失函数中
            l = loss(net(X), y) + lambd * l2_penalty(w)
            l.sum().backward()  # 反向传播求梯度
            d2l.sgd([w, b], lr, batch_size)  # 使用随机梯度下降更新参数
        if (epoch + 1) % 5 == 0:
            # 记录训练和测试损失，并更新动画图表
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss),
                                     d2l.evaluate_loss(net, test_iter, loss)))
    d2l.plt.show()
    print('w的L2范数是：', torch.norm(w).item())  # 打印权重向量的L2范数

train(lambd=0)  # 忽视正则化直接训练
train(lambd=3)  # 使用权重衰减
def train_concise(wd):
    # 简洁实现
    # 创建一个包含单个线性层的神经网络
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    for param in net.parameters():
        param.data.normal_()  # 初始化参数
    # 定义损失函数为均方误差损失
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 使用SGD优化器，并制定权重衰减参数
    trainer = torch.optim.SGD([
        {"params": net[0].weight, 'weight_decay': wd},  # 对权重添加权重衰减
        {"params": net[0].bias}], lr=lr)  # 偏置参数不添加权重衰减
    # 设置动画图表，用于记录损失变化
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log', xlim=[5, num_epochs], legend=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()  # 梯度清零
            l = loss(net(X), y)  # 计算损失
            l.mean().backward()  # 反向传播计算梯度
            trainer.step()  # 更新参数
        if (epoch + 1) % 5 == 0:
            animator.add(epoch + 1,
                         (d2l.evaluate_loss(net, train_iter, loss),
                         d2l.evaluate_loss(net, test_iter, loss)))
    d2l.plt.show()
    print('w的L2范数：', net[0].weight.norm().item())  # 打印权重向量的L2范数

train_concise(0)
train_concise(3)
