import torch
from IPython import display  # display通常用于在 IPython 环境下显示丰富的输出，比如图像、HTML、音频等。使用这个命令可以在 IPython 环境下更加方便地展示各种类型的内容
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size = 256  # 每批次读256个图片
# win环境将d2l加载数据的num-workers改成0
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 初始化参数模型
# softmax回归输入需要是一个向量，1*28*28拉长成一个向量，损失掉很多空间信息，留给卷积网络处理
num_inputs = 784  # softmax网络输入
num_outputs = 10
# torch.normal(均值，方差，形状，是否需要梯度)
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) # 权重
b = torch.zeros(num_outputs, requires_grad=True)  # 偏移 对每一个输出都要有一个偏移，所以b是长度为10的向量

# 测试数据的不降维求和
# batch_size=2 两个测试数据
# 回顾：对于矩阵，可以按照行或列指定轴求和
X = torch.tensor([[1.0, 2., 3.],
                  [4., 5., 6.]])
print(X.sum(0, keepdim=True))
print(X.sum(1, keepdim=True))

# # 定义Softmax操作
# 实现softmax由三个步骤组成：
# 1.对每个项求幂（使用exp）；
# 2.对每一行求和（小批量中每个样本是一行），得到每个样本的规范化常数；
# 3.将每一行除以其规范化常数，确保结果的和为1。
# X 矩阵 对矩阵来讲，是对每一行做softmax
def softmax(X):
    X_exp = torch.exp(X)  # 对每个元素做指数计算 保证每个元素非负
    partition = X_exp.sum(1, keepdim=True) # 按照维度1求和 对每一行求和
    return X_exp/partition # 应用了广播机制  每一行的元素都除以每一行的和 保证和为1

X = torch.normal(0, 1, (2,5)) # 创建正态分布两行五列的矩阵
print(X)
X_prop = softmax(X) # 每个元素值非负 且每行值和为1
print(X_prop)
print(X_prop.sum(1, keepdim=True))  # 验证每行和是否为1

# 定义模型
def net(X):
    # W.shape[0] 784 输入维度，所以-1自动计算出来的是batch-size的大小
    # softmax输出元素值大于0，行和为1的矩阵
    return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

# 细节：怎么在预测值矩阵里面，根据标号把对应的预测值取出来。
y = torch.tensor([0, 2])  # 两个真实的标号
y_hat = torch.tensor([[0.1, 0.3, 0.6],[0.3, 0.2, 0.5]]) # 预测值 两个样本
print(y_hat[[0,1], y]) # 对第0样本，拿出y[0]标号对应的输出，对第1样本，拿出y[1]标号对应的输出
print(y_hat[[0,1], [0,2]])  # 第一个数组是样本标号，第二个数组是真实值的标号，拿到每个样本真实标号的预测值
print(y_hat[range(len(y_hat)), y])
# 高级索引 y_hat[[0,1], [0,2]] 选择了 y_hat 中位置为 (0, 0) 和 (1, 2) 的两个元素。
#  第一个数组对应行索引，第二个数组对应列索引。
# 定义损失函数--交叉熵损失函数
def cross_entropy(y_hat, y):
    # range(len(y_hat)) 预测值的样本标号数组 y真实值的标号--类别
    # torch.log 计算输入张量的自然对数， 对张量的每个元素取对数
    # 负的真实类别标号的预测值取对数--回想定义 -log(h_hat_y)
    return -torch.log(y_hat[range(len(y_hat)), y])
print(cross_entropy(y_hat, y))

# 分类精度--判断预测的类别是否是正确的
def accuracy(y_hat, y):
    """计算预测正确的数量"""
    # 如果预测值是二维矩阵，而且每一行列数大于1
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        # 每一行元素预测值最大的下标存到y_hat中 argmax返回的是索引下标--获取概率最高类别的索引
        y_hat = y_hat.argmax(axis=1)
    # print(y_hat.dtype, y_hat.type(torch.float).dtype)
    # y_hat.type(y.dtype) 设置元素数值类型和y保持一致
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())  # 预测类型成功的个数
print(accuracy(y_hat, y)/len(y))  # 预测成功率

class Accumulator:
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        # print(self.data, args)
        self.data = [a + float(b) for a,b in zip(self.data, args)]
        # print(self.data, args)
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 评估模型net的准确率
def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 将模型设置为评估模式
        # net.eval() 作用1. 网络结构如果有Batch Normalization 和 Dropout 层的，做模型评测必须加上这个命令以防止报错
        # 作用2： 关闭梯度计算
    # Accumulator 累加器 不断迭代数据X y 不断累加评测结果
    # Accumulator 初始化传入数值2 表示有数组有两个变量，第一个位置存正确测试数，第二个位置存样本总数，根据批次遍历依次累加
    metric = Accumulator(2) # 正确预测数、预测总数
    with torch.no_grad():
        for X, y in data_iter: # 迭代一次 一次批量
            # metric.add(该批次的预测类别正确的样本数，批次的总样本数)
            # y.numel() pytorch中用于返回张量的元素总个数
            metric.add(accuracy(net(X), y), y.numel())

    # 返回值=分类正确样本数/样本总数=精度
    return metric[0] / metric[1]

print(evaluate_accuracy(net, test_iter)) # 0.1409  模型随机初始化，总共十个类别，初始概率应该是0.1左右
# raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
# RuntimeError: DataLoader worker (pid(s) 90412, 47140, 98280, 126560) exited unexpectedly
# 在加载数据的时候data.DataLoader(num_workers) win环境下num_workers要设置为0

# 训练 对训练数据迭代一次
def trian_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期（定义见第三章）"""
    # 如果模型是用nn模组定义的
    if isinstance(net, torch.nn.Module):
        net.train()  # 将模型设置为训练模式 告诉pytorch要计算梯度
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)  # 三个参数需要累加的迭代器
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y) # 计算损失函数
        # 如果优化器是pytorch内置的优化器
        # 下面两个加的结果有什么区别
        # print(float(l)*len(y), accuracy(y_hat,y), y.size().numel(),
        #       float(l.sum()), accuracy(y_hat, y), y.numel())
        if isinstance(updater, torch.optim.Optimizer):
            # 使用pytorch内置的优化器和损失函数
            updater.zero_grad() # 1.梯度先置零
            l.mean().backward() # 2.计算梯度
            updater.step()      # 3.调用step参数进行一次更新
            metric.add(float(l)*len(y), accuracy(y_hat,y), y.size().numel())
        else:
            # 使用定制的优化器和损失函数
            # 自己实现的l是一个向量
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    # 损失累加/总样本数  训练正确的/总样本数
    return metric[0] / metric[2], metric[1] / metric[2]

# 实现一个动画展示训练进程
class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
                 xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
                 nrows=1, nclos=1, figsize=(3.5, 2.5)):

        # 增量的绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, nclos, figsize=figsize)
        if nrows * nclos == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend
        )
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)  # n=3 y有三个值，对应三条曲线的值
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a,b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        display.display(self.fig)
        display.clear_output(wait=True)

# 开启训练
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3,0.9],
                        legend=['train-loss', 'train-acc', 'test-acc'])
    for epoch in range(num_epochs):
        train_metrics = trian_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch+1, train_metrics+(test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc

lr = 0.1
def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
d2l.plt.show()

def predict_ch3(net, test_iter, n=6): #@save
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [trus + '\n' + pred for trus, pred in zip(trues, preds)]
    d2l.show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    d2l.plt.show()


predict_ch3(net, test_iter)

breakpoint()