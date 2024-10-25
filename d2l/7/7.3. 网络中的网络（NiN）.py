import torch
import torch_npu
from torch import nn
from d2l import torch as d2l

# 确保使用 NPU 设备 0
device = torch.device("npu:0")

def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, strides, padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU()
    )

# 将模型移动到 NPU
net = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(3, stride=2),
    nn.Dropout(0.5),
    # 标签类别数是10
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    # 将四维的输出转成二维的输出，其形状为（批量大小，10）
    nn.Flatten()
).to(device)  # 移动模型到 NPUprint(d2l.__file__)

# 创建输入数据并移动到 NPU
X = torch.rand(size=(1, 1, 224, 224)).to(device)  # 将输入数据移动到 NPU
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__, 'output_shape"\t', X.shape)

# 设置超参数
lr, num_epochs, batch_size = 0.1, 10, 128
# 加载数据集并将数据移动到 NPU
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, device)