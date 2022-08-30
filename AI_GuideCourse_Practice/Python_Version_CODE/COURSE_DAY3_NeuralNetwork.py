# 基于numpy构建简单的三层回归神经网络
# 反向传播网络
# 梯度下降法进行神经网络的权值更新

# 输入和输出只有一个神经元 中间隐藏层可设置N个 sigmoid激活函数 随机噪声构建数据集
# 下面是代码：
import numpy as np
import matplotlib.pyplot as plt


# 激活函数的梯度


def sigmoid_derivative(s):
    ds = s * (1 - s)
    return ds


# 激活函数的定义


def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


'''
N是batch的大小
D_in是输入的维度
H是隐藏的维度
D_out是输出的维度
'''

N, D_in, H, D_out = 20, 1, 128, 1

np.random.seed(0)
# arange用于生成数组，在给定间隔内返回均匀间隔的值
# numpy.arange(n).reshape(a, b)用于依次生成n个自然数，并且以a行b列的数组形式显示
x = np.arange(0, N, 1).reshape(N, D_in) * 10  # 20 * 1
y = x + np.random.randn(N, D_out)  # 20 * 1

# Randomly initialize weights

w1 = np.random.randn(D_in, H)  # 1 * 64
w2 = np.random.randn(H, D_out)  # 64 * 1
learning_rate = 1e-3
# Forward pass: 计算预测的y值
for t in range(20000):
    h = x.dot(w1)
    h_relu = sigmoid(h)
    y_pred = h_relu.dot(w2)

    # 损失计算

    loss = np.square(y_pred - y).sum()

    # 反向传播计算w1 w2的关于损失的梯度

    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h = grad_y_pred.dot(w2.T)  # [N, H] = [N, 1] * [1, H]
    grad_h = grad_h * sigmoid_derivative(h_relu)  # [N, H] = [N, H] . [N, H]]
    grad_w1 = x.T.dot(grad_h)  # [1, H] = [1, N] * {N, H}

    # 更新权值

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

    if (t % 4000 == 0):
        plt.cla()
        plt.scatter(x, y)
        plt.scatter(x, y_pred)
        plt.plot(x, y_pred, 'r-', lw=1, label="plot figure")
        plt.text(0.5, 0, 't=%d:Loss=%.4f' % (t, loss), fontdict={'size': 20, 'color': 'red'})
        plt.show()

'''
运行代码，观察效果
修改不同参数，如学习率，循环次数，回归函数修改（如 y=x2），观察效果。
思考：代码中的激活函数采用的是 sigmoid 函数，试修改为 ReLU 函数，观察效果。    
'''
