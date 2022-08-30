# 基于Pytorch构建简单的三层分类神经网络

# Pytorch下采用Adam梯度下降法进行神经网络的权值更新

'''
使用库构建简单的二分类数据集，编写简单的三层分类神经网络，
输入和输出均有两个神经元，中间隐藏层可设置 N 个神经元，
采用 Relu 函数作为激活函数。
数据集按两个类构建，每个类的样本为两维，分散在类中心附近。
学习梯度计算，及梯度下降法进行神经网络的权值修改。
'''

import torch
import torch.nn.functional
import matplotlib.pyplot as plt

# torch.manual_seed(1)  # reproducible

# make fake data

n_data = torch.ones(100, 2)
x0 = torch.normal(2 * n_data, 1)  # class0 x data (tensor), shape=(100, 2)
y0 = torch.zeros(100)  # class0 y data (tensor), shape=(100, 1)
x1 = torch.normal(-2 * n_data, 1)  # class1 x data (tensor), shape=(100, 2)
y1 = torch.ones(100)  # class1 y data (tensor), shape=(100, 1)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)  # shape (200, 2) FloatTensor = 32-bit floating
y = torch.cat((y0, y1), ).type(torch.LongTensor)  # shape (200,) LongTensor = 64-bit integer


class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.out = torch.nn.Linear(n_hidden, n_output)  # output layer

    def forward(self, x):
        x = torch.sigmoid(self.hidden(x))  # activation function for hidden layer

        x = self.out(x)
        return x


net = Net(n_feature=2, n_hidden=10, n_output=2)  # define the network
print(net)  # net architecture
optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()  # the target label is NOT an one-hotted
plt.ion()  # something about

for t in range(100):
    out = net(x)  # input x and predict based on x
    loss = loss_func(out, y)  # must be (1. nn output, 2. target), the target label is NOTone-hotted
    optimizer.zero_grad()  # clear gradients for next train
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients
    if t % 2 == 0:  # plot and show learning process
        plt.cla()
        prediction = torch.max(out, 1)[1]
        pred_y = prediction.data.numpy()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap='RdYlGn')
        accuracy = float((pred_y == target_y).astype(int).sum()) / float(target_y.size)
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
plt.ioff()
plt.show()
