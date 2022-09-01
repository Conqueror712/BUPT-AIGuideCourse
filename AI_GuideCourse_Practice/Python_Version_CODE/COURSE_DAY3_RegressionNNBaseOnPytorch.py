# 基于Pytorch构建简单的三层回归神经网络

# Pytorch下采用Adam梯度下降法进行神经网络的权值更新

'''
在调整模型更新权重和偏差参数的方式时，
不同的优化算法能使模型产生不同的效果
用梯度下降，随机梯度下降，还是Adam方法？
详见：https://zhuanlan.zhihu.com/p/27449596
'''

import torch
import numpy as np
import matplotlib.pyplot as plt

N, D_in, H, D_out = 20, 1, 128, 1

# 随机初始化数据
np.random.seed(0)
x = torch.tensor(np.arange(0, N, 1).reshape(N, D_in), dtype=torch.float32)
y = x ** 2 + torch.tensor(np.random.randn(N, D_out), dtype=torch.float32)

# 模型和损失函数的定义
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')


# 优化器的定义以及参数的更新，在这里使用Adam
learning_rate = 1e-4  # 设置学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(5000):
    # 通过传递x到模型中计算y的预测值
    y_pred = model(x)
    loss = loss_fn(y_pred, y) # 计算和打印损失
    if t % 1000 == 0:
        plt.cla()   # y_pred = model.predict(x)
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.scatter(x.data.numpy(), y_pred.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=1, label="plot figure")
        plt.text(0.5, 0, 't=%d:Loss=%.4f' % (t, loss), fontdict={'size': 20, 'color': 'red'})
        plt.show()

        '''
        在向后传递之前，使用优化器对象将它将更新的变量(模型的可学习权重)的所有梯度归零。
        这是因为默认情况下，每当调用.backward()时，梯度会在缓冲区中累积(即不覆盖)。
        '''
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()