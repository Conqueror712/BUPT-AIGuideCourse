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

N, D_in, H, D_out = 20, 1, 64, 1

# Create random input and output data
np.random.seed(0)
x = torch.tensor(np.arange(0, N, 1).reshape(N, D_in), dtype=torch.float32)  # 20*1
y = x + torch.tensor(np.random.randn(N, D_out), dtype=torch.float32)  # 20*1

# Use the nn package to define our model and loss function.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)
loss_fn = torch.nn.MSELoss(reduction='sum')

# Use the optim package to define an Optimizer that will update the weights of
# the model for us. Here we will use Adam; the optim package contains many other
# optimization algorithms. The first argument to the Adam constructor tells the
# optimizer which Tensors it should update.
learning_rate = 1e-4  # 设置学习率
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for t in range(5000): # 通过传递x到模型中计算y的预测值
    y_pred = model(x)
    loss = loss_fn(y_pred, y) # 计算和打印损失
    if t % 200 == 0:
        plt.cla()   # y_pred = model.predict(x)
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.scatter(x.data.numpy(), y_pred.data.numpy())
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'r-', lw=1, label="plot figure")
        plt.text(0.5, 0, 't=%d:Loss=%.4f' % (t, loss), fontdict={'size': 20, 'color': 'red'})
        plt.show()

        '''
        Before the backward pass, use the optimizer object to zero all of the
        gradients for the variables it will update (which are the learnable
        weights of the model). This is because by default, gradients are
        accumulated in buffers( i.e, not overwritten) whenever .backward()
        is called. Checkout docs of torch.autograd.backward for more details.
        '''
    optimizer.zero_grad()

    # 反向传播: 计算损失关于模型的梯度

    # 参数
    loss.backward()

    # 在优化器上调用step函数会更新它

    # 参数

    optimizer.step()

'''
运行代码，观察效果
修改不同参数，如学习率，循环次数，回归函数修改（如 y=x2），观察效果。
思考：代码中的激活函数采用的是 ReLU 函数，试修改为 Sigmoid 函数，观察效果。    
'''
