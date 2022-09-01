import numpy as np
import matplotlib.pyplot as plt


def func_forward(x, w, b=None):
    assert x.shape[1] == w.shape[0]
    if b is None:
        return np.matmul(x, w)
    assert w.shape[1] == b.shape[0]
    return np.matmul(x, w) + b


def func_backward(x, w, z_grad):
    assert z_grad.shape[0] == x.shape[0]
    assert z_grad.shape[1] == w.shape[1]

    x_grad = z_grad @ w.T
    w_grad = x.T @ z_grad / x.shape[0]
    b_grad = np.mean(z_grad, axis=0)

    return x_grad, w_grad, b_grad


def relu_forward(z):
    return z * (z > 0)


def relu_backward(z, a_grad):
    return a_grad * (z > 0)


def func_compute_loss_grad(y_true, y_pred):
    assert y_true.shape == y_pred.shape
    return 2 * (y_pred - y_true)


def func_update(params, grads, learning_rate=1e-2):
    assert isinstance(params, list)
    for i in range(len(params)):
        params[i] -= learning_rate * grads[i]
    # return params


# def sigmoid_derivative(s):
#     ds = s * (1 - s)
#     return ds
#
# def ReLU_derivative(s):
#     ds = s * (1 - s)
#     return ds
#

# def sigmoid(x):
#     s = 1 / (1 + np.exp(-x))
#     return s
#
#
# def ReLU(x):
#     if (x > 0):
#         return x
#     else:
#         return 0
#


# Initialize

x = np.linspace(-1, 1, 1000).reshape((1000, 1))
y_true = x ** 2 + 2 + np.random.normal(0, 0.1, (1000, 1))

w1 = np.random.normal(0, 0.1, (1, 100))
w2 = np.random.normal(0, 0.1, (100, 1))
b1 = np.zeros((100,))
b2 = np.zeros((1,))
learning_rate = 1e-1
for t in range(1000):
    # Forward
    z = func_forward(x, w1, b1)
    a = relu_forward(z)
    y_pred = func_forward(a, w2, b2)

    # 损失计算
    loss = func_compute_loss_grad(y_true, y_pred)

    # Backward
    pred_grad = func_compute_loss_grad(y_true, y_pred)
    a_grad, w2_grad, b2_grad = func_backward(a, w2, pred_grad)
    z_grad = relu_backward(z, a_grad)
    _, w1_grad, b1_grad = func_backward(x, w1, z_grad)

    # 更新权值
    params = [w1, w2, b1, b2]
    grads = [w1_grad, w2_grad, b1_grad, b2_grad]
    func_update(params, grads, learning_rate)

    if (t % 100 == 0):
        plt.scatter(x, y_true, 20)
        plt.plot(x, y_pred, 'r-', lw=1, label="plot figure")
        # plt.text(0, 2, "Loss %.4f" % loss, fontdict={'size': 20, 'color': 'red'})
        plt.show()
