import matplotlib

print("The matplotlib version is: ", matplotlib.__version__)
import matplotlib.pyplot as plt
import numpy as np

fig, ax = plt.subplots(2, 1)  # 生成 figure 及包含于其中的 2 行 1 列共 2 个 axes 子图
ax[0].plot([1, 2, 3, 4], [1, 4, 2, 3])  # 在子图 1 上绘制数据，两个列表分别表示数据的横纵坐标，即四个点的坐标分别为（1,1），（2,4），（3,2），（4,3）
ax[1].plot([0, 1, 4, 9])  # 在子图 2 上绘制数据，参数列表表示数据的纵坐标，横坐标默认从 0 开始的整数值，即[0,1,2,3]，因此四个点的坐标分别为（0,0），（1,1），（2,4），（3,9）
fig.show()  # 显示窗口

print(">>>-------------------------------------------------------->>>")

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)  # 从-π到π均匀取 256 个点
C, S = np.cos(X), np.sin(X)  # 分别得到这些点的余弦值和正弦值生成的一维数组
plt.plot(X, C)  # 横坐标为 X，纵坐标为 C，绘制连线
plt.plot(X, S)  # 横坐标为 X，纵坐标为 S，绘制连线
plt.show()

print(">>>-------------------------------------------------------->>>")

fig, ax = plt.subplots(figsize=(5, 2.7))
y1, y2 = np.random.randint(20, size=(2, 10))
x = np.arange(len(y1))
ax.plot(x, y1, color='blue', linewidth=3, linestyle='--')
line, = ax.plot(x, y2, color='orange', linewidth=2, marker='o')
line.set_linestyle(':')  # 设置连线形状为点线
fig.show()

print(">>>-------------------------------------------------------->>>")

fig, ax = plt.subplots()
x = np.linspace(-np.pi, np.pi, 256)
ax.plot(x, np.sin(x), '-g', label='sin(x)')  # '-g'表示线条样式为'-'，颜色为'g'
ax.plot(x, np.cos(x), ':b', label='cos(x)')  # ':b'表示线条样式为':'，颜色为'b'
ax.legend(fancybox=True)

print(">>>-------------------------------------------------------->>>")

fig, ax = plt.subplots()
rng = np.random.RandomState(0)  # 生成伪随机数生成器，随机数种子为 0
for marker in ['o', '.', ',', 'x', '+', 'v', '^', '<', '>', 's', 'd']:
    ax.plot(rng.rand(5), rng.rand(5), marker, label='marker={}'.format(marker))
ax.legend()
ax.set_xlim(0, 1.6)
fig.show()

print(">>>-------------------------------------------------------->>>")

fig, ax = plt.subplots()
rng = np.random.RandomState(0)
x = rng.randn(100)
y = rng.randn(100)
colors = rng.rand(100)
sizes = 1000 * rng.rand(100)
axs = ax.scatter(x, y, c=colors, s=sizes, alpha=0.3)
fig.colorbar(axs, ax=ax)  # 设置右侧的颜色条
fig.show()

print(">>>-------------------------------------------------------->>>")

x = np.linspace(0, 10, 11)  # 11 个超参数值
y = np.random.rand(10, 11)  # 模拟 10 次实验不同超参数值下的准确率
mean = y.mean(axis=0)  # 按列求平均
std = y.std(axis=0)  # 按列求方差
fig, ax = plt.subplots()
ax.errorbar(x, mean, yerr=std, fmt='.k', ecolor='r', elinewidth=2, capsize=4)
fig.show()

print(">>>-------------------------------------------------------->>>")

fig, ax = plt.subplots(2, 2)  # 2 行 2 列四个子图
x = np.linspace(0, 5, 50)
y = np.linspace(0, 3, 30)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) ** 10 + np.cos(10 + Y * X) * np.cos(X)
h = ax[0][0].hist(Z)  # 左上子图绘制直方图
axs01 = ax[0, 1].pcolormesh(X, Y, Z, vmin=-1, vmax=1, cmap='RdBu_r')  # 右上子图
fig.colorbar(axs01, ax=ax[0, 1])  # 子图旁绘制色条
axs10 = ax[1][0].contourf(X, Y, Z, 20)  # 左下子图绘制带有填充色的等高线图
fig.colorbar(axs10, ax=ax[1][0])  # 子图旁绘制色条
ax[1][1].contour(X, Y, Z, colors='k')  # 右下子图绘制等高线
fig.show()

print(">>>-------------------------------------------------------->>>")

zline = np.linspace(0, 15, 1000)
xline = np.sin(zline)
yline = np.cos(zline)
zdata = 15 * np.random.random(100)
xdata = np.sin(zdata) + 0.1 * np.random.randn(100)
ydata = np.cos(zdata) + 0.1 * np.random.randn(100)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D(xline, yline, zline, 'gray')
ax = plt.axes(projection='3d', xlabel='x', ylabel='y', zlabel='z')
ax.plot3D(xline, yline, zline, 'gray')
ax.scatter3D(xdata, ydata, zdata, c=zdata, cmap='Greens')
plt.show()

print(">>>-------------------------------------------------------->>>")

x = np.linspace(-6, 6, 30)
y = np.linspace(-6, 6, 30)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X ** 2 + Y ** 2))
fig = plt.figure()
ax = fig.add_subplot(221)  # 221 表示当作 2 行 2 列，生成第 1 个二维子图，即左上子图
ax.contourf(X, Y, Z, 50)  # 绘制二维等高线
ax = fig.add_subplot(222, projection='3d')  # 生成第 2 个三维子图，即右上子图
ax.contour3D(X, Y, Z, 50)  # 绘制三维等高线
ax = fig.add_subplot(223, projection='3d')  # 生成第 3 个三维子图，即左下子图
ax.plot_wireframe(X, Y, Z, color='black')  # 绘制网格图
ax = fig.add_subplot(224, projection='3d')  # 生成第 4 个三维子图，即右下子图
ax.plot_surface(X, Y, Z)  # 绘制曲面图
plt.show()

print(">>>-------------------------------------------------------->>>")

c = ax.contour3D(X, Y, Z, 50)  # X, Y, Z 为前例中的数据
dist, elev, azim = ax.dist, ax.elev, ax.azim  # 得到观测距离、仰角和方位角
ax = fig.add_subplot(232, projection='3d', xlabel='x', ylabel='y', zlabel='z', title='仰角变小')
c = ax.contour3D(X, Y, Z, 50)
ax.view_init(elev=elev - 10)  # 图形沿 y 方向顺时针旋转 10 度，仰角变小
ax = fig.add_subplot(233, projection='3d', xlabel='x', ylabel='y', zlabel='z', title='仰角变大')
c = ax.contour3D(X, Y, Z, 50)
ax.view_init(elev=elev + 10)  # 图形沿 y 方向逆时针旋转 10 度，仰角变大
ax = fig.add_subplot(234, projection='3d', xlabel='x', ylabel='y', zlabel='z', title='方位角变小')
c = ax.contour3D(X, Y, Z, 50)
ax.view_init(azim=azim - 10)  # 图形沿 z 方向逆时针旋转 10 度，方位角变小
ax = fig.add_subplot(235, projection='3d', xlabel='x', ylabel='y', zlabel='z', title='方位角变大')
c = ax.contour3D(X, Y, Z, 50)
ax.view_init(azim=azim + 10)  # 图形沿 z 方向顺时针旋转 10 度，方位角变大
ax = fig.add_subplot(236, projection='3d', xlabel='x', ylabel='y', zlabel='z', title='距离变大')
c = ax.contour3D(X, Y, Z, 50)
ax.dist = dist + 5  # 观测距离变大
plt.show()
