import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义圆锥的高度和底面半径
h = 5
r = 3

# 创建一个空的三维数组，表示图像
N = 50
image = np.zeros((N, N, N))

# 根据圆锥的高度和底面半径，在图像数组中设置圆锥的部分为1
for z in range(N):
    for y in range(N):
        for x in range(N):
            if (x - N // 2) ** 2 + (y - N // 2) ** 2 <= ((z * r / h) - N // 2) ** 2:
                image[z, y, x] = 1

# 创建一个绘图对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 使用ax.voxels绘制圆锥
ax.voxels(image, edgecolor='k')

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()
