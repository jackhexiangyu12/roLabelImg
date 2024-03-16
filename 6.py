import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 创建一个20*20*20的圆锥体数组
N = 20
h = 5
r = 3

def create_cone(N, h, r):
    image = np.zeros((N, N, N))
    for z in range(N):
        for y in range(N):
            for x in range(N):
                if (x - N // 2) ** 2 + (y - N // 2) ** 2 <= ((z * r / h) - N // 2) ** 2:
                    image[z, y, x] = 1
    return image

cone = create_cone(N, h, r)

# 创建一个图像并绘制圆锥体
fig = plt.figure()
ax = fig.gca()
ax.voxels(cone, edgecolor='k')

# 设置图像的显示范围
ax.set_xlim(0, N)
ax.set_ylim(0, N)
ax.set_zlim(0, N)

plt.show()
