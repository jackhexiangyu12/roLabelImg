import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 圆锥体参数
h = 1  # 圆锥体高度
r = 1  # 圆锥底面半径

# 生成圆锥数据
theta = np.linspace(0, 2*np.pi, 100)
x = r * np.cos(theta)
y = r * np.sin(theta)
z_base = np.zeros_like(theta)  # 圆锥底面的z坐标

# 绘制圆锥底面
ax.fill(x, y, z_base, color='b', alpha=0.5)

# 绘制圆锥侧面
z_top = np.full_like(theta, h)  # 圆锥顶点的z坐标
ax.plot(x, y, z_top, color='b')  # 连接底面和顶点的侧面

# 连接底面和顶点的直线
for i in range(len(theta)):
    ax.plot([0, x[i]], [0, y[i]], [0, h], color='b')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Cone')

plt.show()
