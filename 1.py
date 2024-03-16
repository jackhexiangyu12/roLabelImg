import numpy as np
import matplotlib.pyplot as plt
import numpy
from scipy.interpolate import RegularGridInterpolator

# import scipy.interpolate.RegularGridInterpolator as RegularGridInterpolator

# Geometry settings:
geo = {"N": 2 ** 7 + 1, "x_Wb": -0.25e-3, "x_Eb": 0.25e-3, "y_Sb": -0.25e-3, "y_Nb": 0.25e-3}
# [-] number of discretization points in the x1-direction x1方向上的离散点数
# [m] xl-coordinate of cell at the West boundary
# [m] x1-coordinate of cell at the East boundary
# [m] x2-coordinate of cell at the South boundary
# [m] x2-coordinate of cell at the North boundary

geo["dx1"] = (geo["x_Eb"] - geo["y_Nb"]) / (geo["N"] - 1)
# [m] Spacing in x-direction
geo["dx2"] = (geo["y_Nb"] - geo["y_Sb"]) / (geo["N"] - 1)
# [m] Spacing in y-direction
# [m] xi-coordinates with uniform discretization
geo["x"] = np.linspace(geo["x_Wb"], geo["x_Eb"], geo["N"])
geo["y"] = np.linspace(geo["y_Sb"], geo["y_Nb"], geo["N"])
# [m] x2-coordinates with uniform discretization
x_matr, y_matr = np.meshgrid(geo["x"], geo["y"])

print(x_matr)
print(y_matr)
print(geo)
print()

# -*- coding: utf-8 -*-
# 每天给自己一个希望,试着不为明天而烦恼,不为昨天而叹息,只为今天更美好!

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# 简单方法画出漂亮的圆锥体（底面在上，顶点在原点）
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 生成圆锥数据，底面半径为1，高度为1，其余的情形留待发挥
# 先根据极坐标方式生成数据
u = np.linspace(0, 2 * np.pi, 50)  # linspace的功能用来创建等差数列
v = np.linspace(0, np.pi, 50)

# 数据转化为平面坐标数据
x = np.outer(np.cos(u), np.sin(v))  # outer（a，b） 外积：a的每个元素乘以b的每个元素，二维数组
y = np.outer(np.sin(u), np.sin(v))
z = np.sqrt(x ** 2 + y ** 2)  # 圆锥体的高

N = 2 ** 7 + 1
image = np.zeros((N, N, N))

for i in range(50):
    image[x[i], y[i], z[i]] = 1

# 把整个圆锥三维图形都保存在"image.npy"
np.save("image.npy", image)

# Plot the surface
ax.plot_surface(x, y, z, color='b')
plt.show()


# 想要在矩阵（桌面）上反应一个圆锥。圆锥的侧面放在桌子上，头朝y轴。输入参数 是 x-matr y-matr，圆锥的高h和 底面半径r。最后返回一个高度矩阵，矩阵数值的大小是圆锥下表面到桌面的距离

# 广义的图像变换函数
# 输入参数为三维数组img, 变换的中心x_center, y_center, z_center及3x3的变换矩阵transform_matrix
# 须注意的是，数组的轴依序为(z,y,x)
# 这里省略了各种检查,比如矩阵的可逆性
def generalTransform(image, x_center, y_center, z_center, transform_matrix, method='linear'):
    # inverse matrix
    trans_mat_inv = numpy.linalg.inv(transform_matrix)
    # create coordinate meshgrid
    Nz, Ny, Nx = image.shape
    x = numpy.linspace(0, Nx - 1, Nx)
    y = numpy.linspace(0, Ny - 1, Ny)
    z = numpy.linspace(0, Nz - 1, Nz)
    zz, yy, xx = numpy.meshgrid(z, y, x, indexing='ij')
    # calculate transformed coordinate
    coor = numpy.array([xx - x_center, yy - y_center, zz - z_center])
    coor_prime = numpy.tensordot(trans_mat_inv, coor, axes=((1), (0)))
    xx_prime = coor_prime[0] + x_center
    yy_prime = coor_prime[1] + y_center
    zz_prime = coor_prime[2] + z_center
    # get valid coordinates (cell with coordinates within Nx-1, Ny-1, Nz-1)
    x_valid1 = xx_prime >= 0
    x_valid2 = xx_prime <= Nx - 1
    y_valid1 = yy_prime >= 0
    y_valid2 = yy_prime <= Ny - 1
    z_valid1 = zz_prime >= 0
    z_valid2 = zz_prime <= Nz - 1
    valid_voxel = x_valid1 * x_valid2 * y_valid1 * y_valid2 * z_valid1 * z_valid2
    z_valid_idx, y_valid_idx, x_valid_idx = numpy.where(valid_voxel > 0)
    # interpolate using scipy RegularGridInterpolator
    image_transformed = numpy.zeros((Nz, Ny, Nx))
    data_w_coor = RegularGridInterpolator((z, y, x), image, method=method)
    interp_points = numpy.array([zz_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                                 yy_prime[z_valid_idx, y_valid_idx, x_valid_idx],
                                 xx_prime[z_valid_idx, y_valid_idx, x_valid_idx]]).T
    interp_result = data_w_coor(interp_points)
    image_transformed[z_valid_idx, y_valid_idx, x_valid_idx] = interp_result
    return image_transformed


# 旋转函数,通过Rodrigues rotation formula计算出旋转矩阵,然后调用图四中的变换函数generalTransform以计算旋转结果
def rodriguesRotate(image, x_center, y_center, z_center, axis, theta):
    v_length = numpy.linalg.norm(axis)
    if v_length == 0:
        raise ValueError("length of rotation axis cannot be zero.")
    if theta == 0.0:
        return image
    v = numpy.array(axis) / v_length
    # rodrigues rotation matrix
    W = numpy.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rot3d_mat = numpy.identity(3) + W * numpy.sin(theta) + numpy.dot(W, W) * (1.0 - numpy.cos(theta))
    # transform with given matrix
    return generalTransform(image, x_center, y_center, z_center, rot3d_mat, method='linear')


def main():
    # 读取图像
    image = numpy.load("image.npy")
    # 旋转
    image_rotated = rodriguesRotate(image, 0, 0, 0, [1, 0, 0], numpy.pi / 2)
    # 保存结果
    numpy.save("image_rotated.npy", image_rotated)
    # 显示结果
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(image_rotated, edgecolor='k')
    plt.show()


if __name__ == '__main__':
    main()
