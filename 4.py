import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


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


# 假设有一个3D图像
# 定义圆锥的高度和底面半径
h = 5
r = 3

import numpy as np

N = 50
image = np.zeros((N, N, N))

# 圆锥参数
h = 10
r = 10
center = (25, 0, 0)

# 设置圆锥部分为1
for z in range(N):
    for y in range(N):
        for x in range(N):
            # 计算当前点到圆锥中心的距离
            distance = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
            # 如果当前点在圆锥内，则设置为1
            if distance <= r and z <= center[2] + h:
                image[z, y, x] = 1

# 定义旋转轴和角度
axis = [0, 1, 0]  # 旋转轴
theta = - math.atan(r / h)  # 旋转角度

# 调用 rodriguesRotate 进行旋转
rotated_image = rodriguesRotate(image, M, M, M, axis, theta)

# # 可视化旋转前后的图像
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# 用plt显示三维图形

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(image, edgecolor='k')
plt.title('Original Image')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.voxels(rotated_image, edgecolor='k')
plt.title('Rotated Image')
plt.show()

# 将rotated_image投影到xoy平面，只取最接近xoy平面的侧面，按照距离附上颜色

# for x in range(N):
#     for y in range(N):
#         for z in range(N):

print()
