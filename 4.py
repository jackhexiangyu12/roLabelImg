import math
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import numpy
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import FuncFormatter
import time

fenbianlv = 1.5
skip_arg = False


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


def divide_by_thousand(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x * 1e-3)


def func(h, r):
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
    # x_matr, y_matr = np.meshgrid(geo["x"], geo["y"])
    # rotated_center = [N // 2, N // 2, N // 2]
    M = N // 2
    # 创建一个空的三维数组，表示图像
    image = np.zeros((N, N, N))
    # length = N//2*h//r
    # 根据圆锥的高度和底面半径，在图像数组中设置圆锥的部分为1
    for z in range(N):
        for y in range(N):
            for x in range(N):
                if (x - N // 2) ** 2 + (y - N // 2) ** 2 <= ((z * r / h)+N//2) ** 2:
                    image[z, y, x] = 1

    # 定义旋转轴和角度
    axis = [0, 1, 0]  # 旋转轴
    theta = math.atan(r / h)  # 旋转角度

    # 调用 rodriguesRotate 进行旋转
    rotated_image = rodriguesRotate(image, -N//2, N // 2, 0, axis, theta)
    # np.save("rotated_image.npy", rotated_image)
    if not skip_arg:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(image, edgecolor='k')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('Original Image')
        plt.savefig(time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime()) + "Original Image.png")
        plt.show()
        # # 可视化旋转前后的图像
        # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        # 用plt显示三维图形
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.voxels(rotated_image, edgecolor='k')
        plt.title('Rotated Image')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.savefig(time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime()) + "Rotated Image.png")
        plt.show()
    # rotated_image = np.load("rotated_image.npy", encoding='bytes', allow_pickle=True)
    # 将rotated_image投影到xoy平面，只取最接近xoy平面的侧面，按照距离附上颜色

    # image2d先填充为70*70的70矩阵
    image2d = np.full((N, N), 0)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                if rotated_image[x, y, z] == 1:
                    image2d[x, y] = z - M
                    break
    # 可视化投影结果
    # formatter = FuncFormatter(divide_by_thousand)
    fig, ax = plt.subplots()
    # ax = plt.gca()
    # label = [i for i in range(5)]  # 填写自己的标签
    # ax.set_xticklabels(label)
    # ax.xaxis.set_major_formatter(formatter)
    # ax.yaxis.set_major_formatter(formatter)
    plt.xticks([])  # 去掉x轴
    plt.yticks([])  # 去掉y轴
    plt.axis('off')  # 去掉坐标轴
    ax.imshow(image2d, cmap='viridis', origin='lower', interpolation='bilinear')
    ax.axis('off')
    plt.savefig(time.strftime("%Y_%m_%d_%H_%M_%S_", time.localtime()) + "Projection Image.png")
    plt.show()

    # image2d_txt = []
    # for z in range(N):
    #     for y in range(N):
    #         for x in range(N):
    #             if rotated_image[z, y, x] == 1:
    #                 image2d_txt.append([geo["x"][x], geo["x"][y], float(x - M)])
    #                 break
    # # 保存image2d_txt
    # np.savetxt("image2d.txt", image2d_txt)
    # image2d_txt = np.load("image2d.txt",encoding='bytes', allow_pickle=True)
    # 按照image2d_txt输出可视化结果

    # 创建一个三维图形
    # fig = plt.figure()
    #
    # # 提取坐标和深度值
    # xs = [point[0] for point in image2d_txt]
    # ys = [point[1] for point in image2d_txt]
    # zs = [point[2] for point in image2d_txt]

    # plt.scatter(xs, ys, c=zs, s=10, cmap='viridis')
    # plt.colorbar(label='Depth')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.show()

    print()
    return image2d


if __name__ == '__main__':
    # N = 2 ** 7 + 1
    N = 50
    func(N, N*1/30)
    print()
