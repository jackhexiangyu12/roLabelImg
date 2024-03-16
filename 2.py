import numpy as np

def calculate_cone_height_matrix(x_matr, y_matr, h, r):
    # 计算圆锥顶部到每个点的距离
    distances = np.sqrt((x_matr ** 2 + y_matr ** 2) * (h ** 2) / (r ** 2))

    # 用圆锥高度减去距离得到高度矩阵
    height_matrix = h - distances
    return height_matrix

# Example usage
N = 7
x_Wb = -0.25e-3
x_Eb = 0.25e-3
y_Sb = -0.25e-3
y_Nb = 0.25e-3
x = np.linspace(x_Wb, x_Eb, N)
y = np.linspace(y_Sb, y_Nb, N)
x_matr, y_matr = np.meshgrid(x, y)
h = 0.1  # 圆锥高度
r = 0.05  # 圆锥底面半径
height_matrix = calculate_cone_height_matrix(x_matr, y_matr, h, r)
print(height_matrix)
