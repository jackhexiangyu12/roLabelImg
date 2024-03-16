import numpy as np
import matplotlib.pyplot as plt
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
# 圆锥参数
center = (0, 0)
radius = 0.1

# 计算每个点到圆锥中心的距离
distance_matr = np.sqrt((x_matr - center[0])**2 + (y_matr - center[1])**2)

# 在圆锥范围内的点值为1，否则为0
cone = np.where(distance_matr <= radius, 1, 0)

# 绘制圆锥
plt.imshow(cone, origin='lower', extent=(geo["x_Wb"], geo["x_Eb"], geo["y_Sb"], geo["y_Nb"]))
plt.colorbar()
plt.xlabel('x')
plt.ylabel('y')
plt.title('Cone')
plt.show()
