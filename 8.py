import numpy as np
import matplotlib.pyplot as plt
def get_kugel(x_matr, y_matr, r):
    """
    get the Ball-on-disc matrix

    parameters :
    x_matr : x Matrix
    y_matr : y Matrix
    r      : Ball Radius

    Return ： Ein Matrix von der Hoehe
    """
    h_prof = r - np.sqrt(r ** 2 - x_matr ** 2 - y_matr ** 2)  # [m] gap height variation induced by profile  轮廓引起的间隙高度变化
    return h_prof


def matr_prof(matrix, x1, x2, y1, y2):
    plt.imshow(matrix, cmap='viridis', origin='lower', extent=[x1, x2, y1, y2])
    plt.colorbar()
    plt.title('Matrix Visualization')
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.show()


def main():
    N = 2 ** 7 + 1
    x_Wb = -0.25e-3
    x_Eb = 0.25e-3
    y_Sb = -0.25e-3
    y_Nb = 0.25e-3
    x = np.linspace(x_Wb, x_Eb, N)
    y = np.linspace(y_Sb, y_Nb, N)
    x_matr, y_matr = np.meshgrid(x, y)
    r = 0.1
    h_prof = get_kugel(x_matr, y_matr, r)
    matr_prof(h_prof, x_Wb, x_Eb, y_Sb, y_Nb)


if __name__ == "__main__":
    main()
