import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
from mpl_toolkits.mplot3d import Axes3D
from scipy import integrate


M, N = 64, 10
a = 4
# [-a, a]
h_x = (a + a) / N


# входное поле
def calc_input_field(x):
    return np.exp(2 * 1j * (x ** 3))


# гауссовский пучок
def calc_gaussian(x):
    return np.exp(- x * x)


def change_half(temp_field):
    left_temp, right_temp, new_field = [], [], [],
    lenght_of_field = len(temp_field)
    for i in range(int(lenght_of_field / 2)):
        left_temp.append(temp_field[i])
    for i in np.arange(int(lenght_of_field / 2), lenght_of_field):
        right_temp.append(temp_field[i])
    new_field.extend(right_temp)
    new_field.extend(left_temp)
    return new_field


# БПФ
def create_fft():
    # дополнение нулями и перестановка
    input_field, temp_field, result = [], [], []
    for x_i in range(N):
        input_field.append(calc_gaussian(x_i))
    zeros_mass = np.zeros(int(M/2))
    temp_field.extend(zeros_mass)
    temp_field.extend(input_field)
    temp_field.extend(zeros_mass)
    input_field = change_half(temp_field)
    # БПФ, вырезание центральной части и вычисление границ
    res_F = np.fft.fftshift((np.fft.fft(input_field)) * h_x)
    new_b = N ** 2 / (4 * a * M)
    return res_F, new_b


# подсчёт амплитуды для преобразования
def amplitude_conversion(mass):
    abs_mass = []
    for i in range(len(mass)):
        abs_mass.append(np.sqrt(mass[i].imag ** 2 + mass[i].real ** 2))
    return abs_mass


# графики
def create_charts_2d(values, name):
    x_points = np.linspace(float(-values[1]), float(values[1]), len(values[0]))
    plt.subplot(2, 1, 1)
    plt.plot(x_points, amplitude_conversion(values[0]), color='red')
    plt.title("Амплитуда преобразования", fontsize=10)
    plt.xlabel("x", fontsize=10)
    plt.ylabel("A", fontsize=10)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    pha = []
    for i in range(len(values[0])):
        pha.append(np.angle(values[0][i]))
    plt.plot(x_points, pha, color='blue')
    plt.title("Фаза преобразования", fontsize=10)
    plt.xlabel("x", fontsize=10)
    plt.ylabel("phase", fontsize=10)
    plt.grid(True)
    plt.savefig(name)
    plt.show()


def create_chart_3d(values, borders, name):
    x = np.linspace(float(-borders), float(borders), len(values))
    y, z = [], []
    for i in range(len(values)):
        y.append(values[i].real)
        z.append(values[i].imag)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, label='Входное поле')
    plt.show()
    plt.savefig(name)


if __name__ == '__main__':
    values = create_fft()
    create_charts_2d(values, '3.png')
    input_field = []
    for x_i in range(N):
        input_field.append(calc_gaussian(x_i))
    create_chart_3d(input_field, values[1], 'В2.png')
    create_chart_3d(values[0], values[1], '1.png')