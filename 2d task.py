import numpy as np
import matplotlib.pyplot as plt


M, N = 512, 100
a = 4
calc_input_field = lambda temp: (np.exp(-2 * 1j * (temp[:, :, 0] ** 3 + temp[:, :, 1] ** 3)))
calc_gaussian = lambda temp: (np.exp(-(temp[:, :, 0] ** 2 + temp[:, :, 0] ** 2)))
b = N ** 2 / (4 * a * M)
step = 2 * a / (N - 1)


"""
    БПФ
     values - 2d массив дискретизированной f;
    1. дополнение до размера M нулями слева и справа
    2. перестановка левой и правой части массива
    3. 2d БПФ -> вектор F 
    4. умножить на step и возвести в квадрат(2d)
    5. перестановка левой и правой части F
    6. выбор центральной части F
    
"""


# добавление нулей
def add_zeros(values):
    new_values = np.zeros((M, M), dtype=complex)
    length = len(values)
    left = (M - length) // 2
    right = left + length
    new_values[left:right, left:right] = values
    return new_values


# перестановка половин
def swap_half(values):
    length = len(values)
    half_length = length // 2
    new_values = values
    # перестановка относительно столбцов
    new_values = np.c_[new_values[:, half_length:], new_values[:, :half_length]]
    # перестановка относительно строк
    new_values = np.r_[new_values[half_length:, :], new_values[:half_length, :]]
    return new_values


# выбираем центральную часть массива
def get_center(values):
    left = (len(values) - N) // 2
    right = left + N
    return values[left:right, left:right]


def calc_fft(values):
    values = add_zeros(values)
    values = swap_half(values)
    fft_res = np.fft.fft2(values)
    vect_F = fft_res * step ** 2
    vect_F = swap_half(vect_F)
    vect_F = get_center(vect_F)
    return vect_F


"""
    Построение амплитуд и фаз
"""


# построение графика
def create_chart(values, b, name):
    x = np.linspace(float(-b), float(b), N)
    y = np.linspace(float(-b), float(b), N)
    amplitude = np.abs(values)
    fig, ax = plt.subplots(ncols=1, nrows=2)
    ax[0].imshow(amplitude, extent=(-b, b, -b, b))
    phase = np.angle(values)
    ax[1].imshow(phase, extent=(-b, b, -b, b))
    plt.savefig(name)
    plt.show()


if __name__ == '__main__':
    # проверки
    assert M >= N
    assert M % 2 == 0
    assert N % 2 == 0

    # разбивка на точки
    x = np.linspace(-a, a, N)
    x_shifted = x - step / 2

    # декартово произведение (x,y) для БПФ, т.к. операции матричного характера
    cartesian_product = list(((x_shifted[i], x_shifted[j])
      for i in range(len(x_shifted))
      for j in range(len(x_shifted))))
    cartesian_product = np.reshape(cartesian_product, (N, N, 2))

    gauss_fft = calc_gaussian(cartesian_product)
    input_field_fft = calc_input_field(cartesian_product)
    fft_func = np.array(input_field_fft, dtype=np.complex)
    # вычисления и построение графиков
    y_fft = calc_fft(fft_func)
    create_chart(y_fft, b, 'fft-var,1d.png')

