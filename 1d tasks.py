import numpy as np
import matplotlib.pyplot as plt

# входные парметры
M, N = 512, 100
a = 4
b = N ** 2 / (4 * a * M)
step = 2 * a / (N - 1)
# входное поле в точке
calc_input_field = lambda x: np.exp(2 * 1j * (x ** 3))
# гауссовский пучок в точке
calc_gaussian = lambda x: np.exp(-x**2)
# ядро преобразования
calc_kernel = lambda x, u: np.exp((-2 * np.pi * 1j) * x * u)


"""
 values - 2d массив дискретизированной f;
    1. дополнение до размера M нулями слева и справа
    2. перестановка левой и правой части массива
    3. 2d БПФ -> вектор F 
    4. умножить на step и возвести в квадрат(2d)
    5. перестановка левой и правой части F
    6. выбор центральной части F
"""


# перестановка половин
def swap_half(values):
    length = len(values)
    half_length = length // 2
    new_values = values
    new_values = np.r_[new_values[half_length:], new_values[:half_length]]
    return new_values


# добавление нулей
def add_zeros(values):
    new_values = np.zeros(M, dtype=complex)
    length = len(values)
    left = (M - length) // 2
    right = left + length
    new_values[left:right] = values
    return new_values


# выбираем центральную часть массива
def get_center(values):
    left = (len(values) - N) // 2
    right = left + N
    return values[left:right]


# БПФ
def calc_finite_fft(values):
    values = add_zeros(values)
    values = swap_half(values)
    fft_res = np.fft.fft(values)
    vect_F = fft_res * step
    vect_F = swap_half(vect_F)
    vect_F = get_center(vect_F)
    return vect_F


"""
    Интегрирование через матричные операции
"""


# численный расчёт интеграла
def calc_finite_integral(x, values, u):
    kernel = calc_kernel(x, u)
    result = kernel * values
    int_weights, int_weights[0], int_weights[-1] = np.ones(N) * step, 0.5 * step, 0.5 * step
    result = result * np.broadcast_to(int_weights[:, np.newaxis], (N, N))
    result = np.sum(result, axis=0)
    return result


"""
    Расчёт амплитуды (np.abs) и фазы (np.angle)
    Построение графиков
"""


def create_chart(values, border, name):
    x_points = np.linspace(float(-border), float(border), len(values))
    plt.subplot(2, 1, 1)
    plt.plot(x_points, np.abs(values), color='red')
    plt.title("Амплитуда", fontsize=10)
    plt.xlabel("x", fontsize=10)
    plt.ylabel("A", fontsize=10)
    plt.grid(True)
    plt.subplot(2, 1, 2)
    pha = np.angle(values)
    plt.plot(x_points, pha, color='blue')
    plt.title("Фазa", fontsize=10)
    plt.xlabel("x", fontsize=10)
    plt.ylabel("phase", fontsize=10)
    plt.grid(True)
    plt.show()
    plt.savefig(name)


if __name__ == '__main__':
    # проверки
    assert M >= N
    assert M % 2 == 0
    assert N % 2 == 0
    # разбивка на точки
    old_x = np.linspace(-a, a, N)
    x_shifted = old_x - step / 2
    new_x = np.linspace(-b, b, N)
    # подсчёт гаусса и функции из варианта для БПФ
    gauss_fft = calc_gaussian(x_shifted)
    input_field_fft = calc_input_field(x_shifted)
    # подсчёт гаусса и функции из варианта для интегрирования
    gauss_integ = calc_gaussian(old_x)
    input_field_integ = calc_input_field(old_x)
    # функции для интегрирования и БПФ - массивы комлексных чисел
    integral_func = np.array(gauss_integ, dtype=np.complex)
    fft_func = np.array(gauss_fft, dtype=np.complex)
    # для интегрирования на матричных операциях преобразование x, u, integral_func
    x_for_integ = np.broadcast_to(old_x[:, np.newaxis], (N, N))
    u_for_integ = np.broadcast_to(new_x[np.newaxis, :], (N, N))
    integral_func = np.broadcast_to(integral_func[:, np.newaxis], (N, N))
    # вычисления
    y_fft = calc_finite_fft(fft_func, step)
    y_integral = calc_finite_integral(step, x_for_integ, integral_func, u_for_integ)
    # построение графиков
    create_chart(y_fft, b, 'fft-gauss,1d.png')
    create_chart(y_integral, b, 'integral-gauss,1d.png')