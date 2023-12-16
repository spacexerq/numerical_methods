import numpy as np
from matplotlib import pyplot as plt
import random as rd

a = 0
b = 20
n = 200
h = (b - a) / n
lamb = -1


def core(x, t):
    return np.exp(lamb * (x - t))


def u_func(x):
    return 5


def func_exact(x):
    return 5 + (5 / 2 - 5 * np.exp(-2 * x) / 2)


def quadratic(n, a, b):
    t = np.linspace(a, b, n)
    x_output = [0] * n
    x_output[0] = u_func(t[0])
    for i in range(1, n):
        sum = 0
        for m in range(1, i):
            sum += core(t[i], t[m]) * x_output[m]
        x_output[i] = (u_func(i) + h / 2 * core(t[i], t[0]) * x_output[0] + h * sum) / (1 - h / 2 * core(t[i], t[i]))
    return x_output, t


def solve_quad(k, f, h, x):
    N = len(f)
    x[0] = f[0]  # начальное значение
    for i in range(1, N):
        s = np.sum(K[i, 1:i] * x[1:i])
        x[i] = (f[i] + h / 2 * k[i, 0] + h * s) / (1 - h / 2 * k[i, i])
    return x


def iter(y, h, x, n, k, f):
    yk = y.copy()
    for i in range(n):
        yk[i] = 0
        for j in range(i):
            yk[i] = yk[i] + 2 * k(x[i], x[j]) * y[j]
        yk[i] = yk[i] - k(x[i], x[0]) * y[0] - k(x[i], x[i]) * y[i]
        yk[i] = f(x[i]) + yk[i] * h / 2
    return yk


def solve_iter(k, f, x: np.ndarray, h: float, eps: float = 1e-1):
    n = len(x)
    y = f(x)
    yk = iter(y, h, x, n, k, f)
    i = 0
    err = np.linalg.norm(y - yk) / np.linalg.norm(y)
    while err > eps:
        y = yk.copy()
        yk = iter(y.copy(), h, x, n, k, f)
        err = error(y, yk)
        i += 1
        if i > 1000: break
    return yk, i


print(h)
x_outp_100, t_100 = quadratic(n, a, b)
error = x_outp_100 - func_exact(t_100)
plt.plot(t_100, func_exact(t_100))
plt.plot(t_100, x_outp_100)

n_noise = 50
n = 500
sample = [[0, 0]] * (n_noise + n)
x_upper_lim = 100
x_lower_lim = 0
sigma_noise = 4
noise_upper_lim = round(100 / np.e * sigma_noise)
noise_lower_lim = 0
for i in range(n_noise):
    sample[i] = [rd.randint(x_lower_lim, x_upper_lim), rd.randint(noise_lower_lim, noise_upper_lim)]
k_sample = 0.7
for i in range(n):
    x_temp = rd.randint(x_lower_lim, x_upper_lim)
    y_noise = rd.gauss(mu=50, sigma=sigma_noise)
    sample[i + n_noise] = [x_temp, k_sample * x_temp + y_noise]
test_sample = [[1, 1], [1, 10], [2, 3], [3, 2], [4, 5], [5, 4], [6, 8], [7, 5], [8, 8], [9, 10], [10, 10]]
sample_x = list(zip(*sample))[0]
sample_y = list(zip(*sample))[1]
