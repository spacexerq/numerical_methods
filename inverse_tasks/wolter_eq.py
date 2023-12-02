import numpy as np
from matplotlib import pyplot as plt

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


print(h)
x_outp_100, t_100 = quadratic(n, a, b)
error = x_outp_100 - func_exact(t_100)
plt.plot(t_100, func_exact(t_100))
plt.plot(t_100, x_outp_100)