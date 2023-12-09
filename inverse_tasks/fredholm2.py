import numpy as np
from matplotlib import pyplot as plt
import random as rd
from scipy.integrate import cumtrapz, simpson, solve_ivp

epsilon = 1e-2


# fredholm second kind
# looking for the solution of the problem:
# y(x)-lambda*Int[g(x)*h(t)*y(t) {t,a,b}] == f(x)
# case 1: h(t) = 1, g(x) = sin(beta*x)
# solution: y(x) = f(x) + lambda*k*g(x)
# k = Int[h(t)*f(t) {t,a,b}] / (1 - lambda*Int[g(t)*h(t) {t,a,b}]
# that means: core = sin(beta*x)
# core_type = 0, beta = 2
#
# case 2: h(t) = sin(beta*t), g(x) = 1
# solution: y(x) = f(x) + lambda*k*g(x)
# k = Int[h(t)*f(t) {t,a,b}] / (1 - lambda*Int[g(t)*h(t) {t,a,b}]
# that means: core = sin(beta*t)
# core_type = 0, beta = 2
#
# watching for different f(x)

def rel_err(result, exact):
    non_norm_err = (exact - result) / result
    return np.linalg.norm(non_norm_err)


def core_fredh(t, tau, lambd=1, type_func=0, beta=2):
    N = len(t)
    K = np.empty((N, N))
    for i in range(N):
        for j in range(i + 1):
            if type_func == 0:
                K[i, j] = np.sin(beta * t[i])
            elif type_func == 1:
                K[i, j] = np.cos(lambd * t[i])
            elif type_func == 2:
                K[i, j] = np.sin(beta * tau[j])
            elif type_func == 3:
                K[i, j] = np.cos(lambd * (t[i] - tau[j]))
            elif type_func == 4:
                K[i, j] = np.cos(lambd * tau[j]) / np.cos(lambd * t[i])
    return K


def func_fredg(t, N, beta=2, type_func=1, noise=0):
    func = np.empty(N)
    for i in range(N):
        if type_func == 1:
            func[i] = 1
        elif type_func == 2:
            func[i] = np.cos(t[i])
        elif type_func == 3:
            func[i] = np.exp(beta * t[i])
    if noise == 1:
        func += np.random.rand(N)
    return func


def solve_quad(f, h, x, t, tau, lambd=1, beta=2, core_type=0):
    N = len(f)
    x[0] = f[0]
    K_loc = core_fredh(t, tau, lambd=lambd, type_func=core_type, beta=beta)
    for i in range(1, N):
        s = 0
        for n in range(1, i):
            s += K_loc[n, i] * x[n]
        # s = np.sum(K_loc[i, 1:i] * x[1:i])
        x[i] = f[i] + (K_loc[i, i] + h / 2 * K_loc[i, 0] + h*s) / (1 - h / 2 * K_loc[i, i])
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


def solve_iter(k, f, x: np.ndarray, h: float):
    n = len(x)
    y = f(x)
    yk = iter(y, h, x, n, k, f)
    i = 0
    err = np.linalg.norm(y - yk) / np.linalg.norm(y)
    while err > epsilon:
        y = yk.copy()
        yk = iter(y.copy(), h, x, n, k, f)
        err = rel_err(y, yk)
        i += 1
        if i > 1000:
            break
    return yk, i


def fredh_exact(core_type, func_type, t, beta=2, lmbd=1):
    if core_type == 0 and func_type == 1:
        func_ret = np.array(1 + (np.sin(beta * t) * lmbd) / (1 - (lmbd * (1 - np.cos(beta))) / (beta)))
    else:
        func_ret = None
    # calling some troubles with wolfram
    #
    # if core_type == 1 and func_type == 1:
    #     func_ret = np.array((1 - lmbd * np.cos(lmbd * t) + np.sin(lmbd)) / (1 + np.sin(lmbd)))
    # elif core_type == 1 and func_type == 2:
    #     func_ret = np.array(
    #         (np.cos(t) + lmbd * np.cos(lmbd * t) * np.sin(1) + np.cos(t) * np.sin(lmbd)) / (1 + np.sin(lmbd)))
    # elif core_type == 1 and func_type == 3:
    #     func_ret = np.array((np.exp(t * beta) * beta + lmbd * np.cos(t * lmbd) - np.exp(beta) * lmbd * np.cos(
    #         t * lmbd) + np.exp(beta * t) * beta * np.sin(lmbd)) / (beta * (1 + np.sin(lmbd))))
    # else TBD
    return func_ret


a = 0
b = 10
step = 1e-2
N = int(np.abs(a - b) / step)

t = np.linspace(a, b, N)
#t = np.logspace(a, 4, 10)
f_fredg = func_fredg(t, N, type_func=1)
x_fredg = np.zeros_like(f_fredg)
x_fredg = solve_quad(f_fredg, step, x_fredg, t, t, core_type=2)
# x_ex = [2 * np.exp(t[i]) - 1 for i in range(N)]
x_ex = fredh_exact(0, 1, t)
K_loc = core_fredh(t, t, lambd=1, type_func=0, beta=2)
plt.plot(t, K_loc)
plt.plot(t, x_ex, label="Exact sol.")
plt.plot(t, x_fredg, label="Numerical sol.")
plt.legend()
plt.show()
