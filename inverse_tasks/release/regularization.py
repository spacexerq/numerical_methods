import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import random as rd
from matplotlib.animation import FuncAnimation
from methods_num import *
from scipy.fft import fftfreq, fft, ifft
from scipy import optimize as opt


# Int[K(t, tau) x(tau) {a,b,tau}] = f(t)

def core(t, tau):
    result = np.exp(-beta * t) + A * np.exp(-beta * t - delta_t)
    # result = np.exp(-beta * t)
    return result


def x_var(var):
    result = np.cos(2 * np.pi * f0 * var + mu * var * var)
    return result


def exact_solution(t_l, h_l):
    x_exact = x_var(t_l)
    core_loc = core(t_l, 0)
    # plt.plot(t_l, x_exact, label="Exact x")
    # plt.show()
    f_sol = np.convolve(core_loc, x_exact)
    n_c = len(f_sol)
    b_c = h_l * n_c
    t_for_convolve = np.linspace(a, b_c, n_c)
    # plt.plot(t_for_convolve, f_sol)
    # plt.show()
    return f_sol, t_for_convolve


def make_noisy(f_l, t_l, sigma_l=2e-1, mu_l=0):
    f_l += np.random.normal(mu_l, sigma_l, size=f_l.shape)
    return f_l


def tikhonov_regular(f_l, core_loc, t_step, alpha=1e-1):
    f_four = fft(f_l)
    k_four = fft(core_loc, len(f_four))
    w = fftfreq(f_l.shape[0], t_step)
    solution_k_sp = f_four * np.conjugate(k_four) / (np.conjugate(k_four) * k_four + alpha * np.abs(w))
    solution_r_sp = np.real(ifft(solution_k_sp))
    return solution_r_sp


def rel_err(exact, solution):
    return np.linalg.norm(exact - solution) / np.linalg.norm(solution)


def search_for_alpha_supp(sigma, t_l, h_l, alpha_t):
    f1, t1 = exact_solution(t_l, h_l)
    f1_n = make_noisy(f1, t1, sigma_l=sigma)
    sol_t = tikhonov_regular(f1_n, core(t_l, 0), h_l, alpha=alpha_t)
    return sol_t[:t.shape[0]]


def search_for_alpha(t_l, h_l, sigma_loc, sample_size=500):
    alpha0 = 1e-3
    alpha_out1 = []
    x_exact = x_var(t_l)
    for i in range(sample_size):
        f_aim = lambda alpha: rel_err(x_exact,
                                      search_for_alpha_supp(sigma_loc, t_l, h_l, alpha)[:t.shape[0]])
        res = opt.minimize(f_aim, alpha0)
        alpha_out1.append(res.x[0])
    return alpha_out1


def alpha_hist_output(t_l, h_l, sigma_loc):
    alpha1 = search_for_alpha(t_l, h_l, sigma_loc)
    f1_loc, t1_loc = exact_solution(t_l, h_l)
    f1_n_loc = make_noisy(f1_loc, t1_loc, sigma_l=sigma_loc)
    plt.plot(t1_loc, f1_n_loc)
    plt.show()
    plt.text(-7, 20,  "sigma = " + str(sigma_loc), bbox=dict(boxstyle="round",
                   ec=(1., 0.5, 0.5),
                   fc=(1., 0.8, 0.8),
                   ))
    plt.hist(np.log(np.abs(alpha1)))
    plt.show()


A = 0.5
delta_t = 2 * 1e-3
beta = 3 * 1e3
f0 = 300
mu = 8 * 1e5
a = 0
b = 0.005
h = 1e-1
# N = int((b-a)/h)
N = 1000
t = np.linspace(a, b, N)

# exact solution
x_exact_plt = x_var(t)
plt.plot(t, x_exact_plt, label="Exact x")
plt.show()

# noisy function
f1, t1 = exact_solution(t, h)
f1_n = make_noisy(f1, t1, sigma_l=1e-1)
plt.plot(t1, f1_n)
plt.show()
# finding x via regularization
sol_tikh = tikhonov_regular(f1_n, core(t, 0), h, alpha=1e-2)
plt.plot(t, x_var(t))
plt.plot(t, sol_tikh[:t.shape[0]], '--')
plt.show()

sigma0 = 1e-3
sigma1 = 5e-1
sigma2 = 1e0
alpha_hist_output(t, h, sigma0)
alpha_hist_output(t, h, sigma1)
alpha_hist_output(t, h, sigma2)
