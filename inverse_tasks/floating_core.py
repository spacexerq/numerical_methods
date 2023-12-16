import numpy as np

from regularization import *


def report_noisy_core():
    f1, t1 = exact_solution(t, h)
    sample_betas = 100
    betas = np.linspace(beta, beta * 1.5, sample_betas)
    xs = np.empty(sample_betas)
    error_1 = np.empty(sample_betas)
    error_ex = np.empty(sample_betas)
    error_ex_1 = np.empty(sample_betas)
    f_noise = make_noisy(f1, t1)

    for i in range(sample_betas):
        a = optimal_alpha(sigma, k_(t, betas[i]), f_noise)
        xs[i] = tikhanov_regularization(f_noise, k_(t, betas[i]), t, a)[:t.shape[0]]
        a = optimal_alpha(sigma, k_(t), f_noise)
        x_b = tikhanov_regularization(f_noise, k_(t), t, a)[:t.shape[0]]

        error_1[i] = np.linalg.norm((xs[i] - x_b)) ** 2
        error_ex[i] = np.linalg.norm((x_exact - x_b)) ** 2
        error_ex_1[i] = np.linalg.norm((x_exact - xs[i])) ** 2


report_noisy_core()
