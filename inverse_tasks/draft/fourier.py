import numpy as np
from matplotlib import pyplot as plt
import scipy


def moduller(w):
    return w * w


def core(beta, x, dt, a):
    return np.exp(-beta * x) + a * np.exp(-beta * x - dt)


def x_func(t, f0, mu):
    return np.cos(2 * np.pi * f0 * t + mu * t * t)


def gaussian_noise(shape: tuple[int, int], mu: float = 0., sigma: float = 1e-2) -> np.ndarray:
    return np.random.normal(mu, sigma, size=shape)


beta = 3 * 10 ^ 3
a = 0.5
dt = 2 * 10 ^ -3
f0 = 300
mu = 8 * 10 ^ 5


def funct_integrate(low_bord, up_bord, step, t_sample):
    sample = np.linspace(low_bord, up_bord, int((up_bord - low_bord) / step))
    integral = np.empty_like(t_sample)
    for j in range(len(t_sample)):
        tau = np.linspace(low_bord, up_bord, int((up_bord - low_bord) / step))
        t = t_sample[j]
        integral_step = 0
        for i in range(len(sample)):
            int_coord = (tau[i]-tau[i-1])/2
            integral_step += core(beta, t - int_coord, dt, a) * x_func(int_coord, f0, mu) * step
        integral[j] = integral_step
    return integral


t_sample = np.linspace(0, 0.03, 100)
integ = funct_integrate(0, 0.03, 0.001, t_sample)

# plt.plot(t_sample, x_func(t_sample, f0, mu))
# plt.show()
# plt.plot(t_sample, core(beta, t_sample, dt, a))
# plt.show()
plt.plot(t_sample, integ)
plt.show()

alpha = 0.001

f_noise = integ
core_out = core(beta, t_sample, dt, a)
core_out_conj = core_out.conjugate()
f_fourier = np.fft.fft(f_noise)
core_fourier = np.fft.fft(core_out)
core_fourier_conj = np.fft.fft(core_out_conj)
x_freq = f_fourier * core_fourier / (core_fourier_conj * core_fourier + alpha * moduller(
    np.fft.fftfreq(len(t_sample), t_sample[1] - t_sample[0])))
x_freq_inv = np.real(np.fft.ifft(x_freq))
plt.plot(x_freq_inv)
plt.plot(x_func(t_sample, f0, mu))
plt.show()
