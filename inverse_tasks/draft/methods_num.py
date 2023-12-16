import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import numba as nb

from scipy.fft import fftfreq, fft, ifft  # tikhanov_regularization


def gaussian_noise(shape: tuple[int, int], mu: float = 0., sigma: float = 1e-2) -> np.ndarray:
    return np.random.normal(mu, sigma, size=shape)


def relative_error(exact: np.ndarray, solution: np.ndarray) -> float:
    return np.linalg.norm(exact - solution) / np.linalg.norm(solution)


@nb.jit("float32(float32[:],float32[:])")
def norm_pow_2(exact, solution):
    delta = exact - solution
    res = sum(delta ** 2)
    return res


def plot_settings():
    plt.rcParams["axes.facecolor"] = '#0d1117'
    plt.rcParams["figure.facecolor"] = '#0d1117'

    # plt.rcParams['figure.figsize'] = [7.0, 3.0]
    plt.rcParams['figure.dpi'] = 100

    plt.rcParams["legend.labelcolor"] = 'w'
    plt.rcParams["axes.titlecolor"] = "w"

    # plt.rcParams["axes.spines.bottom.color"]
    # plt.rcParams["axes.spines.left"] = '#0d1117'
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.top"] = False

    plt.rcParams["axes.edgecolor"] = "#eef7f4"

    plt.rcParams["xtick.color"] = '#eef7f4'
    plt.rcParams["ytick.color"] = '#eef7f4'

    plt.rcParams["axes.labelcolor"] = '#eef7f4'

    plt.rcParams["grid.color"] = '#eef7f4'

    plt.rcParams["legend.frameon"] = False

    plt.rcParams['axes.prop_cycle'] = cycler(color=['g', 'r', 'b', 'y'])


def tikhanov_regularization(f: np.ndarray, k: np.ndarray, t: np.ndarray,
                            alpha: float = 0.1, m=lambda o: np.abs(o)) -> np.ndarray:
    """
    f: right part of eq
    k: kernel
    t: t.shape = f.shape
    m: lambda o: np.abs(o)
    return:
        solution of eq (real)
    """

    F = fft(f)
    K = fft(k, len(f))
    # print(f.shape[0], t[1] - t[0])
    omega = fftfreq(f.shape[0], d=t[1] - t[0])

    x_alpha_o = np.empty_like(f)
    # print(K.conjugate(), K)
    x_alpha_o = F * np.conjugate(K) / (np.conjugate(K) * K + alpha * m(omega))

    x_alpha = ifft(x_alpha_o)

    return np.real(x_alpha)