import matplotlib.pyplot as plt
import numpy as np


def integration(function, a=0, b=1, step=0.001):
    # trapezoidal integration
    num_steps = int(np.abs(b - a) / step)
    sample = np.linspace(a, b, num_steps)
    f_output = 0
    f_output += 1 / 2 * function(a) * step
    f_output += 1 / 2 * function(b) * step
    for i in range(1, num_steps - 1):
        f_output += 1 / 2 * function(a + step * i) * step
    return f_output, sample


def integr_test(function, analytic, a=0, b=1, step=0.001):
    num_steps = int(np.abs(b - a) / step)
    sample = np.linspace(a, b, num_steps)
    error = analytic(sample) - integration(function)
    assert np.max(error) > 1e-7, "integration incorrect"


def function(x):
    return np.exp(x - 2) * 2 * x


def analytic(x):
    return 2 * np.exp(x - 2) * (x - 1)


f_out, t_sample = integration(function)
f_anal = analytic(t_sample)

plt.plot(t_sample, f_out)
plt.plot(t_sample, f_anal)
plt.show()
