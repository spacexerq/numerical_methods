import methods_num
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import random as rd


def primal_ransac(k, b, epsilon, array):
    alpha = np.arctan(k)
    x_sw = epsilon * np.cos(alpha)
    y_sw = epsilon * np.sin(alpha)
    new_array = []
    len_sample = len(array)
    for j in range(len_sample):
        x_p = array[j][0]
        y_p = array[j][1]
        y_bound_low = k * x_p + b + y_sw
        y_bound_up = k * x_p + b - y_sw
        condition = y_bound_low >= y_p >= y_bound_up
        if condition:
            new_array.append(array[j])
    return new_array


def line(point1, point2):
    k = (point1[1] - point2[1]) / (point1[0] - point2[0])
    b = point1[1] - k * point1[0]
    return k, b


def ransac(array, epsilon, prob_ratio=0.75):
    len_sample = len(array)
    points_used = [[0] * len_sample for _ in range(len_sample)]
    K = int(np.log(1 - prob_ratio) / (np.log(1 - prob_succ(len_sample, int(prob_ratio * len_sample))[0])))
    print("Number of iterations needed is", K)
    output_array = []
    k_out = None
    b_out = None
    len_outp = 0
    for i in range(int(len_sample * (len_sample - 1) / 2) - 1):
        p1 = rd.randint(0, len_sample - 1)
        p2 = rd.randint(0, len_sample - 1)
        if p1 != p2 and array[p1][0] != array[p2][0]:
            if points_used[p1][p2] == 0:
                k, b = line(array[p1], array[p2])
                new_array = primal_ransac(k, b, epsilon, array)
                points_used[p1][p2] = 1
                points_used[p2][p1] = 1
                if len(new_array) > len_outp:
                    len_outp = len(new_array)
                if len(new_array) >= int(prob_ratio * len_sample):
                    output_array = new_array
                    k_out = k
                    b_out = b
                    break
    return output_array, k_out, b_out, len_outp


def prob_succ(num_values, num_trusted):
    p_1 = np.math.factorial(num_trusted) * np.math.factorial(num_values - 2) / (
            np.math.factorial(num_values) * np.math.factorial(num_trusted - 2))
    return p_1, (1 - p_1)


n_noise = 50
n = 200
sample = [[0, 0]] * (n_noise + n)
for i in range(n_noise):
    sample[i] = [rd.randint(0, 100), rd.randint(0, 100)]
k_sample = 0.1
dispertion = 20
for i in range(n):
    x_temp = rd.randint(0, 100)
    y_noise = rd.randint(0, dispertion)
    sample[i + n_noise] = [x_temp, k_sample * x_temp + y_noise]
sample_x = list(zip(*sample))[0]
sample_y = list(zip(*sample))[1]
result, k_out, b_out, len_out = ransac(sample, dispertion, prob_ratio=0.75)
if len(result) != 0:
    result_x = list(zip(*result))[0]
    result_y = list(zip(*result))[1]
    plt.plot(result_x, result_y, "o", color="black", markersize=5)
else:
    print("RANSAC did not find the solution")
plt.plot(sample_x, sample_y, "o", color="green", markersize=2.5)
plt.show()
