import numpy as np
import matplotlib.pyplot as plt
import random as rd


def minmax_(array):
    array_x = list(zip(*array))[0]
    array_y = list(zip(*array))[1]
    y_min = min(array_y)
    y_max = max(array_y)
    x_min = min(array_x)
    x_max = max(array_x)
    return {"x_max": x_max, "y_max": y_max, "x_min": x_min, "y_min": y_min}


def primal_ransac(k, b, epsilon, array):
    alpha = np.arctan(k)
    y_sw = epsilon * np.sin(alpha)
    new_array = []
    len_sample = len(array)
    # x_p = 0
    # x_lim = minmax_(array)["x_max"]
    # y_bound_low = str_line(x_p, k, (b + y_sw))
    # y_bound_up = str_line(x_p, k, (b - y_sw))
    # plt.plot((x_p, x_lim), (y_bound_low, str_line(x_lim, k, (b + y_sw))), color="red")
    # plt.plot((x_p, x_lim), (y_bound_up, str_line(x_lim, k, (b - y_sw))), color="orange")
    for j in range(len_sample):
        x_p = array[j][0]
        y_p = array[j][1]
        y_bound_low = k * x_p + b + y_sw
        y_bound_up = k * x_p + b - y_sw
        condition = y_bound_low >= y_p >= y_bound_up
        if condition:
            new_array.append(array[j])
    return new_array


def str_line(x, k, b):
    y = k * x + b
    return y


def line(point1, point2):
    k = (point1[1] - point2[1]) / (point1[0] - point2[0])
    b = point1[1] - k * point1[0]
    return k, b


def ransac(array, epsilon, prob_ratio=0.75, show_steps=False):
    len_sample = len(array)
    points_used = [[0] * len_sample for _ in range(len_sample)]
    K = 2 * len_sample
    # K = int(np.log(1 - prob_ratio) / (np.log(1 - prob_succ(len_sample, int(prob_ratio * len_sample))[0])))
    print("Expect number of iterations needed is", K)
    output_array = []
    k_out = None
    b_out = None
    iterations = None
    len_outp = 0
    # int(len_sample * (len_sample - 1) / 2) - 1
    for i in range(K):
        p1 = rd.randint(0, len_sample - 1)
        p2 = rd.randint(0, len_sample - 1)
        if p1 != p2 and array[p1][0] != array[p2][0] and points_used[p1][p2] == 0 and array[p1][1] != array[p2][1]:
            k, b = line(array[p1], array[p2])
            if show_steps:
                plt.plot([array[p1][0], array[p2][0]], [array[p1][1], array[p2][1]], color="blue")
            new_array = primal_ransac(k, b, epsilon, array)
            points_used[p1][p2] = 1
            points_used[p2][p1] = 1
            if len(new_array) > len_outp and len(new_array) >= int(prob_ratio * len_sample):
                len_outp = len(new_array)
                output_array = new_array
                k_out = k
                b_out = b
                iterations = i
    if show_steps and len_outp == 0:
        plt.show()
    return output_array, k_out, b_out, len_outp, iterations


def prob_succ(num_values, num_trusted):
    p_1 = np.math.factorial(num_trusted) * np.math.factorial(num_values - 2) / (
            np.math.factorial(num_values) * np.math.factorial(num_trusted - 2))
    return p_1, (1 - p_1)


def lstsqr(array):
    array_x = list(zip(*array))[0]
    array_y = list(zip(*array))[1]
    A = np.vstack([array_x, np.ones(len(array_x))]).T
    m, c = np.linalg.lstsq(A, array_y, rcond=None)[0]
    x_out_min = minmax_(array)["x_min"]
    x_out_max = minmax_(array)["x_max"]
    y_out_min = m * x_out_min + c
    y_out_max = m * x_out_max + c
    plt.plot((x_out_min, x_out_max), (y_out_min, y_out_max), color="yellow", label='Least squares')
    return {"k": m, "b": c}


def report_ransac(input_flag=0):
    flag = "Y"
    dispersion = 1
    while True:
        if flag != "Y":
            print("Calculation finish.")
            break
        else:
            dispersion = round(dispersion * 1.5)
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
        result, k_out, b_out, len_out, iterations = ransac(sample, dispersion, prob_ratio=0.85, show_steps=False)
        print("Number of iterations calculated:", iterations)
        plt.plot(sample_x, sample_y, "o", color="green", markersize=2.5)
        if len(result) != 0:
            print("Result found for", dispersion, "dispersion")
            result_x = list(zip(*result))[0]
            result_y = list(zip(*result))[1]
            x_out_min = minmax_(result)["x_min"]
            x_out_max = minmax_(result)["x_max"]
            y_out_min = str_line(x_out_min, k_out, b_out)
            y_out_max = str_line(x_out_max, k_out, b_out)
            plt.plot(result_x, result_y, "o", color="black", markersize=5)
            plt.plot(sample_x, sample_y, "o", color="green", markersize=2.5)
            plt.plot([x_out_min, x_out_max], [y_out_min, y_out_max], color="red", linewidth=2, label="RANSAC")
            k_lsqr = lstsqr(sample)["k"]
            plt.legend()
            plt.show()
            print("k sample:", round(k_sample, 3), "k RANSAC:", round(k_out, 3), "k LsSqr:", round(k_lsqr, 3))
            flag = "No"
        else:
            print('\n' + "RANSAC did not find the solution")
            print("Result for", dispersion, "dispersion")
            print("Do you want to repeat with higher dispersion?")
            print("Y/N?")
            if input_flag:
                flag = input()
            else:
                flag = "Y"
