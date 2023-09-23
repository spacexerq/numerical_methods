from matrix_full import *


def test_lu():
    num_sample = [1, 3, 10, 100, 250, 400]
    for n in num_sample:
        m_full = FullMatrix.zero(n, n, float)
        for i in range(n):
            for j in range(n):
                m_full[i, j] = float(random.randrange(0, 10000))
        if not equal(m_full.lu()[0] * m_full.lu()[1], m_full):
            print("LU decomposition is not complete for full matrix")
            return -1
    return 0


def test_lup():
    num_sample = [10, 100, 500]
    for n in num_sample:
        m_full = FullMatrix.zero(n, n, float)
        for i in range(n):
            for j in range(n):
                m_full[i, j] = float(random.randrange(0, 10000))
        temp, l, u, p = m_full.lup()
        if not equal(l * u * p, m_full):
            print("LUP decomposition is not complete for full matrix")
            return -1
    return 0


def test_laplace():
    for n in range(1, 30):
        matrix = D(n)[0]
        x, y, x_v, y_v = vectors(n)
        vec_result = matrix.solve_levi(y)
        for i in range(n):
            if np.abs(vec_result[i, 0] - x_v[i]) > numerical_error:
                print("x_num:")
                print(vec_result)
                print("x_anal:")
                print(x)
                print("Error occurred", n)
                print(vec_result[i, 0], x_v[i])
                print(np.abs(vec_result[i, 0] - x_v[i]))
                return -1
    return 0


def test_qr():
    num_sample = np.linspace(1, 500, 6, dtype=int)
    for n in num_sample:
        m_full = FullMatrix.zero(n, n, float)
        for i in range(n):
            for j in range(n):
                m_full[i, j] = float(random.randrange(0, 10000))
        if not equal(m_full.qr()[0] * m_full.qr()[1], m_full):
            print("LU decomposition is not complete for full matrix")
            return -1
    return 0


def test_fourier():
    num_sample = np.linspace(1, 15, 5, dtype=int)
    # for matrices about 50 elements evaluating too long
    for n in num_sample:
        n_real = 4 * n
        res, y_res = fourier(n_real)
        if not np.any(np.abs(res - y_res) <= numerical_error):
            print("Fourier transformation does not complete")
            return -1
    return 0


def test_qr_lsm():
    num_sample = np.logspace(1, 4, 5, dtype=int)
    for n in num_sample:
        vector = FullMatrix.zero(n, 2, 0.0)
        for i in range(n):
            for j in range(2):
                vector[i, j] = float(random.randrange(-10000, 10000))
        x, y = vector.lsm_qr()
        res_poly = np.polyfit(x, vector[:, 1].data, 1)
        res_poly = res_poly.flatten()
        poly_np = np.poly1d(res_poly)
        if not np.any(np.abs(poly_np(x) - y) <= numerical_error):
            print("Error in least sqares method (by QR)")
            return -1
    return 0


lu = test_lu()
lup = test_lup()
qr = test_qr()
laplace = test_laplace()
qr_lsm = test_qr_lsm()
fourier = test_fourier()

test = lu + lup + qr + laplace + qr_lsm + fourier
if test == 0:
    print("All tests executed correctly")
