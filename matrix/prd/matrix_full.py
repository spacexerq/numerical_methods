import random

import numpy as np
from fractions import Fraction
from numbers import Number
import matplotlib.pyplot as plt


class TextBlock:
    def __init__(self, rows):
        assert isinstance(rows, list)
        self.rows = rows
        self.height = len(self.rows)
        self.width = max(map(len, self.rows))

    @classmethod
    def from_str(_cls, data):
        assert isinstance(data, str)
        return TextBlock(data.split('\n'))

    def format(self, width=None, height=None):
        if width is None: width = self.width
        if height is None: height = self.height
        return [f"{row:{width}}" for row in self.rows] + [' ' * width] * (height - self.height)

    @staticmethod
    def merge(blocks):
        return [" ".join(row) for row in zip(*blocks)]


class Matrix:
    """Общий предок для всех матриц."""

    @property
    def shape(self):
        raise NotImplementedError

    @property
    def dtype(self):
        raise NotImplementedError

    @property
    def width(self):
        return self.shape[1]

    @property
    def height(self):
        return self.shape[0]

    def __repr__(self):
        """Возвращает текстовое представление для матрицы."""
        text = [[TextBlock.from_str(f"{self[r, c]}") for c in range(self.width)] for r in range(self.height)]
        width_el = np.array(list(map(lambda row: list(map(lambda el: el.width, row)), text)))
        height_el = np.array(list(map(lambda row: list(map(lambda el: el.height, row)), text)))
        width_column = np.max(width_el, axis=0)
        width_total = np.sum(width_column)
        height_row = np.max(height_el, axis=1)
        result = []
        for r in range(self.height):
            lines = TextBlock.merge(
                text[r][c].format(width=width_column[c], height=height_row[r]) for c in range(self.width))
            for l in lines:
                result.append(f"| {l} |")
            if len(lines) > 0 and len(lines[0]) > 0 and lines[0][0] == '|' and r < self.height - 1:
                result.append(f'| {" " * (width_total + self.width)}|')
        return "\n".join(result)

    def empty_like(self, width=None, height=None):
        raise NotImplementedError

    def ident_like(self):
        raise NotImplementedError

    def __getitem__(self, key):
        raise NotImplementedError

    def __setitem__(self, key, value):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.width and self.height == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] + other[r, c]
            return matrix
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.width and self.height == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = self.empty_like()
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] - other[r, c]
            return matrix
        return NotImplemented

    def __mul__(self, other):
        return self.__matmul__(other)

    def __matmul__(self, other):
        # multiplication righthanded only (matrix*number)
        if isinstance(other, Matrix):
            assert self.width == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            matrix = FullMatrix.zero(self.height, other.width, self[0, 0] - self[0, 0])
            for r in range(self.height):
                for c in range(other.width):
                    acc = None
                    for k in range(self.width):
                        add = self[r, k] * other[k, c]
                        acc = add if acc is None else acc + add
                    matrix[r, c] = acc
            return matrix
        elif isinstance(other, Number):
            matrix = self.zero(self.height, self.width, self[0, 0] - self[0, 0])
            for r in range(self.height):
                for c in range(self.width):
                    matrix[r, c] = self[r, c] * other
            return matrix
        return NotImplemented

    def __truediv__(self, other):
        if isinstance(other, Matrix):
            assert self.width == other.width and self.height == other.height, f"Shapes does not match: {self.shape} != {other.shape}"
            divider = self.inverse()
            matrix = self * divider
            return matrix
        return NotImplemented

    def inverse(self):
        l, u = self.lu()
        null = self[0, 0] - self[0, 0]
        l_inv = l.empty_like()
        u_inv = u.empty_like()
        for i in range(self.height):
            for j in range(self.width):
                if i == j:
                    u_inv[i, i] = u.invert_element(u[i, i])
                    l_inv[i, j] = l.invert_element(l[i, j])
                elif i > j:
                    u_inv[i, j] = null
                    temp = null
                    for k in range(i):
                        temp += l_inv[k, j] * l[i, k]
                    l_inv[i, j] = -l.invert_element(l[i, i]) * temp
                else:
                    l_inv[i, j] = null
                    temp = null
                    for k in range(j):
                        temp += u_inv[i, k] * u[k, j]
                    u_inv[i, j] = -u.invert_element(u[j, j]) * temp
        return u_inv * l_inv

    def invert_element(self, element):
        if isinstance(element, Number):
            return 1 / element
        if isinstance(element, Fraction):
            return 1 / element
        if isinstance(element, Matrix):
            return element.inverse()
        raise TypeError

    def lu(self):
        raise NotImplementedError

    def det(self):
        assert self.width == self.height, f"Matrix is not square: {self.height} != {self.width}"
        l, u = self.lu()
        det = 1
        for i in range(u.height):
            det *= u[i, i]
        return det

    def lup(self):
        temp = self
        p = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        for i in range(self.height):
            if self[i, i] != 0:
                p[i, i] = self[0, 0] / self[0, 0]
        for i in range(self.height):
            ref_val = 0
            ref_num = -1
            for j in range(i, self.width):
                if np.abs(temp[j, i]) >= ref_val:
                    ref_val = np.abs(temp[j, i])
                    ref_num = j
            if ref_val != 0:
                temp.swap_rows(ref_num, i)
                p.swap_rows(ref_num, i)
                for j in range(i + 1, self.height):
                    temp[j, i] /= temp[i, i]
                    for k in range(i + 1, self.height):
                        temp[j, k] -= temp[j, i] * temp[i, k]
        u = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        l = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        for i in range(self.height):
            for j in range(self.height):
                if i == j:
                    u[i, j] = temp[i, j]
                    l[i, j] = 1
                elif i < j:
                    u[i, j] = temp[i, j]
                    l[i, j] = 0
                else:
                    l[i, j] = temp[i, j]
                    u[i, j] = 0
        return temp, l, u, p

    def swap_rows(self, num1, num2):
        matrix = self
        temp = self.empty_like()
        for i in range(self.height):
            temp[num1, i] = self[num1, i]
            matrix[num1, i] = self[num2, i]
            matrix[num2, i] = temp[num1, i]
        return matrix

    def swap_cols(self, num1, num2):
        matrix = self
        temp = self.empty_like()
        for i in range(self.width):
            temp[i, num1] = self[i, num1]
            matrix[i, num1] = self[i, num2]
            matrix[i, num2] = temp[i, num1]
        return matrix

    def transpone(self):
        raise NotImplementedError

    def zero(self, width, height, param, low_bandw, upp_bandw):
        raise NotImplementedError

    def solve(self, vector):
        # only for invertible matrices
        if isinstance(vector, Matrix):
            assert vector.width == 1 and self.width == vector.height and self.height == self.width, f"Vector or matrix shape is wrong: {self.shape}, {vector.shape}"
            garbage, l, u, p = self.lup()
            y_vec = vector.empty_like()
            pb = p * vector
            result = vector.empty_like()
            temp = 0
            flag = 0
            # solving Ly=Pb
            for i in range(self.height):
                y_vec[i, 0] = (pb[i, 0] - temp) / l[i, i]
                temp = 0
                flag += 1
                for j in range(i + 1):
                    if i < self.height - 1:
                        temp += l[i + 1, j] * y_vec[j, 0]
            # solving Ux=y
            temp = 0
            for i in range(self.height - 1, -1, -1):
                result[i, 0] = (y_vec[i, 0] - temp) / u[i, i]
                temp = 0
                for j in range(self.height - 1, i - 1, -1):
                    temp += u[i - 1, j] * result[j, 0]
            return result
        return NotImplemented

    def qr(self):
        # old version some mistakes
        q = FullMatrix.empty_like(self)
        u = FullMatrix.empty_like(self)
        u[:, 0] = self[:, 0].data
        # q[:, 0] = u[:, 0] / np.sqrt(np.sum(u[:, 0] ** 2))
        q[:, 0] = u[:, 0] / np.linalg.norm(u[:, 0])
        for i in range(1, self.height):
            u[:, i] = self[:, i].data
            for j in range(i):
                scal1 = np.sum([self.data[k, i] * q.data[k, j] for k in range(self.height)])
                scal = np.sum([self[k, i] * q[k, j] for k in range(self.height)])
                u[:, i] -= (scal * q[:, j])
            q[:, i] = u[:, i] / np.linalg.norm(u[:, i])
        r = FullMatrix.zero(self.width, self.width, 0.0)
        for i in range(self.height):
            for j in range(i, self.width):
                r[i, j] = self[:, j].dot(q[:, i])
        return q, r

    def qr_new(self):
        # gram_schmidt complete for any sizes
        q = FullMatrix.zero(self.height, self.height, 0.0)
        cnt = 0
        num_rows = self.width
        num_cols = self.height
        mat_tr = self.transpone()
        for i in range(num_rows):
            u = mat_tr[i, :]
            for j in range(0, cnt):
                temp = q[:, j] @ mat_tr[i, :]
                temp *= q[:, j]
                u -= temp
            e = u / np.linalg.norm(u.data)
            q[:, cnt] = e
            cnt += 1
        r = q.transpone() * self
        return q, r

    def lsm_qr(self):
        assert self.width == 2, "Vector array of points has incorrect shape, needs (n,2) has: "
        num_values = self.height
        A = FullMatrix.zero(num_values, 2, 1.0)
        A[:, 0] = self[:, 0]
        y = FullMatrix.zero(num_values, 1, 0.0)
        y[:, 0] = self[:, 1]
        q_A, r_A = A.qr_new()
        r = r_A.data
        r_trunc = r[:r.shape[1], :]
        x = FullMatrix.zero(r_A.width, 1, 0.)
        qy = q_A.transpone() * y
        for i in range(r.shape[1], 0, -1):
            x[i - 1, 0] = (qy[i - 1, 0] - (r_trunc[i - 1, :i - 1:-1].dot(x.data[:i - 1:-1, 0]))) / r_trunc[i - 1, i - 1]
        Ax = A.data.dot(x.data)
        x_sample = self[:, 0].data
        return list(x_sample), Ax

    def lsm_svd(self):
        assert self.width == 2, "Vector array of points has incorrect shape, needs (n,2) has: "
        num_values = self.height
        A = FullMatrix.zero(num_values, 2, 1.0)
        A[:, 0] = self[:, 0]
        y = FullMatrix.zero(num_values, 1, 0.0)
        y[:, 0] = self[:, 1]
        u, sigma, vt = np.linalg.svd(self.data)
        sigma = np.diag(sigma)
        print(u)
        print(sigma)
        print(vt)
        sigma_inv = np.linalg.inv(sigma)
        print(sigma_inv)
        matrix = u@sigma_inv@vt
        x = matrix@y.data
        return self[:, 0].data, x


class FullMatrix(Matrix):
    """
    Заполненная матрица с элементами произвольного типа.
    """

    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data

    def transpone(self):
        matrix = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        for i in range(self.width):
            for j in range(self.height):
                matrix[i, j] = self[j, i]
        return matrix

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]
        data = np.empty((height, width), dtype=dtype)
        return FullMatrix(data)

    def ident_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]
        data = np.empty((height, width), dtype=dtype)
        for i in range(self.height):
            data[i, i] = self[i, i] * self.invert_element(self[i, i])
        return FullMatrix(data)

    def lu(self):
        assert self.width == self.height, f"Matrix is not square: {self.height} != {self.width}"
        u = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        l = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        for i in range(self.height):
            l[i, i] = self[0, 0] / self[0, 0]
        for i in range(self.height):
            for j in range(self.height):
                if i <= j:
                    temp = u[0, 0] - u[0, 0]
                    for k in range(i + 1):
                        temp = temp + l[i, k] * u[k, j]
                    u[i, j] = self[i, j] - temp
                else:
                    temp = u[0, 0] - u[0, 0]
                    for k in range(j + 1):
                        temp = temp + l[i, k] * u[k, j]
                    l[i, j] = (self[i, j] - temp) * u.invert_element(u[j, j])
        return l, u

    @classmethod
    def zero(_cls, height, width, default=0, low_bandw=0, upp_bandw=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, width), dtype=type(default))
        data[:] = default
        return FullMatrix(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        row, column = key
        if type(row) == int and row < 0:
            row = self.width - row % self.width
        if type(column) == int and column < 0:
            column = self.height - column % self.height
        return self.data[row, column]

    def __setitem__(self, key, value):
        row, column = key
        if type(row) == int and row < 0:
            row = self.width - row % self.width
        if type(column) == int and column < 0:
            column = self.height - column % self.height
        self.data[row, column] = value


class SymmetricMatrix(Matrix):
    """
    Симметричная матрица
    """

    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]
        data = np.empty((height, width), dtype=dtype)
        return SymmetricMatrix(data)

    def ident_like(self, width=None, height=None):
        data = self.empty_like()
        for i in range(self.height):
            data[i, i] = self[i, i] * self.invert_element(self[i, i])
            for j in range(self.height):
                if i != j:
                    data[i, j] = self[0, 0] - self[0, 0]
        return data

    def lu(self):
        # cholecky decomposition used for this (l i not more uni-left-triangle)
        matrix = FullMatrix.zero(self.width, self.height, self[0, 0] - self[0, 0])
        matrix[0,0] = np.sqrt(self[0,0])
        for i in range(self.height):
            for j in range(i + 1):
                temp = 0
                for k in range(j):
                    sl1 = matrix[i, k]
                    sl2 = matrix[j, k]
                    plus = sl1 * sl2
                    temp += plus
                if i == j:
                    matrix[i, j] = np.sqrt(self[i, i] - temp)
                else:
                    print(matrix[j, j])
                    inv = matrix.invert_element(matrix[j, j])
                    matrix[i, j] = (self[i, j] - temp) * inv
        l, u = FullMatrix.zero(self.width, self.height, self[0, 0] - self[0, 0]), FullMatrix.zero(self.width,
                                                                                                  self.height,
                                                                                                  self[0, 0] - self[
                                                                                                      0, 0])
        for i in range(self.height):
            for j in range(self.width):
                if j > i:
                    l[i, j] = 0
                    u[i, j] = matrix[i, j]
                elif j < i:
                    u[i, j] = 0
                    l[i, j] = matrix[i, j]
                else:
                    u[i, j] = matrix[i, j]
                    l[i, j] = matrix[i, j]
        return l, u

    @classmethod
    def zero(_cls, width, height, default=0, low_bandw=0, upp_bandw=0):
        """
        Создает матрицу размера `height` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, height), dtype=type(default))
        data[:] = default
        return SymmetricMatrix(data)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        row, column = key
        return self.data[row, column]

    def __setitem__(self, key, value):
        row, column = key
        self.data[row, column] = value
        self.data[column, row] = value


class BandMatrix(Matrix):
    """
    Заполненная матрица с элементами произвольного типа.
    """

    def __init__(self, data, low_bandw, upp_bandw):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data
        self.lw_bw = low_bandw
        self.up_bw = upp_bandw

    def transpone(self):
        matrix = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        for i in range(self.width):
            for j in range(self.height):
                matrix[i, j] = self[j, i]
        return matrix

    @classmethod
    def zero(_cls, height, width, default=0, low_bandw=0, upp_bandw=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, low_bandw + upp_bandw + 1), dtype=type(default))
        data[:] = default
        return BandMatrix(data, low_bandw, upp_bandw)

    @property
    def shape(self):
        real_shape = (self.data.shape[0], self.data.shape[0])
        return real_shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def low_bandw(self):
        return self.lw_bw

    @property
    def upp_bandw(self):
        return self.up_bw

    def __getitem__(self, key):
        row, column = key
        if row < column - self.lw_bw or row > column + self.up_bw:
            return 0
        else:
            return self.data[row, self.upp_bandw + row - column - 1]

    def __setitem__(self, key, value):
        row, column = key
        self.data[row, self.upp_bandw + row - column - 1] = value

    def lu(self):
        assert self.width == self.height, f"Matrix is not square: {self.height} != {self.width}"
        u = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        l = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        for i in range(self.height):
            if self[i, i] != 0:
                l[i, i] = self[0, 0] / self[0, 0]
        for i in range(self.height):
            for j in range(self.height):
                if i <= j:
                    temp = u.zero(u.width, u.height, u[0, 0] - u[0, 0])
                    for k in range(i + 1):
                        temp[i, j] = temp[i, j] + l[i, k] * u[k, j]
                    u[i, j] = self[i, j] - temp[i, j]
                elif i > j:
                    temp = u.zero(u.width, u.height, u[0, 0] - u[0, 0])
                    for k in range(j + 1):
                        temp[i, j] = temp[i, j] + l[i, k] * u[k, j]
                    l[i, j] = (self[i, j] - temp[i, j]) * u.invert_element(u[j, j])
        return l, u


class ToeplitzMatrix(Matrix):
    """
    Матрица Тёплица.
    """

    def __init__(self, data):
        """
        Создает объект, хранящий матрицу в виде np.ndarray `data`.
        """
        assert isinstance(data, np.ndarray)
        self.data = data

    @property
    def width(self):
        return int((self.data.shape[0] + 1) / 2)

    @property
    def height(self):
        return int((self.data.shape[0] + 1) / 2)

    @property
    def shape(self):
        real_shape = (int((self.shape[0] + 1) / 2), int((self.shape[0] + 1) / 2))
        return real_shape

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        row, column = key
        return self.data[row - column]

    def __setitem__(self, key, value):
        # a[i,j] = a.s[i-j(mod n)]
        row, column = key
        self.data[row - column] = value

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]
        data = np.empty((height, width), dtype=dtype)
        return ToeplitzMatrix(data)

    @classmethod
    def zero(_cls, height, width, default=0.0, low_bandw=0, upp_bandw=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty(height + width - 1, dtype=type(default))
        data[:] = default
        return ToeplitzMatrix(data)

    def solve_levi(self, vector):
        assert vector.height == self.width, "Vector size does not conduct with matrix" + " v - " + str(
            vector.height) + " m - " + str(self.width)
        to = [0.0] * self.height
        back = [0.0] * self.height
        vec_res = [0.0] * self.height
        to[0] = [self.invert_element(self[0, 0])]
        back[0] = [self.invert_element(self[0, 0])]
        vec_res[0] = [vector[0, 0] / self[0, 0]]

        for i in range(self.height - 1):
            forw = to[i]
            backw = back[i]
            e_to = 0
            e_back = 0
            for j in range(i + 1):
                e_to += self[i + 1, j] * forw[j]
                e_back += self[0, j + 1] * backw[j]
            det = 1 - e_to * e_back
            to[i + 1] = 1 / det * np.concatenate((to[i], [0])) - e_to / det * np.concatenate(([0], back[i]))
            back[i + 1] = to[i + 1][::-1]

        for i in range(self.height - 1):
            e_to = 0
            res_temp = vec_res[i]
            for j in range(i + 1):
                e_to += self[i + 1, j] * res_temp[j]
            vec_res[i + 1] = np.concatenate([vec_res[i], [0]]) + (vector[i + 1, 0] - e_to) * back[i + 1]
        len_res = len(vec_res)
        x_v = FullMatrix.zero(len_res, 1, 0.0)
        for i in range(len_res):
            x_v[i, 0] = vec_res[self.height - 1][i]
        return x_v


numerical_error = 9e-7


def equal(a, b):
    if isinstance(a, Number) and isinstance(b, Number):
        if abs(a - b) < numerical_error:
            return True
        else:
            return False
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        if abs(a - b) < numerical_error:
            return True
        else:
            return False
    if isinstance(a, Matrix) and isinstance(b, Matrix):
        if a.shape != b.shape:
            return False
        for i in range(a.height):
            for j in range(a.width):
                if not equal(a[i, j], b[i, j]):
                    return True
        return True
    raise TypeError


def laplace(n, n2=float(random.randrange(0, 10000))):
    mat = np.zeros((n, n, n, n), dtype=int)
    n1 = -4 * n2
    for i1 in range(n):
        for i2 in range(n):
            mat[i1, i2, i1, i2] = n1
            mat[i1, i2, (i1 - 1) % n, i2] = n2
            mat[i1, i2, (i1 + 1) % n, i2] = n2
            mat[i1, i2, i1, (i2 - 1) % n] = n2
            mat[i1, i2, i1, (i2 + 1) % n] = n2
    mat_res = mat.reshape((pow(n, 2), pow(n, 2)))
    return mat, mat_res


def D(n):
    mat, mat_res = laplace(n)
    mat_toep = ToeplitzMatrix.zero(len(mat_res[0, :]), len(mat_res[0, :]), 0.0)
    for i in range(len(mat_res[0, :])):
        mat_toep[0, i] = mat_res[0, i]
        mat_toep[i, 0] = mat_res[0, i]
    return mat_toep, mat_res


def vectors(n):
    sample = np.linspace(-1, 1, n)
    x_v = np.ravel([[not_homogen_part(x, y) for x in sample] for y in sample])
    x_v -= np.sum(x_v.T)
    d = D(n)[0]
    mat_d = FullMatrix.zero(d.height, d.width, 0.0)
    for i in range(d.height):
        for j in range(d.width):
            mat_d[i, j] = d[i, j]
    y_v = mat_d.data.dot(x_v.T)
    X = FullMatrix.zero(len(x_v), 1, 0.0)
    Y = FullMatrix.zero(len(x_v), 1, 0.0)
    for i in range(len(x_v)):
        X[i, 0] = x_v[i]
        Y[i, 0] = y_v[i]
    return X, Y, x_v, y_v


def not_homogen_part(x, y):
    return np.cos(np.pi * x) * np.cos(np.pi * y)


def fourier(num_values):
    num_sample = np.linspace(-1, 1, num_values)
    x_analyt = [[not_homogen_part(x, y) for x in num_sample] for y in num_sample]
    x_analyt -= np.sum(x_analyt)
    x_analyt = np.ravel(x_analyt)
    mat, mat_res = laplace(num_values)
    F_laplace = np.fft.ifft2(np.fft.fft2(mat, axes=[0, 1]), axes=[2, 3])
    y = mat_res.dot(x_analyt.T)
    F_y = np.fft.fft2(y.reshape((num_values, num_values)))
    diagonal = np.array([F_laplace.reshape((num_values ** 2, num_values ** 2))[i, i] for i in range(num_values ** 2)])
    F_x = np.ravel(F_y) / diagonal
    F_x[0] = 0
    result = mat_res.dot(np.fft.ifft2((F_x.reshape((num_values, num_values)))).reshape((num_values ** 2,)).T)
    return result, y
