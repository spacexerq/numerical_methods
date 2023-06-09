import random

import numpy as np
from fractions import Fraction
from numbers import Number


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
            matrix = self.zero(self.height, other.width, self[0, 0] - self[0, 0])
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
        return self.data[row, column]

    def __setitem__(self, key, value):
        row, column = key
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
        for i in range(self.height):
            for j in range(i + 1):
                temp = 0
                for k in range(j):
                    temp += matrix[i, k] * matrix[j, k]
                if i == j:
                    matrix[i, j] = np.sqrt(self[i, i] - temp)
                else:
                    matrix[i, j] = (self[i, j] - temp) * matrix.invert_element(matrix[j, j])
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
        self._height = len(data)
        self._width = self._height

    @property
    def width(self):
        return self.shape[0]

    @property
    def height(self):
        return self.shape[0]

    @property
    def shape(self):
        return [self.shape[0], self.shape[0]]

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, key):
        row, column = key
        return self.data[abs(row - column)]

    def __setitem__(self, key, value):
        #a[i,j] = a.s[i-j(mod n)]
        row, column = key
        self.data[abs(row - column)] = value

    def empty_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]
        data = np.empty((height, width), dtype=dtype)
        return ToeplitzMatrix(data)

    '''
    def transpone(self):
        matrix = self.zero(self.width, self.height, self[0, 0] - self[0, 0])
        for i in range(self.width):
            for j in range(self.height):
                matrix[i, j] = self[j, i]
        return matrix

    

    def ident_like(self, width=None, height=None):
        dtype = self.data.dtype
        if width is None:
            width = self.data.shape[1]
        if height is None:
            height = self.data.shape[0]
        data = np.empty((height, width), dtype=dtype)
        for i in range(self.height):
            data[i, i] = self[i, i] * self.invert_element(self[i, i])
        return ToeplitzMatrix(data)

    @classmethod
    def zero(_cls, height, width, default=0, low_bandw=0, upp_bandw=0):
        """
        Создает матрицу размера `width` x `height` со значениями по умолчанию `default`.
        """
        data = np.empty((height, width), dtype=type(default))
        data[:] = default
        return ToeplitzMatrix(data)
        
    def solve_levi(self, vector):
        vec_result
        return vec_result
    '''

error = 1e-10


def equal(a, b):
    if isinstance(a, Number) and isinstance(b, Number):
        if abs(a - b) < error:
            return True
        else:
            return False
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        if abs(a - b) < error:
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
