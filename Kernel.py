# -*- coding: utf-8 -*-
# @Time    : 13/10/2018 01:55
# @Author  : weiziyang
# @FileName: Image.py
# @Software: PyCharm

import numpy as np
from util import count_time


class ShapeException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


class Kernel(object):
    """
        The Template of Kernel.
    """
    array = None
    shape = None

    def __init__(self, array):
        try:
            if isinstance(array, list):
                array = np.array(array)
            self.shape = array.shape
            if len(self.shape) != 2:
                raise ShapeException('The input shape must be a M*N matrix or 2D list')
            if not self.shape[0] % 2 or not self.shape[1] % 2:
                raise ShapeException('M or N must be an odd number')
            self.array = array
        except Exception as e:
            raise Exception(e)

    def __str__(self):
        return '{M} * {N} Kernel: \n {Array}'.format(M=self.shape[0], N=self.shape[1], Array=self.array)

    def __repr__(self):
        return self.__str__()


class GaussianKernel(Kernel):
    """
        A special Kernel based on Gaussian function:
        g(x,y,sigma) = 1/(2πσ^2) * e^(-(x^2+y^2)/2*σ^2)
        The low-pass filter can be implemented by this
    """
    def __init__(self, size, sigma=1):
        if isinstance(size, int):
            size = (size, size)
        if not size[0] % 2 or not size[1] % 2:
            raise ShapeException('M or N must be an odd number')
        m, n = size[0], size[1]
        center_m, center_n = (m//2, n//2)
        array = np.zeros(size)
        for y in range(m):
            for x in range(n):
                array[y, x] = (1/(2*np.pi*sigma**2))*np.e**(-((y-center_m)**2 + (x-center_n)**2)/(2*sigma**2))
        super().__init__(array)