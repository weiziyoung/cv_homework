# -*- coding: utf-8 -*-
# @Time    : 13/10/2018 01:55
# @Author  : weiziyang
# @FileName: Image.py
# @Software: PyCharm
from datetime import datetime

import numpy as np
from numpy import fft
import matplotlib.pyplot as plt
import matplotlib.image as mat_img

import Kernel


class TypeException(Exception):
    def __init__(self, message):
        super().__init__()
        self.message = message


def calculate(matrix1, matrix2):
    """
    :param matrix1:ndarray
    :param matrix2:ndarray, contains the weight parameter
    :return:matrix1 .* matrix2
    """
    if matrix1.shape != matrix2.shape:
        raise Exception('The shape of Matrix1 and matrix2 should be same')
    return np.dot(matrix1.flatten(), matrix2.flatten().T)


class Image(object):
    pixels = [[]]
    shape = (0, 0, 0)  # if it is an RGB picture, shape will be a three-tuple
    size = (0, 0)  # M * N
    img_name = ''
    is_grey = False

    low_pass = None

    def __init__(self, matrix=None, image_path=None, name=None):
        if image_path:
            self.pixels = mat_img.imread(image_path)
            self.img_name = image_path.split('/')[-1].split('.')[0]
        if matrix is not None:
            self.pixels = matrix
            self.img_name = str(datetime.now()) + str(name)
        self.shape = self.pixels.shape
        # if it's a grey pic, we should flag it.
        if len(self.shape) == 3:
            if np.all(self.pixels[:, :, 0] == self.pixels[:, :, 1]) \
                    and np.all(self.pixels[:, :, 1] == self.pixels[:, :, 2]):
                self.pixels = self.pixels[:, :, 0]
                self.is_grey = True
            else:
                self.is_grey = False
        else:
            self.is_grey = True
        self.size = (self.shape[0], self.shape[1])

    def __repr__(self):
        return '{M} * {N} {COLOR} image, name:{name}\n{matrix}'.format(M=self.shape[0], N=self.shape[1],
                                                                       COLOR='GREY' if self.is_grey else 'RGB',
                                                                       name=self.img_name, matrix=self.pixels)

    def __str__(self):
        return self.__repr__()

    def __mul__(self, other):
        if isinstance(other, Kernel.Kernel):
            return self.convolute(other)
        else:
            raise Exception('Operator "*" is a convolution operator')

    def show(self, plot_loc=None):
        if plot_loc:
            plt.subplot(plot_loc)
        plt.imshow(self.pixels.astype(np.uint8), 'gray')
        plt.title(self.img_name)

    def convolute(self, kernel, fourier=None):
        """
        The implement of convolution.
        It will choose a better algorithm to do it depends on the size of kernel and image.

        :param kernel:
        :param fourier:Whether use fourier transform to calculate the convolution
        :return:
        """
        if not isinstance(kernel, Kernel.Kernel):
            try:
                kernel = Kernel.Kernel(kernel)
            except Exception:
                raise Exception('The input must be a kernel')

        kernel_size = kernel.shape[0] * kernel.shape[1]
        image_size = np.sqrt(self.shape[0] * self.shape[1])
        if fourier is not None:
            if fourier:
                return self.__fourier_convolution(kernel)
            else:
                return self.__base_convolution(kernel)

        # According to the Mark Nixon's book(p.87 3rd version),If m² < 4*log(N) + 1,
        #  then we should use direct implementation, otherwise the fourier transform should be considered.
        else:
            if kernel_size < 4 * np.log(image_size) + 1:
                print('regular')
                return self.__base_convolution(kernel)
            else:
                print('fourier')
                return self.__fourier_convolution(kernel)

    def __base_convolution(self, kernel):
        """
        Convolution calculation using standard loop(fast when the kernel size is small)
        :param kernel: The template of kernel
        :return: The output image that is convoluted by this kernel
        """
        # Initializing some parameters, since convolution is time-consuming, grey and RGB should be treated differently.
        if self.is_grey:
            pixels_list = [self.pixels]
        else:
            pixels_list = [self.pixels[:, :, i] for i in range(3)]
        channel_list = []

        for pixels in pixels_list:
            output_matrix = np.zeros(self.size)
            # Create a extension matrix based on the size of kernel, so as to do the convolution operation
            extra_m, extra_n = kernel.shape[0]//2, kernel.shape[1]//2
            temp_size = (self.shape[0] + extra_m * 2, self.shape[1] + extra_n * 2)
            temp_pixels = np.zeros(temp_size)
            temp_pixels[extra_m:extra_m+self.shape[0], extra_n:extra_n + self.shape[1]] = pixels

            # Start loop!
            for y_index, y in enumerate(range(extra_m, self.shape[0] + extra_m)):
                for x_index, x in enumerate(range(extra_n, self.shape[1] + extra_n)):
                    temp_matrix = temp_pixels[y-extra_m:y+extra_m+1, x-extra_n:x+extra_n+1]
                    temp_result = calculate(temp_matrix, kernel.array)
                    output_matrix[y_index, x_index] = temp_result
            channel_list.append(output_matrix)

        # return output image, treat grey and RGB pic separately
        if self.is_grey:
            return Image(channel_list[0])
        else:
            output = np.zeros(self.shape)
            for i in range(3):
                output[:, :, i] = channel_list[i]
            return Image(matrix=output)

    def __fourier_convolution(self, kernel):
        """
        A more faster convolution way when the size of kernel is large
        :param kernel:The template of kernel
        :return:The output image that is convoluted by this kernel
        """
        # Create a padding matrix:
        output_matrix = np.zeros(self.size if self.is_grey else self.shape)
        for channel in range(1 if self.is_grey else 3):
            pixels = self.pixels[:, :, channel] if not self.is_grey else self.pixels
            padding_matrix = np.zeros(self.size)
            start_point = (self.shape[0]//2 - kernel.shape[0]//2, self.shape[1]//2 - kernel.shape[1]//2)
            padding_matrix[start_point[0]:start_point[0]+kernel.shape[0],
                           start_point[1]:start_point[1]+kernel.shape[1]] = kernel.array
            image_transform = (fft.fft2(pixels))
            template_transform = (fft.fft2(padding_matrix))
            inverted_transform = np.abs(fft.fftshift(fft.ifft2(image_transform * template_transform)))
            if not self.is_grey:
                output_matrix[:, :, channel] = inverted_transform
            else:
                output_matrix = inverted_transform
        return Image(matrix=output_matrix)

    def low_pass_filter(self, sigma, size=None, fourier=False):
        """
        Filter implemented by Gaussian Kernel
        :param fourier:
        :param sigma: int, should be 1.0, 1.5 or anything else
        :param size: size of the kernel
        :return:
        """
        if size:
            size = size
        else:
            size = int(8 * sigma + 1)
        low_filter_kernel = Kernel.GaussianKernel(size, sigma=sigma)
        self.low_pass = self.convolute(low_filter_kernel, fourier=fourier)
        self.low_pass.img_name = self.img_name + ' low-pass filter'
        return self.low_pass

    def high_pass_filter(self, sigma=None, size=None, fourier=None):
        if not self.low_pass and not sigma:
            raise Exception('The sigma should provided')
        else:
            if not self.low_pass:
                self.low_pass_filter(sigma, size=size, fourier=fourier)
            temp_result = self.pixels - self.low_pass.pixels
            result_min = temp_result.min()
            result_max = temp_result.max()
            result = temp_result + abs(result_min)
            interval = result_max - result_min
            if interval > 255:
                ratio = 255/interval
                result *= ratio
            return Image(result, name=self.img_name + ' high pass filter')

