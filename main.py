# -*- coding: utf-8 -*-
# @Time    : 13/10/2018 15:48
# @Author  : weiziyang
# @FileName: main.py
# @Software: PyCharm

import matplotlib.pyplot as plt

import Image


def main():
    plt.figure(figsize=(20, 12))
    image1 = Image.Image(image_path='data/cat.bmp')
    high_image = image1.high_pass_filter(sigma=1, fourier=False)
    image2 = Image.Image(image_path='data/dog.bmp')
    low_image = image2.low_pass_filter(sigma=1, fourier=False)
    mix = high_image.mix(low_image, ratio=0.5)
    low_image.show(231)
    high_image.show(233)
    mix.show(is_hybrid=True)
    plt.show()


if __name__ == "__main__":
    main()