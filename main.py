# -*- coding: utf-8 -*-
# @Time    : 13/10/2018 15:48
# @Author  : weiziyang
# @FileName: main.py
# @Software: PyCharm

import matplotlib.pyplot as plt

import Image


def main():
    cat_image = Image.Image(image_path='data/cat.bmp')
    high_image1 = cat_image.high_pass_filter(sigma=7)
    dog_image = Image.Image(image_path='data/dog.bmp')
    low_image = dog_image.low_pass_filter(sigma=7)
    mix = Image.Image(high_image1.pixels * 0.5 + low_image.pixels * 0.5)
    low_image.show(221)
    high_image1.show(222)
    mix.show(223)
    plt.show()
    pass


if __name__ == "__main__":
    main()