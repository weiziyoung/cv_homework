# -*- coding: utf-8 -*-
# @Time    : 06/11/2018 19:51
# @Author  : weiziyang
# @FileName: util.py
# @Software: PyCharm

import datetime


def count_time(func):
    def int_time(*args, **kwargs):
        start_time = datetime.datetime.now()  # start time
        result = func(*args, **kwargs)
        over_time = datetime.datetime.now()   # end time
        total_time = (over_time-start_time).total_seconds()
        print('This function %s costs %s seconds' % (func.__name__, total_time))
        return result
    return int_time