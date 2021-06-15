#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/15 21:59:17'

import os
import pandas as pd
from ztml.tools import get_train_test_index


def deal_data(fn, fn1, fn2):
    data = pd.read_csv(fn)
    vt_data, vtest_data = get_train_test_index(data, column_index=['NC_atom_unit'], ratio=0.749, to_data=True)
    data = vt_data[:-1]
    vtest_data = vtest_data.append(vt_data.iloc[-1])
    data.to_csv(fn1, index=False)
    vtest_data.to_csv(fn2, index=False)


if __name__ == '__main__':
    head_dir = r'..\data'
    fn = os.path.join(head_dir, r'valid_40.csv')
    fn1 = r'10_for_check.csv'
    fn2 = r'30_for_predict.csv'
    deal_data(fn, os.path.join(head_dir, fn1), os.path.join(head_dir, fn2))

