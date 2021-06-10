#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/10 10:07:24'

import pandas as pd
import numpy as np


def read_data(csv_file):
    data = pd.read_csv(csv_file)
    return data


def norepeat_randint(data: list, ratio=0.3):
    num = round(len(data) * ratio)
    a = np.random.randint(0, len(data), num)
    if len(list(set(a.tolist()))) != num:
        dd = list(range(len(data)))
        np.random.shuffle(dd)
        return dd[:num]
    else:
        return a


def get_test_index(csv_file, column_index, ratio):
    data = read_data(csv_file)
    dd = data.groupby(column_index)
    np.random.seed(27)
    test_index = []
    for i, j in dd.groups.items():
        d = j[norepeat_randint(j, ratio=ratio)]
        test_index.extend(d.tolist())
    return test_index


def run():
    origin_csv_file = r'G:\ztml\ztml\data\clean_data.csv'
    column_index = ['Temperature', 'N_atom_unit']
    ratio = 0.3
    test_index = get_test_index(origin_csv_file, column_index, ratio)
    data = read_data(r'G:\ztml\ztml\data\clean_data_normalized.csv')
    
    # get train data
    c = np.ma.array(data.index.tolist(), mask=False)
    c.mask[test_index] = True
    ti = c.compressed().tolist()
    train_data = data.iloc[ti]
    train_data.to_csv('train_data_from_normalized_data.csv', index=False)

    # get test data
    test_data = data.iloc[test_index]
    test_data.to_csv('test_data_from_normalized_data.csv', index=False)

    
if __name__ == '__main__':
    run()
