#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2020/12/06 10:06:47'

import pandas as pd
import numpy as np
from ztml.data.run_rm_correlation_plt import auto_get_corelated_group
import matplotlib.pyplot as plt


def read_data():
    csv_file = r'G:\ztml\ztml\data\clean_data.csv'
    data = pd.read_csv(csv_file)
    return data


def plt_each_dis(dd, fn=None):
    fig, axes = plt.subplots(4, 9)
    num = 0
    for i in axes:
        for m in i:
            if num < 34:
                m.hist(dd[:, num])
                num += 1
    if fn:
        plt.savefig(fn)
    else:
        plt.show()
    return None


def normalize(data):
    dmin = np.min(data, axis=0)
    dmax = np.max(data, axis=0)
    scale = 1 / (dmax - dmin)
    dd = (data - dmin) * scale
    return dd


def run():
    data = read_data()
    column = data.columns.values.tolist()[:-2]
    train_data = data.values[:, :-2]
    a = np.corrcoef(train_data, rowvar=0)
    b = np.triu(a)
    gp = auto_get_corelated_group(data=b, coref_val=0.85, is_triu=True, get_remove=True)
    print(len(gp[1]))
    for i in sorted(gp[1], reverse=True):
        column.pop(i)
    
    feature = data[column]
    ffe = normalize(feature.values)
    
    # plt_each_dis(ffe, 'nor.pdf')
    # plt_each_dis(feature.values, '0.pdf')
    column.append(data.columns.values.tolist()[-2])
    column.append(data.columns.values.tolist()[-1])

    label = data[column[-2:]]
    
    now_data = np.hstack((ffe, label))
    
    pd.DataFrame(now_data, columns=column).to_csv('clean_data_normalized.csv')
    return ffe, label
    
    
if __name__ == '__main__':
    run()
