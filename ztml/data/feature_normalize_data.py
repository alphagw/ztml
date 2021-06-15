#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/9 15:07:24'

import pandas as pd
import numpy as np
from ztml.tools import auto_get_corelated_group
import matplotlib.pyplot as plt


def read_data():
    # csv_file = r'G:\ztml\ztml\data\clean_data.csv'
    csv_file = r'G:\ztml\ztml\data\1.csv'
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


def get_normalize_data(data, gogal_column=None):
    
    if gogal_column is not None:
        pass
    else:
        column = data.columns.values.tolist()[:-2]
        train_data = data.values[:, :-2]
        a = np.corrcoef(train_data, rowvar=0)
        b = np.abs(np.triu(a))
        gp = auto_get_corelated_group(data=b, coref_val=0.85, is_triu=True, get_remove=True)
        print('corelated column:', len(gp[1]))
        for i in sorted(gp[1], reverse=True):
            column.pop(i)
        gogal_column = column
        
    feature = data[gogal_column]
    ffe = normalize(feature.values)
    print(gogal_column)
    print("final column: ", len(gogal_column))
    # plt_each_dis(ffe, 'nor.pdf')
    # plt_each_dis(feature.values, '0.pdf')
    gogal_column.append(data.columns.values.tolist()[-2])
    gogal_column.append(data.columns.values.tolist()[-1])
    
    label = data[gogal_column[-2:]]
    
    return pd.DataFrame(np.hstack((ffe, label)), columns=gogal_column), gogal_column[:-2]
    

def run():
    data = read_data()
    now_data, _ = get_normalize_data(data)
    now_data.to_csv('clean_data_normalized.csv')
    
    
if __name__ == '__main__':
    run()
