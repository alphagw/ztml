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
from copy import deepcopy
from ztml.tools import read_data


def add_tempeture(data):
    all_label = data.columns.get_values()
    tempeture = [str(i) for i in range(100, 700, 50)]
    t_index = ["nop_%sK", "%s_K"]
    new_ii = ["Nop", "ZT"]
    share_label = [tl for tl in all_label if tl not in [ti % tm for tm in tempeture for ti in t_index]]
    # share_label.remove('A1B2C4')
    
    copydata = deepcopy(data[share_label])

    num = 0
    fn_data = None
    for tm in tempeture:
        tmp_data = deepcopy(copydata)
        tmp_data.insert(tmp_data.shape[1], 'T', tm)
    
        for ti in range(len(t_index)):
            tmp_data.insert(tmp_data.shape[1], new_ii[ti], data[t_index[ti] % tm])
        
        num += 1
        
        if num == 1:
            fn_data = tmp_data
        else:
            fn_data = fn_data.append(tmp_data)
        
    return pd.DataFrame(fn_data.get_values(), columns=fn_data.columns.get_values())


def change_np(data):
    # _tt = {"p": 1, "N": 0, "P": 1, "n": 0}
    for i in data.index.tolist():
        # data.loc[i, 'Type'] = _tt[data['Type'][i].strip()]
        data.loc[i, 'Nop'] = 0 if data['Nop'][i] > 0 else 1

    return data


def get_clean_data(new_data, is_nop_to_01=True):
    # 添加12个温度参数，并将所有温度对应的Kpa，N，ZT三个值添加上, 将70组数据扩展为840组数据
    add_t_data = add_tempeture(new_data)
    
    # 将P类型还是N类型半导体改成数值类型 P型：0  N型：1
    if is_nop_to_01:
        fdata = change_np(add_t_data)
    else:
        fdata = add_t_data

    # 将Type列放置到倒数第2列，删除N_optimal列
    # columns = fdata.columns.tolist()
    # columns.insert(-1, columns.pop(columns.index('Type')))
    # new_pf = pd.DataFrame(fdata, columns=columns)
    # print(fdata)
    #
    # data = fdata['N_optimal']
    # import numpy as np
    #
    # data = sorted(data.tolist())
    # print(np.max(data))
    # print(np.min(data))
    # print(np.mean(data))
    # print(np.median(data))
    # exit()
    return fdata


def write(fdata, gfn):
    fdata.to_csv(gfn, index=False)
    

def run(fn, gfn):
    new_data = read_data(fn)
    data = get_clean_data(new_data)
    write(data, gfn)


if __name__ == '__main__':
    # tfn = "20201203_descriptors.csv"
    # run(r'1-train_30.csv', r'2-train_30clean_data.csv')
    pass
