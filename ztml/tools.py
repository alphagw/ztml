#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/10 16:29:05'

import numpy as np
import pandas as pd


def read_data(csv_file):
    data = pd.read_csv(csv_file)
    return data


def norepeat_randint(data: list, ratio=0.3):
    """
    输入一个list，将其且分为训练集和测试集
    :param data:  原始数据
    :param ratio: 测试集比例
    :return: 测试集
    """
    num = round(len(data) * ratio)
    a = np.random.randint(0, len(data), num)
    if a.shape[0] != 0:
        if len(list(set(a.tolist()))) != num:
            dd = list(range(len(data)))
            np.random.shuffle(dd)
            return dd[:num]
        else:
            return a
    else:
        return None
    

def get_train_test_index(data, column_index, ratio, to_data=False):
    """
    读取csv数据，根据指定列先分组，并在将每一组都分成训练集和测试集，返回测试集的索引
    :param data
    :param column_index: 列表，列名构成的列表
    :param ratio: 测试集比例
    :param retrun_data: 决定返回数据还是索引
    :return: 训练集和测试集的索引 或者 训练集和测试集数据
    """
    dd = data.groupby(column_index)
    np.random.seed(31)
    test_index = []
    for i, j in dd.groups.items():
        d = j[norepeat_randint(j, ratio=ratio)]
        test_index.extend(d.tolist())
    c = np.ma.array(data.index.tolist(), mask=False)
    c.mask[test_index] = True
    train_index = c.compressed().tolist()
    
    if to_data:
        return pd.DataFrame(data.iloc[train_index].values, columns=data.columns), \
               pd.DataFrame(data.iloc[test_index].values, columns=data.columns)
    else:
        return train_index, test_index


def get_random_groupby_index(data, column_index, ratio, to_data=False,
                             test_index=None, train_index=None):
    """
    读取csv数据，根据指定列先分组，并在将每一组都分成训练集和测试集，返回测试集的索引
    :param data
    :param column_index: 列表，列名构成的列表
    :param ratio: 测试集比例
    :param retrun_data: 决定返回数据还是索引
    :return: 训练集和测试集的索引 或者 训练集和测试集数据
    """
    dd = data.groupby(column_index)
    # num = data.shape[0]
    np.random.seed(31)
    # _tmp = list(dd.groups.keys())
    # if train_index is not None and len(train_index) < int(num * ratio):
    #     ratio = (int(num * ratio) - len(train_index)) / num
    #     _tmp_train_index = norepeat_randint(range(len(dd.groups)), ratio=ratio)
    #     train_index.extend(_tmp_train_index)
    # elif train_index is not None and len(train_index) == int(num * ratio):
    #     pass
    # elif train_index is not None and len(train_index) > int(num * ratio):
    #     raise ValueError("train_index: %d > %d" %(len(train_index), int(num * ratio)))
    # else:
    #     train_index = norepeat_randint(range(len(dd.groups)), ratio=ratio)
    if train_index is not None:
        pass
    else:
        train_index = []
        _tmp_index = norepeat_randint(range(len(dd.groups)), ratio=ratio)
        for jdd, (kk, ss) in enumerate(dd.groups.items()):
            if jdd in _tmp_index:
                train_index.append(int(data.loc[dd.groups[kk][0]]['Index'] * 69 + 1))
        
    if test_index is not None:
       pass
    else:
        test_index = []
        
    test_data, train_data = [], []
    for j, (m, n) in enumerate(dd.groups.items()):
        jj = int(data.loc[dd.groups[m][0]]['Index'] * 69 + 1)
        if jj in train_index:
            train_data.append(data.loc[dd.groups[m]])
        else:
            if jj not in test_index:
                test_index.append(j)
            test_data.append(data.loc[dd.groups[m]])
            
    train_data = pd.concat(train_data, ignore_index=True, copy=True)
    test_data = pd.concat(test_data, ignore_index=True, copy=True)
    
    if to_data:
        return train_data, test_data
    else:
        return train_index, test_index


def method2(index_final):
    lindex_final = list(index_final.tolist())
    final_d = []
    bd = np.zeros(index_final.shape[0], dtype=np.int) + 1
    for i in range(0, index_final.shape[0]):
        if bd[i]:
            dd = list(index_final[i])
            for j in range(len(lindex_final)):
                if bd.any() == 0:
                    break
                else:
                    if bd[j]:
                        x, y = lindex_final[j][0], lindex_final[j][1]
                        if (x in dd) or (y in dd):
                            bd[j] = 0
                            dd.extend([x, y])
            final_d.append(dd)
        else:
            continue
    return [list(set(t)) for t in final_d]


def auto_get_corelated_group(data, coref_val=0.9, is_triu=False, get_remove=True):
    """
    自动获取相关性较高的数组
    :param data:
    :param coref_val:
    :param is_triu:
    :param get_remove:
    :return:
    """
    if is_triu:
        pass
    else:
        data = np.triu(data)
    
    double_index = np.where(data > coref_val)
    index_final = np.array([[double_index[0][m], double_index[1][m]]
                            for m in range(len(double_index[0]))
                            if double_index[0][m] != double_index[1][m]])
    
    gp = method2(index_final)
    
    if get_remove:
        sv, rm = [], []
        for i in gp:
            i = sorted(i)
            sv.append(i[0])
            for m in i[1:]:
                rm.append(m)
        # print(gp)
        return sv, list(set(rm))
    else:
        return gp
