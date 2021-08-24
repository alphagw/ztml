#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/10 16:39:37'

from ztml.tools import get_random_groupby_index, norepeat_randint
from ztml.data.feature_normalize_data import get_normalize_data

from ztml.rdata.clean_csv_data import get_clean_data, read_data
from ztml.rdata.rename_cloumn import get_rename_column_data
from copy import deepcopy


def read_rename_clean_datafile():
    tmp_file = 'temp_clean_data.csv'
    clean_train_data = read_data(tmp_file)
    # os.remove(tmp_file)
    
    # 获取归一化之后数据，并删除相关列
    normalized_data, _ = get_normalize_data(clean_train_data)
    normalized_data.to_csv('normalized_data.csv', index=False)


def __rm_point_columns(data, index=None):
    if index is None:
        index = ["Index"]
        
    ll = data.columns.get_values().tolist()
    for i in index:
        ll.remove(i)
    return data[ll]


def run_compounds_split(fn, head_dir, is_nop_to_01=True):
    """
    根据化合物分割训练集合测试集，并输出到CSV文件中
    :param fn: 原始数据文件名
    :param head_dir:
    :param is_nop_to_01: 控制掺杂类型是分类问题还是连续问题, 是否将nop转化为0-1，0：大于0， 1： 小于0
    :return: None
    """
    
    # 重命名每一列
    import os
    if not os.path.isdir(head_dir):
        os.mkdir(head_dir)
    
    origin_data = read_data(fn)
    rename_data = get_rename_column_data(origin_data)
    # train_data, test_data = get_train_test_index(rename_data, column_index=['N_atom_unit'], ratio=0.575, to_data=True)

    # 添加12个温度参数，并将所有温度对应的Kpa，N，ZT三个值添加上, 将70组数据扩展为840组数据
    # is_nop_to_01
    clean_train_data = get_clean_data(rename_data, is_nop_to_01=is_nop_to_01)
    tmp_file = os.path.join(head_dir, 'temp_clean_data.csv')
    clean_train_data.to_csv(tmp_file, index=False)
    clean_train_data = read_data(tmp_file)
    # os.remove(tmp_file)

    # 获取归一化之后数据，并删除相关列
    # normalized_data, _ = get_normalize_data(clean_train_data, gogal_column=['Index',
    #                                                                         'NA', 'NB', 'NC', 'NT', 'VA', 'VB', 'VC',
    #                                                                         'RA', 'RB', 'RC', 'ZA', 'ZB', 'ZC', 'rA',
    #                                                                         'rB', 'rC', 'rCA', 'rCB', 'rCC', 'PA',
    #                                                                         'PB', 'PC', 'PAv', 'PSd', 'a_b', 'c',
    #                                                                         'MAv', 'LAC', 'LBC', 'LAv',
    #                                                                         'Temperature'])
    normalized_data, _ = get_normalize_data(clean_train_data)
    __rm_point_columns(normalized_data).to_csv(os.path.join(head_dir, 'normalized_data.csv'), index=False)
    # normalized_data = read_data('normalized_data.csv')
    # 根据总原子（化合物）将数据分为30个训练和40个验证集
    # 实际Excel中的Index值
    # valid_index = [1, 2, 3, 5, 11, 14, 20, 22, 27, 29, 30, 32, 35, 37, 38, 41, 43, 44, 45, 49,
    #                50, 51, 52, 54, 56, 58, 59, 64, 65, 67]
    check_index = [12, 18, 21, 26, 34, 36, 39, 53, 60, 69]
    train_index = [4, 6, 7, 8, 9, 10, 13, 15, 16, 17, 19, 23, 24, 25, 27, 28, 33, 40, 42, 46, 47, 48,
                   55, 57, 61, 62, 63, 66, 68, 70]
    use2train_data, use2valid_data = get_random_groupby_index(normalized_data, column_index=['Index'],
                                                              ratio=3/7, to_data=True, train_index=train_index)
    print(use2train_data.shape, use2valid_data.shape)
    # 将训练及在次分为训练集和测试集
    train_train_data, train_test_data = get_random_groupby_index(use2train_data, column_index=['Index'],
                                                                 ratio=0.7, to_data=True)
    print(train_train_data.shape, train_test_data.shape)
    
    valid_10data, valid_30data = get_random_groupby_index(use2valid_data, column_index=['Index'],
                                                          ratio=0.25, to_data=True, train_index=check_index)
    print(valid_10data.shape, valid_30data.shape)
    
    print("Final features: %d" % (train_train_data.shape[1] - 3))
    # 输出数据
    # __rm_point_columns(normalized_data).to_csv(os.path.join(head_dir, 'normalized_data.csv'), index=False)
    __rm_point_columns(train_train_data).to_csv(os.path.join(head_dir, 'train_30_train.csv'), index=False)
    __rm_point_columns(train_test_data).to_csv(os.path.join(head_dir, 'train_30_test.csv'), index=False)
    __rm_point_columns(valid_10data).to_csv(os.path.join(head_dir, '10_for_check.csv'), index=False)
    __rm_point_columns(valid_30data).to_csv(os.path.join(head_dir, '30_for_predict.csv'), index=False)


if __name__ == '__main__':
    # now_head_dir = "2_rmcoref_data"  # 包含nop数值和 zt数值
    # now_head_dir = "all_data"      # nop 被转换为01，且没有根据相关系数实施特征工程
    now_head_dir = "all_rmcoref_data"  # nop 被转换为01， 根据相关系数删除相关系数大于0.9的项
    file_name = r'simple_dataset.csv'
    now_is_nop_to_01 = True
    run_compounds_split(file_name, head_dir=now_head_dir, is_nop_to_01=now_is_nop_to_01)
