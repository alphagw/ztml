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

from ztml.tools import get_train_test_index, get_random_groupby_index
from ztml.data.clean_csv_data import get_clean_data, read_data
from ztml.data.feature_normalize_data import get_normalize_data
from ztml.data.rename_cloumn import get_rename_column_data
from ztml.data.split_tran_and_test import split_to_train2test


def run(fn):
    # 重命名每一列
    origin_data = read_data(fn)
    rename_data = get_rename_column_data(origin_data)
    # train_data, test_data = get_train_test_index(rename_data, column_index=['N_atom_unit'], ratio=0.575, to_data=True)
    
    # 添加12个温度参数，并将所有温度对应的Kpa，N，ZT三个值添加上, 将70组数据扩展为840组数据
    clean_train_data = get_clean_data(rename_data)
    tmp_file = 'temp_clean_data.csv'
    clean_train_data.to_csv(tmp_file, index=False)
    clean_train_data = read_data(tmp_file)
    # os.remove(tmp_file)
    
    # 获取归一化之后数据，并删除相关列
    normalized_data, _ = get_normalize_data(clean_train_data)
    normalized_data.to_csv('normalized_data.csv', index=False)
    
    # 根据总原子（化合物）将数据分为30个训练和40个验证集
    use2train_data, use2valid_data = get_train_test_index(normalized_data, column_index=['NC_atom_unit'], ratio=0.572, to_data=True)
    
    # 将训练及在次分为训练集和测试集
    train_train_data, train_test_data = split_to_train2test(use2train_data, ratio=0.318)
    print(train_train_data.shape, train_test_data.shape)
    # 微调训练集和测试集数据大小
    _rm_nlarge = -1
    dtrain_train_data = train_train_data[:_rm_nlarge]
    dtrain_test_data = train_test_data.append(train_train_data[_rm_nlarge:], ignore_index=True)
    print(dtrain_train_data.shape, dtrain_test_data.shape)
    
    # 输出数据
    dtrain_train_data.to_csv('train_30_train.csv', index=False)
    dtrain_test_data.to_csv('train_30_test.csv', index=False)
    use2valid_data.to_csv('valid_40.csv', index=False)


def run_compounds_split(fn):
    # 重命名每一列
    origin_data = read_data(fn)
    rename_data = get_rename_column_data(origin_data)
    # train_data, test_data = get_train_test_index(rename_data, column_index=['N_atom_unit'], ratio=0.575, to_data=True)

    # 添加12个温度参数，并将所有温度对应的Kpa，N，ZT三个值添加上, 将70组数据扩展为840组数据
    clean_train_data = get_clean_data(rename_data)
    tmp_file = 'temp_clean_data.csv'
    clean_train_data.to_csv(tmp_file, index=False)
    clean_train_data = read_data(tmp_file)
    # os.remove(tmp_file)

    # 获取归一化之后数据，并删除相关列
    normalized_data, _ = get_normalize_data(clean_train_data)
    normalized_data.to_csv('normalized_data.csv', index=False)
    
    # normalized_data = read_data('normalized_data.csv')
    # 根据总原子（化合物）将数据分为30个训练和40个验证集
    use2train_data, use2valid_data = get_random_groupby_index(normalized_data, column_index=['B_Gpa'],
                                                              ratio=4/7, to_data=True)
    
    # 将训练及在次分为训练集和测试集
    train_train_data, train_test_data = get_random_groupby_index(use2train_data, column_index=['B_Gpa'],
                                                                 ratio=0.3, to_data=True)
    print(train_train_data.shape, train_test_data.shape)
    
    valid_10data, valid_30data = get_random_groupby_index(use2valid_data, column_index=['B_Gpa'],
                                                          ratio=0.75, to_data=True)
    print(valid_10data.shape, valid_30data.shape)
    # 输出数据
    train_train_data.to_csv('train_30_train.csv', index=False)
    train_test_data.to_csv('train_30_test.csv', index=False)
    valid_10data.to_csv('10_for_check.csv', index=False)
    valid_30data.to_csv('30_for_predict.csv', index=False)


if __name__ == '__main__':
    file_name = r'0-20201203_descriptors.csv'
    # run(file_name)
    run_compounds_split(file_name)
