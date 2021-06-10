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

from ztml.tools import get_train_test_index
from ztml.data.clean_csv_data import get_clean_data, read_data
from ztml.data.feature_normalize_data import get_normalize_data
from ztml.data.rename_cloumn import get_rename_column_data
from ztml.data.split_tran_and_test import split_to_train2test
import os


def run(fn):
    origin_data = read_data(fn)
    rename_data = get_rename_column_data(origin_data)
    # train_data, test_data = get_train_test_index(rename_data, column_index=['N_atom_unit'], ratio=0.575, to_data=True)
    
    tmp_file = 'temp_train.csv'
    clean_train_data = get_clean_data(rename_data)
    clean_train_data.to_csv(tmp_file, index=False)
    clean_train_data = read_data(tmp_file)
    os.remove(tmp_file)
    normalized_data, _ = get_normalize_data(clean_train_data)
    use2train_data, use2valid_data = get_train_test_index(normalized_data, column_index=['NC_atom_unit'], ratio=0.572, to_data=True)
    
    train_train_data, train_test_data = split_to_train2test(use2train_data, ratio=0.318)
    print(train_train_data.shape, train_test_data.shape)
    dtrain_train_data = train_train_data[:-2]
    dtrain_test_data = train_test_data.append(train_train_data[-2:], ignore_index=True)
    print(dtrain_train_data.shape, dtrain_test_data.shape)
    dtrain_train_data.to_csv('train_30_train.csv', index=False)
    dtrain_test_data.to_csv('train_30_test.csv', index=False)
    use2valid_data.to_csv('valid_40.csv', index=False)
    # tmp_file = 'temp_test.csv'
    # clean_test_data = get_clean_data(test_data)
    # clean_test_data.to_csv(tmp_file, index=False)
    # clean_test_data = read_data(tmp_file)
    # os.remove(tmp_file)
    # test_normalized_data = get_normalize_data(clean_test_data)
    # test_train_data, test_test_data = split_to_train2test(test_normalized_data)
    # test_train_data.to_csv('test_40_train.csv', index=False)
    # test_test_data.to_csv('test_40_test.csv', index=False)
    

if __name__ == '__main__':
    file_name = r'0-20201203_descriptors.csv'
    run(file_name)
