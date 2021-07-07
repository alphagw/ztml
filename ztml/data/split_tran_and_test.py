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

from ztml.tools import get_train_test_index, read_data


def split_to_train2test(data, ratio=0.32, to_data=True):
    column_index = ['Temperature', 'NC_atom_unit']
    train_data, test_data= get_train_test_index(data, column_index, ratio, to_data=to_data)
    return train_data, test_data


def run():
    # origin_csv_file = r'G:\ztml\ztml\data\clean_data.csv'
    data = read_data(r'clean_data_normalized.csv')
    train_data, test_data = split_to_train2test(data)
    train_data.to_csv('train_data_from_normalized_data.csv', index=False)
    test_data.to_csv('test_data_from_normalized_data.csv', index=False)

    
if __name__ == '__main__':
    run()
