#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/10 16:57:00'

from ztml.tools import read_data


def change_valence(data):
    dd = data["Valence_Con_A"].tolist() + data['Valence_Con_B'].tolist() + data['Valence_Con_C'].tolist()
    data_type_num = list(set(dd))
    val_data_chg = dict(zip(sorted(data_type_num), range(len(data_type_num))))
    
    for i in data.index.tolist():
        for n in ["Valence_Con_A", "Valence_Con_B", "Valence_Con_C"]:
            data.loc[i, n] = val_data_chg[data[n][i]]
    
    return data


def get_new_columns_label(data, is_write=False, fn="all_change_column.log"):
    aa = data.columns.get_values()
    re_name = ['_'.join(i.replace('(', ' ').replace('*', ' ').replace('.', ' ').replace(')', ' ').replace('-', ' ')
                        .replace('/', ' ').split()) for i in aa]
    
    re_name[re_name.index("K_cal_100K")] = "100_K"
    re_name[re_name.index("N_optimal_1e20_cm_3__100K")] = "100_K_1"
    re_name[re_name.index("ZT_max_100K")] = "100_K_2"
    
    for i in re_name:
        if i.endswith('_K'):
            re_name[re_name.index(i)] = i.replace('_K', '_K_0')
    
    if is_write:
        with open(fn, 'w', encoding='utf-8') as f:
            for i in range(len(aa)):
                f.write('%s         %s\n' % (aa[i], re_name[i]))
    
    return dict(zip(aa, re_name))


def change_columns(data, new_label):
    data = data.rename(columns=new_label)
    return data


def get_rename_column_data(data):
    # 修改价电子，str --> int
    # {'3S2P2': 0, '3S2P4': 1, '4S2P2': 2, '4S2P3': 3, '4S2P4': 4, '5S2P2': 5, '5S2P3': 6, '5S2P4': 7, '6S2P2': 8,
    #  '6S2P3': 9}
    valed_data = change_valence(data)
    
    # 获取每一列旧名称对应的新名称
    new_label = get_new_columns_label(valed_data, is_write=False, fn=r"data\\all_change_column.log")
    # 实际改变每一列的新名称
    new_data = change_columns(valed_data, new_label)
    return new_data


def run(fn):
    data = read_data(fn)
    new_data = get_rename_column_data(data)
    new_data.to_csv(r'.\\1-70-rename_column.csv')


if __name__ == '__main__':
    run(fn='0-20201203_descriptors.csv')
