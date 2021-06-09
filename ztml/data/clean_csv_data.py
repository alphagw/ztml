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


def read_data(fn):
    data = pd.read_csv(fn, index_col=0)[:70]
    # data = pd.read_csv(fn)[:70]
    return data

 
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


def add_tempeture(data):
    all_label = data.columns.get_values()
    tempeture = [str(i) for i in range(100, 700, 50)]
    t_index = ['_K_0', '_K_1', '_K_2']
    new_ii = ["Kpa", "N_optimal", "ZT"]
    share_label = [tl for tl in all_label if tl not in [tm+ti for tm in tempeture for ti in t_index]]
    share_label.remove('ABC_Compound')
    
    copydata = deepcopy(data[share_label])

    num = 0
    fn_data = None
    for tm in tempeture:
        tmp_data = deepcopy(copydata)
        tmp_data.insert(tmp_data.shape[1], 'Temperature', tm)
    
        for ti in range(len(t_index)):
            tmp_data.insert(tmp_data.shape[1], new_ii[ti], data[tm+t_index[ti]])
        
        num += 1
        
        if num == 1:
            fn_data = tmp_data
        else:
            fn_data = fn_data.append(tmp_data)
        
    return pd.DataFrame(fn_data.get_values(), columns=fn_data.columns.get_values())


def change_np(data):
    _tt = {"p": 0, "N": 1, "P": 0, "n": 1}

    for i in data.index.tolist():
        data.loc[i, 'Type'] = _tt[data['Type'][i].strip()]
        
    return data

    
def write_data_to_csv(data, fn='111'):
    data.to_csv(fn)
    

def run(fn):
    data = read_data(fn)
    
    # 修改价电子，str --> int
    # {'3S2P2': 0, '3S2P4': 1, '4S2P2': 2, '4S2P3': 3, '4S2P4': 4, '5S2P2': 5, '5S2P3': 6, '5S2P4': 7, '6S2P2': 8,
    #  '6S2P3': 9}
    valed_data = change_valence(data)
    
    # 获取每一列旧名称对应的新名称
    new_label = get_new_columns_label(valed_data, is_write=False, fn=r"data\\all_change_column.log")
    # 实际改变每一列的新名称
    new_data = change_columns(valed_data, new_label)
    
    # 添加12个温度参数，并将所有温度对应的Kpa，N，ZT三个值添加上, 将70组数据扩展为840组数据
    add_t_data = add_tempeture(new_data)
    
    # 将P类型还是N类型半导体改成数值类型 P型：0  N型：1
    fdata = change_np(add_t_data)
    write_data_to_csv(fdata, fn=r'data\\clean_data.csv')
    

if __name__ == '__main__':
    tfn = "data/20201203_descriptors.csv"
    run(tfn)
