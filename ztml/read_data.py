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


def read_data(fn):
    data = pd.read_csv(fn, index_col=0)[:70]
    # print(data.columns)
    # print(data.dtypes)
    # print(data[])
    # print(data.index.name)
    return data

def change_valence(data):
    # fndata = data.copy()
    dd = data["Valence_Con_A"].tolist() + data['Valence_Con_B'].tolist() + data['Valence_Con_C'].tolist()
    data_type_num = list(set(dd))
    val_data_chg = dict(zip(sorted(data_type_num), range(len(data_type_num))))
    # print(val_data_chg)
    # print(data.index.values)
    # exit()
    for i in data.index.tolist():
        for n in ["Valence_Con_A", "Valence_Con_B", "Valence_Con_C"]:
            data[n][i] = val_data_chg[data[n][i]]
            # print(data[n][i])
    print(data[["Valence_Con_A", "Valence_Con_B", "Valence_Con_C"]])
    for i, j in val_data_chg.items():
        # print(i, j)
        # data.get
        # fndata.get_loc(i) = j
        # print(data.loc[i])
        pass
    
    # for i in data.iterrows():
    #     i["Valence_Con_A"] = val_data_chg[i["Valence_Con_A"]]
    
if __name__ == '__main__':
    fn = "data/20201203_descriptors.csv"
    data = read_data(fn)
    change_valence(data)
    # print(data)
