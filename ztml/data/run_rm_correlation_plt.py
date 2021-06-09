#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/05/25 09:01:54'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def plt_features_73():
    csv_file = r'G:\ztml\ztml\data\clean_data.csv'
    ori_data = pd.read_csv(csv_file)
    column = ori_data.columns.values.tolist()[:-2]
    data = ori_data.values
    train_data = data[:, :-2]
    dd = np.corrcoef(train_data, rowvar=0)
    print(dd.shape)
    
    ax = sns.heatmap(dd, vmin=0, vmax=1)
    ax.set_xticks(np.array(range(0, len(column))))
    ax.set_xlim(0, len(column))
    ax.set_xticks(np.array(range(0, len(column))) + 0.5, minor=True)

    ax.set_yticks(np.array(range(0, len(column))))
    ax.set_ylim(0, len(column))
    ax.set_yticks(np.array(range(0, len(column))) + 0.5, minor=True)

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([column[i][:5] if i % 2 == 0 else None for i in range(len(column))], fontsize=8, minor=True, rotation=85)
    ax.set_yticklabels([column[i][:5] if i % 2 == 0 else None for i in range(len(column))], fontsize=8, minor=True)  # va='center_baseline',
    # import matplotlib.transforms as mtrans
    # for i in ax.get_xticklabels():
    #     i.set_transform(i.get_transform() + mtrans.Affine2D().translate(5.5, 10))

    ax.grid(alpha=0.2, linewidth=0.1)
    # ax.set_yticklabels(range(0, 35, 5))
    # ax.set_xticks(range(35))
    # ax.set_yticks(range(35))
    #
    ax.tick_params(axis='x', direction='out', labelrotation=85, length=0.00001)
    ax.tick_params(axis='y', direction='out', labelrotation=0, length=0.00001)
    # ax.tick_params(axis='y', labelrotation=-45)
    # print(help(ax))
    plt.savefig('20210609-origin-final.pdf')


def plt_features_34():
    csv_file = r'G:\ztml\ztml\data\clean_data.csv'
    ori_data = pd.read_csv(csv_file)
    column = ori_data.columns.values.tolist()[:-2]
    data = ori_data.values
    train_data = data[:, :-2]
    a = np.corrcoef(train_data, rowvar=0)
    b = np.triu(a)
    gp = auto_get_corelated_group(data=b, coref_val=0.85, is_triu=True, get_remove=True)
    new_data = train_data.transpose().tolist()
    for m in sorted(gp[1], reverse=True):
        new_data.pop(m)
        column.pop(m)
    
    new_data = np.array(new_data).transpose()
    print(new_data.shape, len(column))
    
    dd = np.corrcoef(new_data, rowvar=0)
    
    ax = sns.heatmap(dd, vmin=0, vmax=1)
    ax.set_xticks(np.array(range(len(column))))
    ax.set_xlim(0, len(column))
    ax.set_xticks(np.array(range(len(column))) + 0.5, minor=True)
    
    ax.set_yticks(np.array(range(len(column))))
    ax.set_ylim(0, len(column))
    ax.set_yticks(np.array(range(len(column))) + 0.5, minor=True)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([i[:5] for i in column], fontsize=8, minor=True, rotation=85)
    ax.set_yticklabels([i[:5] for i in column], fontsize=8, minor=True) # va='center_baseline',
    # import matplotlib.transforms as mtrans
    # for i in ax.get_xticklabels():
    #     i.set_transform(i.get_transform() + mtrans.Affine2D().translate(5.5, 10))
    
    ax.grid(alpha=0.2, linewidth=0.1)
    # ax.set_yticklabels(range(0, 35, 5))
    # ax.set_xticks(range(35))
    # ax.set_yticks(range(35))
    #
    ax.tick_params(axis='x', direction='out', labelrotation=85, length=0.00001)
    ax.tick_params(axis='y', direction='out', labelrotation=0, length=0.00001)
    # ax.tick_params(axis='y', labelrotation=-45)
    # print(help(ax))
    plt.savefig('20210609-final.pdf')
    # plt.show()


if __name__ == '__main__':
    plt_features_73()
