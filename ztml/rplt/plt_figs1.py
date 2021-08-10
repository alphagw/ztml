#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/24 16:32:53'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ztml.tools import method2
from ztml.data.clean_csv_data import read_data


def get_correc_column_data():
    tmp_file = r'..\data\temp_clean_data.csv'
    data = read_data(tmp_file)

    column = data.columns.values.tolist()[:-2]
    train_data = data.values[:, :-2]
    a = np.corrcoef(train_data, rowvar=0)
    b = np.abs(np.triu(a))
    
    double_index = np.where(b > 0.85)
    index_final = np.array([[double_index[0][m], double_index[1][m]]
                            for m in range(len(double_index[0]))
                            if double_index[0][m] != double_index[1][m]])
    
    gp = method2(index_final)
    
    fdata, columns = [], []
    for i in gp:
        tdata = train_data[:, i]
        cr = np.corrcoef(tdata, rowvar=0)
        columns.append([column[m] for m in i])
        fdata.append(cr)
        
    return fdata, columns


def plt_cor(data, columns):
    label_font = {"fontsize": 12, 'family': 'Times New Roman'}
    legend_font = {"fontsize": 12, 'family': 'Times New Roman'}
    tick_font_size = 12
    index_label_font = {"fontsize": 20, 'weight': 'bold', 'family': 'Times New Roman'}
    plt.rc('font', family='Times New Roman', weight='normal')
    nrow = 9
    ncol = 9
    index_label = "ABCDEFGH"
    
    index_posi = [(-0.02, 3.1), (-0.09, 7.2), (-0.09, 6.1),
                  (-0.05, 8.2), (-0.05, 9.2), (-0.05, 5.1),
                  (-0.02, 4.1), (-0.02, 3.1)]
    
    plt.figure(figsize=(13, 13))
    # axes = axes.flatten()
    cbarax = plt.subplot2grid((nrow, ncol), (6, 6), colspan=1, rowspan=3)

    for i in range(len(data)):
        # ax = axes[i]
        ax = plt.subplot2grid((nrow, ncol), (int(i/3)*3, int(i%3)*3), colspan=3, rowspan=3)
        
        column = []
        for nn in columns[i]:
            if str(nn).startswith('K') or str(nn).startswith('k'):
                _label = r'$\kappa_{%s}$' % nn[1:]
            elif str(nn).startswith('r') or str(nn).startswith('R') or str(nn).startswith('n') or str(nn).startswith(
                    'm'):
                _label = r'%s$_{%s}$' % (str(nn[0]).lower(), nn[1:])
            elif str(nn) == 'a/b':
                _label = nn
            elif str(nn) == 'NC.1':
                nn = "NCo"
                _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
            else:
                _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
            column.append(_label)
            
        _ = sns.heatmap(data[i], vmin=-1, vmax=1, cmap='coolwarm', ax=ax, cbar=False)
        
        if i == 7:
            _ = sns.heatmap(data[i], vmin=-1, vmax=1, cmap='coolwarm', ax=ax, cbar_ax=cbarax,
                            cbar_kws={"ticks": np.array([1, 0, -1])})
        
        ax.set_xticks(np.array(range(0, len(column))))
        ax.set_xlim(0, len(column))
        ax.set_xticks(np.array(range(0, len(column))) + 0.5, minor=True)

        ax.set_yticks(np.array(range(0, len(column))))
        ax.set_ylim(0, len(column))
        ax.set_yticks(np.array(range(0, len(column))) + 0.5, minor=True)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticklabels([column[i] for i in range(len(column))],
                           fontdict=label_font,
                           minor=True,
                           rotation=85)
        ax.set_yticklabels([column[i] for i in range(len(column))],
                           fontdict=label_font,
                           minor=True)  # va='center_baseline',

        ax.grid(alpha=0.5, linewidth=0.5, color='white')
        
        ax.tick_params(axis='x', direction='out', labelrotation=85, length=0.00001)
        ax.tick_params(axis='y', direction='out', labelrotation=0, length=0.00001)
        pp = index_posi[i]
        ax.text(pp[0], pp[1], index_label[i], fontdict=index_label_font)

    plt.subplots_adjust(left=0.06, bottom=0.06, right=0.98, top=0.97, wspace=1, hspace=1)
    # plt.show()
    plt.savefig('plt_coref_FigS1.tiff')


if __name__ == '__main__':
    data, columns = get_correc_column_data()
    plt_cor(data=data, columns=columns)
