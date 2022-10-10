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
import os


def read_data_for_plt(fn, has_first_column=True):
    if has_first_column:
        star_num = 0
    else:
        star_num = 1
    ori_data = pd.read_csv(fn)
    column = ori_data.columns.values.tolist()[star_num:-2]
    data = ori_data.values
    train_data = data[:, star_num:-2]
    dd = np.corrcoef(train_data, rowvar=0)
    print(dd.shape)
    return dd, column


def plt_fig1():
    label_font = {"fontsize": 14, 'family': 'Times New Roman'}
    index_label_font = {"fontsize": 18, 'weight': 'light', 'family': 'Times New Roman'}
    tick_font_size = 14
    tick_font_dict = {"fontsize": 14, 'family': 'Times New Roman'}
    fig = plt.figure(figsize=(9, 4))
    plt.rc('font', family='Times New Roman', weight='normal')
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["ytick.direction"] = 'in'
    ax = plt.subplot2grid((1, 23), (0, 0), colspan=10, rowspan=1, fig=fig)
    ax2 = plt.subplot2grid((1, 23), (0, 10), colspan=1, rowspan=1,fig=fig)
    ax2.tick_params(axis='both', labelsize=tick_font_size-2)
    ax3 = plt.subplot2grid((1, 23), (0, 13), colspan=10, rowspan=1, fig=fig)
    head_dir = r"G:\ztml\ztml\rdata\all_rmcoref_data"

    csv_file = os.path.join(head_dir, r'temp_clean_data.csv')
    dd, columns = read_data_for_plt(csv_file, has_first_column=False)
    
    column = []
    for nn in columns:
        if str(nn).startswith('K') or str(nn).startswith('k'):
            _label = r'$\kappa_{%s}$' % nn[1:]
        elif str(nn).startswith('r') or str(nn).startswith('R') or str(nn).startswith('n') or str(nn).startswith('m'):
            _label = r'%s$_{%s}$' % (str(nn[0]).lower(), nn[1:])
        elif str(nn) == 'a_b':
            _label = 'a/b'
        elif str(nn) == 'NC.1':
            nn = "NCo"
            _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        else:
            _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        column.append(_label)

    _ = sns.heatmap(dd, vmin=-1, vmax=1, cmap='coolwarm', ax=ax, cbar_ax=ax2,
                    cbar_kws={"ticks": np.arange(1, -1.2, -0.2)})
    ax.set_xticks(np.array(range(0, len(column))))
    ax.set_xlim(0, len(column))
    ax.set_xticks(np.array(range(0, len(column))) + 0.5, minor=True)
    
    ax.set_yticks(np.array(range(0, len(column))))
    ax.set_ylim(0, len(column))
    ax.set_yticks(np.array(range(0, len(column))) + 0.5, minor=True)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([column[i] if i % 2 == 0 else None for i in range(len(column))],
                       fontdict=tick_font_dict, minor=True,
                       rotation=85)
    ax.set_yticklabels([column[i] if i % 2 == 0 else None for i in range(len(column))], fontdict=tick_font_dict,
                       minor=True)  # va='center_baseline',
    # import matplotlib.transforms as mtrans
    # for i in ax.get_xticklabels():
    #     i.set_transform(i.get_transform() + mtrans.Affine2D().translate(5.5, 10))
    
    ax.grid(alpha=0.7, linewidth=0.1, color='gray')
    # ax.set_yticklabels(range(0, 35, 5))
    # ax.set_xticks(range(35))
    # ax.set_yticks(range(35))
    #
    ax.tick_params(axis='x', direction='in', labelrotation=85, length=0.00001)
    ax.tick_params(axis='y', direction='in', labelrotation=0, length=0.00001)
    # ax.tick_params(axis='y', labelrotation=-45)
    ax.text(0.1, 31.7, '(a)', fontdict=index_label_font)
    
    csv_file = os.path.join(head_dir, r'normalized_data.csv')
    dd, columns = read_data_for_plt(csv_file, has_first_column=True)
    column = []
    for nn in columns:
        if str(nn).startswith('K') or str(nn).startswith('k'):
            _label = r'$\kappa_{%s}$' % nn[1:]
        elif str(nn).startswith('r') or str(nn).startswith('R') or str(nn).startswith('n') or str(nn).startswith('m'):
            _label = r'%s$_{%s}$' % (str(nn[0]).lower(), nn[1:])
        elif str(nn) == 'a_b':
            _label = 'a/b'
        elif str(nn) == 'NC.1':
            nn = "NCo"
            _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        elif len(str(nn)) == 1:
            _label = str(nn)
        else:
            _label = r'%s$_{%s}$' % (str(nn[0]).upper(), nn[1:])
        column.append(_label)

    _ = sns.heatmap(dd, vmin=-1, vmax=1, cmap='coolwarm', ax=ax3, cbar=False)
    ax3.set_xticks(np.array(range(len(column))))
    ax3.set_xlim(0, len(column))
    ax3.set_xticks(np.array(range(len(column))) + 0.5, minor=True)

    ax3.set_yticks(np.array(range(len(column))))
    ax3.set_ylim(0, len(column))
    ax3.set_yticks(np.array(range(len(column))) + 0.5, minor=True)

    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([i for i in column], fontdict=tick_font_dict, minor=True, rotation=85)
    ax3.set_yticklabels([i for i in column], fontdict=tick_font_dict, minor=True)  # va='center_baseline',
    # import matplotlib.transforms as mtrans
    # for i in ax.get_xticklabels():
    #     i.set_transform(i.get_transform() + mtrans.Affine2D().translate(5.5, 10))

    ax3.grid(alpha=0.7, linewidth=0.5, color='white')
    # ax.set_yticklabels(range(0, 35, 5))
    # ax.set_xticks(range(35))
    # ax.set_yticks(range(35))
    #
    ax3.tick_params(axis='x', direction='in', labelrotation=85, length=0.00001)
    ax3.tick_params(axis='y', direction='in', labelrotation=0, length=0.00001)
    ax3.text(0.1, 11.3, '(b)', fontdict=index_label_font)

    # plt.savefig('plt_coref_fig1.pdf')
    # plt.tight_layout()
    plt.subplots_adjust(left=0.06, bottom=0.12, top=0.93, right=0.98, wspace=1.2)
    plt.show()
    # plt.savefig('plt_coref_fig1.pdf', dpi=600)
    # plt.savefig('plt_coref_fig1.jpg', dpi=600)
    # plt.savefig('plt_coref_fig1.tiff', dpi=600)


if __name__ == '__main__':
    plt_fig1()
