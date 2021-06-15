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


def read_data_for_plt(fn):
    ori_data = pd.read_csv(fn)
    column = ori_data.columns.values.tolist()[:-2]
    data = ori_data.values
    train_data = data[:, :-2]
    dd = np.corrcoef(train_data, rowvar=0)
    print(dd.shape)
    return dd, column


def plt_fig1():
    label_font_size = 12
    ab_index_fontsize = 16
    
    fig = plt.figure(figsize=(18, 8))
    ax = plt.subplot2grid((1, 22), (0, 0), colspan=10, rowspan=1, fig=fig)
    ax2 = plt.subplot2grid((1, 22), (0, 10), colspan=1, rowspan=1,fig=fig)
    ax3 = plt.subplot2grid((1, 22), (0, 12), colspan=10, rowspan=1, fig=fig)


    csv_file = r'G:\ztml\ztml\data\temp_clean_data.csv'
    dd, column = read_data_for_plt(csv_file)
    
    _ = sns.heatmap(dd, vmin=-1, vmax=1, cmap='coolwarm', ax=ax, cbar_ax=ax2)
    ax.set_xticks(np.array(range(0, len(column))))
    ax.set_xlim(0, len(column))
    ax.set_xticks(np.array(range(0, len(column))) + 0.5, minor=True)
    
    ax.set_yticks(np.array(range(0, len(column))))
    ax.set_ylim(0, len(column))
    ax.set_yticks(np.array(range(0, len(column))) + 0.5, minor=True)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticklabels([column[i][:5] if i % 2 == 0 else None for i in range(len(column))], fontsize=label_font_size, minor=True,
                       rotation=85)
    ax.set_yticklabels([column[i][:5] if i % 2 == 0 else None for i in range(len(column))], fontsize=label_font_size,
                       minor=True)  # va='center_baseline',
    # import matplotlib.transforms as mtrans
    # for i in ax.get_xticklabels():
    #     i.set_transform(i.get_transform() + mtrans.Affine2D().translate(5.5, 10))
    
    ax.grid(alpha=0.2, linewidth=0.1, color='gray')
    # ax.set_yticklabels(range(0, 35, 5))
    # ax.set_xticks(range(35))
    # ax.set_yticks(range(35))
    #
    ax.tick_params(axis='x', direction='out', labelrotation=85, length=0.00001)
    ax.tick_params(axis='y', direction='out', labelrotation=0, length=0.00001)
    # ax.tick_params(axis='y', labelrotation=-45)
    ax.text(1, 66, 'A', fontsize=ab_index_fontsize)
    
    csv_file = r'G:\ztml\ztml\data\normalized_data.csv'
    dd, column = read_data_for_plt(csv_file)

    _ = sns.heatmap(dd, vmin=-1, vmax=1, cmap='coolwarm', ax=ax3, cbar=False)
    ax3.set_xticks(np.array(range(len(column))))
    ax3.set_xlim(0, len(column))
    ax3.set_xticks(np.array(range(len(column))) + 0.5, minor=True)

    ax3.set_yticks(np.array(range(len(column))))
    ax3.set_ylim(0, len(column))
    ax3.set_yticks(np.array(range(len(column))) + 0.5, minor=True)

    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    ax3.set_xticklabels([i[:5] for i in column], fontsize=label_font_size, minor=True, rotation=85)
    ax3.set_yticklabels([i[:5] for i in column], fontsize=label_font_size, minor=True)  # va='center_baseline',
    # import matplotlib.transforms as mtrans
    # for i in ax.get_xticklabels():
    #     i.set_transform(i.get_transform() + mtrans.Affine2D().translate(5.5, 10))

    ax3.grid(alpha=0.2, linewidth=0.1, color='gray')
    # ax.set_yticklabels(range(0, 35, 5))
    # ax.set_xticks(range(35))
    # ax.set_yticks(range(35))
    #
    ax3.tick_params(axis='x', direction='out', labelrotation=85, length=0.00001)
    ax3.tick_params(axis='y', direction='out', labelrotation=0, length=0.00001)
    ax3.text(0.5, 28.3, 'B', fontsize=ab_index_fontsize)

    # plt.savefig('plt_coref_fig1.pdf')
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.1, top=0.95, right=0.98, wspace=0.7)
    # plt.show()
    plt.savefig('plt_coref_fig1.pdf')


if __name__ == '__main__':
    plt_fig1()
