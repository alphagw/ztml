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
import matplotlib.pyplot as plt
import os
import pandas as pd
from ztml.plt.plt_train_fig2 import read_cal_predit


def gather_data(fn):
    pd1 = read_cal_predit(fn)
    t = pd.read_csv(r'..\data\10_for_check.csv')['Temperature'].values
    return pd1, t
    
aa = lambda x: 0.05 if x < 0.5 else 0.95


def plt_predict_cal(fn, fn2, ntype1, ntype2):
    label_font = {"fontsize": 16}
    tick_font_size = 14
    index_label_font = {"fontsize": 18, 'weight': 'bold'}
    # lv #3CAF6F
    plt.figure(figsize=(16, 8))
    npd1 = read_cal_predit(ntype_f1)
    ax0 = plt.subplot2grid((12, 2), (0, 0), colspan=1, rowspan=2)
    for i in range(npd1.shape[0]):
        ax0.scatter(i, npd1[i][0], edgecolors='white',color='#3CAF6F', alpha=0.8, linewidths=0.2, s=90)
        ax0.scatter(i, aa(npd1[i][1]), edgecolors='white',color='darkorange',alpha=0.8, linewidths=0.2, s=90)
    ax0.set_xlim(-2, 122)
    ax0.set_ylim(-0.2, 1.2)
    colors = {'Calculated': '#3CAF6F', 'Predicted': 'darkorange'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    ax0.legend(handles, labels, ncol=2)
    ax0.text(0, 1.3, "A", fontdict=index_label_font)
    # ax0.set_yticklabels({0: 'Ntype', 1:'Ptype'})
    ax0.set_yticks([0, 1])
    ax0.set_yticklabels(['N-type', 'P-type'])
    ax0.tick_params(labelsize=tick_font_size)
    ax0.set_xlabel("The number of compound", fontdict=label_font)
    
    ax = plt.subplot2grid((12, 2), (3, 0), colspan=1, rowspan=9)
    # fig, axes = plt.subplots(3, 2, figsize=(16, 8))
    # ax = axes[0]
    pd1, t = gather_data(fn)
    # cmp = mpl.colors.ListedColormap('magma', N=12)
    # print(cmp)
    # print(pd1.shape, t.shape)
    # print({str(i): cmp[i] for i in list(set(t.tolist()))})
    # exit()
    for m in range(pd1.shape[0]):
        ax.scatter(pd1[m, 0], pd1[m, 1], edgecolors='white', color='#347FE2', linewidths=0.2, alpha=t[m], s=90)
    # ax.scatter(pd1[:, 0], pd1[:, 1], c=t, norm=mpl.colors.BoundaryNorm(t, cmp.N), cmap=cmp)
    slice_set = 0.0, 1.
    _tmp_xy = np.linspace(slice_set, pd1.shape[0])
    ax.plot(_tmp_xy, _tmp_xy, '#F37878', linewidth=3, alpha=0.8)
    ax.set_xlim(slice_set)
    ax.set_ylim(slice_set)
    ax.set_xlabel("Calculated", fontdict=label_font)
    ax.set_ylabel("Predicted", fontdict=label_font)
    ax.text(0.01, 1.01, "B", fontdict=index_label_font)
    # ax.text(-0.175, 1.2, pindex[i], fontsize=12)
    plt.subplots_adjust(hspace=0.3)

    b_x_slice_set = -1, 92
    b_y_slice_set = 0, 0.9
    width = 0.6
    pd2 = read_cal_predit(fn2)
    npd2 = read_cal_predit(ntype_f2)
    
    def get_color(data):
        return ['#F37878' if data[i] < 0.5 else '#347FE2' for i in range(data.shape[0])]
        
    ax2 = plt.subplot2grid((12, 2), (0, 1), colspan=1, rowspan=3)
    ax2.bar(range(1, 91), pd2[:90, 1], width=width, color=get_color(npd2[:90, 1]))
    ax2.set_xlim(b_x_slice_set)
    ax2.set_ylim(0, 0.99)
    ax2.text(0, 1.03, 'C', fontdict=index_label_font)

    colors = {'N-type': '#F37878', 'P-type': '#347FE2'}
    labels = list(colors.keys())
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
    ax2.legend(handles, labels, ncol=2, loc='upper right')
    
    ax3 = plt.subplot2grid((12, 2), (3, 1), colspan=1, rowspan=3)
    ax3.bar(range(1, 91), pd2[90:180, 1], width=width, color=get_color(npd2[90:180, 1]))
    ax3.set_xlim(b_x_slice_set)
    ax3.set_ylim(b_y_slice_set)
    ax3.set_ylabel("The value of ZT by Machine learning", fontdict=label_font)
    ax4 = plt.subplot2grid((12, 2), (6, 1), colspan=1, rowspan=3)
    ax4.bar(range(1, 91), pd2[180:270, 1], width=width, color=get_color(npd2[180:270, 1]))
    ax4.set_xlim(b_x_slice_set)
    ax4.set_ylim(b_y_slice_set)
    ax5 = plt.subplot2grid((12, 2), (9, 1), colspan=1, rowspan=3)
    ax5.bar(range(1, 91), pd2[270:360, 1], width=width, color=get_color(npd2[270:360, 1]))
    ax5.set_xlim(b_x_slice_set)
    ax5.set_ylim(b_y_slice_set)
    ax5.set_xlabel("Number of compounds", fontdict=label_font)
    ax5.set_xticks([0, 18, 36, 54, 72, 90])
    # ax5.set_xticklabels(fontsize=10)
    for i in [ax, ax2, ax3, ax4, ax5]:
        i.tick_params(axis='both', labelsize=tick_font_size)
    plt.tight_layout()
    plt.subplots_adjust(left=0.05,bottom=0.08, right=0.98, top=0.95, hspace=0.0, wspace=0.2)
    plt.savefig('plt_valid_fig3.pdf')
    # plt.show()



if __name__ == '__main__':
    save_dir = r'..\train\training_module'
    fn1 = r'10_for_check.csv'
    fn2 = r'30_for_predict.csv'
    data_file1 = os.path.join(save_dir, 'z_result_valid_%s.out' % fn1)
    data_file2 = os.path.join(save_dir, 'z_result_valid_%s.out' % fn2)
    ntype_f1 = os.path.join(save_dir, 'ntype_z_result_valid_%s.out' % fn1)
    ntype_f2 = os.path.join(save_dir, 'ntype_z_result_valid_%s.out' % fn2)
    plt_predict_cal(data_file1, data_file2, ntype_f1, ntype_f2)
