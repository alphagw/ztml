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
    # return pd.DataFrame(pd1, columns=['Cal', 'Pre', 'T'], index=None)
    return pd1
    
aa = lambda x: 0.05 if x < 0.5 else 0.95


def plt_predict_cal(fn, fn2, ntype1, ntype2):
    
    label_font = {"fontsize": 16}
    tick_font_size = 14
    index_label_font = {"fontsize": 18, 'weight': 'bold'}
    # lv #3CAF6F
    plt.figure(figsize=(16, 8))
    npd1 = read_cal_predit(ntype1)
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
    pd1 = gather_data(fn)
    for i in range(pd1.shape[0]):
        pd1[i][2] = int(pd1[i][2] * 550 + 100)
    c12 = {100: '#e8f1f8', 150: '#deebf7', 200: '#c6dbef', 250: '#a0c8e4', 300: '#9ecae1', 350: '#82c4ed',
           400: '#6baed6', 450: '#5499c0', 500: '#4292c6', 550: '#2171b5', 600: '#08519c', 650: '#08306b'}

    # tc = ['#f7fbff', '#deebf7', '#c6dbef', '#a0c8e4', '#9ecae1', '#82c4ed', '#6baed6', '#5499c0', '#4292c6', '#2171b5', '#08519c', '#08306b']
    # a = {i*50+100: tc[i] for i in range(len(tc))}
    # print(a)
    # exit()
    for m in range(pd1.shape[0]):
        ax.scatter(pd1[m, 0], pd1[m, 1], edgecolors='white', color=c12[pd1[m, 2]], linewidths=0.2, s=90)
        # ax.scatter(pd1[m, 0], pd1[m, 1], edgecolors='white', color='#347FE2', linewidths=0.2,
        #            alpha=(pd1[m, 2] - 100) / 550, s=90)
    # ax.scatter(pd1[:, 0], pd1[:, 1], c=t, norm=mpl.colors.BoundaryNorm(t, cmp.N), cmap=cmp)
    slice_set = 0.0, 1.
    _tmp_xy = np.linspace(slice_set, pd1.shape[0])
    ax.plot(_tmp_xy, _tmp_xy, '#F37878', linewidth=3, alpha=0.8)
    ax.set_xlim(slice_set)
    ax.set_ylim(slice_set)
    ax.set_xlabel("Calculated", fontdict=label_font)
    ax.set_ylabel("Predicted", fontdict=label_font)
    ax.text(0.01, 1.01, "B", fontdict=index_label_font)
    labels = [str(i)+'K' for i in c12.keys()]
    handles = [plt.Circle((0, 0), 1, color=c12[int(label[:-1])]) for label in labels]
    ax.legend(handles, labels, ncol=3, loc='lower right')
    # ax.text(-0.175, 1.2, pindex[i], fontsize=12)
    plt.subplots_adjust(hspace=0.3)

    # b_x_slice_set = -1, 92
    b_y_slice_set = 0, 0.9
    width = 0.6
    pd2 = read_cal_predit(fn2)
    npd2 = read_cal_predit(ntype2)
    __dd = np.hstack((pd2, npd2[:, 0].reshape(-1, 1), npd2[:, 1].reshape(-1, 1)))
    
    new_data = pd.DataFrame(__dd, columns=['CZT', 'PZT', 'T', 'N3', "V1", "CN", "PN"], index=None)
    split_data = new_data.groupby(['N3'])
    dd = []
    txt_label = ['GST124', 'GST147', 'GST225', 'GST326']
    for i, j in split_data.groups.items():
        dd.append(new_data.loc[j])
    # print(split_data)
    # dd = pd.concat(dd)
    # dd = pd.DataFrame(dd.values, index=None, columns=dd.columns.values)
    
    def get_color(data):
        return ['#F37878' if data[i] < 0.5 else '#347FE2' for i in range(data.shape[0])]
    
    from copy import deepcopy
    for i in range(len(txt_label)):
        ax2 = plt.subplot2grid((12, 2), (3*i, 1), colspan=1, rowspan=3)
        _tsd = deepcopy(dd[i])
        data = dd[i].groupby(["V1"])
        __new_ddd = []
        for j, k in data.groups.items():
            _tmp_dd = _tsd.loc[k]
            _tmp_dd.sort_values("T", inplace=True)
            __new_ddd.append(_tmp_dd)
            # print(j, dd[i].iloc[k])
        fdd = pd.concat(__new_ddd)
        fdd = pd.DataFrame(fdd.values, index=None, columns=fdd.columns.values)
        zt_val = fdd[:]['PZT'].values
        ntype_val = fdd[:]['PN'].values
        _colors = get_color(ntype_val)
        t = fdd[:]['T'].values / 2 + 0.5
        for xx in range(fdd.shape[0]):
            ax2.bar(xx, zt_val[xx], width=width, color=_colors[xx], alpha=t[xx])
        
        ax2.set_xlim(-1, len(fdd))
        ax2.set_ylim(0, 0.99)
        
        ax2.tick_params(axis='both', labelsize=tick_font_size)
        if i != 2:
            ax2.text(18, 0.85, txt_label[i], fontdict=label_font)
        else:
            ax2.text(12, 0.85, txt_label[i], fontdict=label_font)

        if i == 0:
            ax2.text(0, 1.03, 'C', fontdict=index_label_font)
            colors = {'N-type': '#F37878', 'P-type': '#347FE2'}
            labels = list(colors.keys())
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
            ax2.legend(handles, labels, ncol=2, loc='upper right')

        if i == 2:
            ax2.set_ylabel("                       The value of ZT predicted "
                           "by Machine learning", fontdict=label_font)
        
        if i == 3:
            ax2.set_xlabel("Index of compounds", fontdict=label_font)
            ax2.set_xticks([0, 18, 36, 54, 72, 90])
            
    plt.tight_layout()
    plt.subplots_adjust(left=0.05,bottom=0.08, right=0.98, top=0.95, hspace=0.0, wspace=0.2)
    plt.savefig('plt_valid_fig3.pdf', dpi=600)
    # plt.show()



if __name__ == '__main__':
    save_dir = r'..\train\training_module'
    fn1 = r'10_for_check.csv'
    fn2 = r'30_for_predict.csv'
    data_file1 = os.path.join(save_dir, 'z_result_valid_has_t_%s.out' % fn1)
    data_file2 = os.path.join(save_dir, 'z_result_valid_has_t_%s.out' % fn2)
    ntype_f1 = os.path.join(save_dir, 'ntype_z_result_valid_has_t_%s.out' % fn1)
    ntype_f2 = os.path.join(save_dir, 'ntype_z_result_valid_has_t_%s.out' % fn2)
    plt_predict_cal(data_file1, data_file2, ntype_f1, ntype_f2)
