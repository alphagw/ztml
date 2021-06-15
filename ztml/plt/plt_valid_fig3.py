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
from ztml.tools import get_train_test_index
from ztml.train.train import ttest
from ztml.plt.plt_train_fig2 import read_cal_predit


def deal_data(fn, fn1, fn2):
    data = pd.read_csv(fn)
    vt_data, vtest_data = get_train_test_index(data, column_index=['NC_atom_unit'], ratio=0.749, to_data=True)
    data = vt_data[:-1]
    vtest_data = vtest_data.append(vt_data.iloc[-1])
    data.to_csv(fn1, index=False)
    vtest_data.to_csv(fn2, index=False)


def use_ml_to_predict(fname):
    
    save_dir = r'..\train\training_module'
    nfeature = 27
    hidden_layer = [500, 100, 50, 20]  # [100, 50, 20]  [100, 100, 50, 20]
    label = '4layer_500' # '3layer_100_Elu', '3layer_100_PRelu', '3layer_100_sigmod', '3layer_100_Tanh', '3layer_100', '4layer_100', '4layer_500'

    ttest(test_csv_fn=os.path.join(r'.', fname),
          mp_fn=os.path.join(save_dir, 'dnn_params_5000_%s.pkl' % label),
          output_fn='z_result_valid_%s.out' % fname,
          save_dir=save_dir, n_feature=nfeature, HIDDEN_NODES=hidden_layer,
          batch_size=500)


def gather_data(fn):
    pd1 = read_cal_predit(fn)
    t = pd.read_csv(r'..\data\10_for_check.csv')['Temperature'].values
    return pd1, t
    
    
def plt_predict_cal(fn, fn2):
    # lv #3CAF6F
    top_index_size = 16
    plt.figure(figsize=(16, 8))
    ax = plt.subplot2grid((3, 2), (0, 0), colspan=1, rowspan=3)
    # fig, axes = plt.subplots(3, 2, figsize=(16, 8))
    # ax = axes[0]
    pd1, t = gather_data(fn)
    # cmp = mpl.colors.ListedColormap('magma', N=12)
    # print(cmp)
    # print(pd1.shape, t.shape)
    # print({str(i): cmp[i] for i in list(set(t.tolist()))})
    # exit()
    for m in range(pd1.shape[0]):
        ax.scatter(pd1[m, 0], pd1[m, 1], edgecolors='white', color='#347FE2', linewidths=0.2, alpha=t[m])
    # ax.scatter(pd1[:, 0], pd1[:, 1], c=t, norm=mpl.colors.BoundaryNorm(t, cmp.N), cmap=cmp)
    slice_set = 0.0, 1.
    _tmp_xy = np.linspace(slice_set, pd1.shape[0])
    ax.plot(_tmp_xy, _tmp_xy, '#F37878', linewidth=3, alpha=0.8)
    ax.set_xlim(slice_set)
    ax.set_ylim(slice_set)
    ax.set_xlabel("Calculated")
    ax.set_ylabel("Predicted")
    ax.text(0.01, 1.01, "A", fontsize=top_index_size)
    # ax.text(-0.175, 1.2, pindex[i], fontsize=12)
    
    b_x_slice_set = -1, 122
    b_y_slice_set = 0, 1.16
    pd2 = read_cal_predit(fn2)
    ax2 = plt.subplot2grid((3, 2), (0, 1))
    ax2.bar(range(1, 121), pd2[:120, 1])
    ax2.set_xlim(b_x_slice_set)
    ax2.set_ylim(b_y_slice_set)
    ax2.text(0, 1.19, 'B', fontsize=top_index_size)
    
    ax3 = plt.subplot2grid((3, 2), (1, 1))
    ax3.bar(range(1, 121), pd2[120:240, 1])
    ax3.set_xlim(b_x_slice_set)
    ax3.set_ylim(b_y_slice_set)
    ax3.set_ylabel("The value of ZT by Machine learning")
    ax4 = plt.subplot2grid((3, 2), (2, 1))
    ax4.bar(range(1, 121), pd2[240:360, 1])
    ax4.set_xlim(b_x_slice_set)
    ax4.set_ylim(b_y_slice_set)
    ax4.set_xlabel("Number of compounds")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.0)
    # plt.savefig('plt_valid_fig3.pdf')
    plt.show()


if __name__ == '__main__':
    head_dir = r'..\data'
    fn = os.path.join(head_dir, r'valid_40.csv')
    fn1 = r'10_for_check.csv'
    fn2 = r'30_for_predict.csv'
    # deal_data(fn, os.path.join(head_dir, fn1), os.path.join(head_dir, fn2))
    # use_ml_to_predict(os.path.join(head_dir, fn2))
    
    save_dir = r'..\train\training_module'
    data_file1 = os.path.join(save_dir, 'z_result_valid_%s.out' % fn1)
    data_file2 = os.path.join(save_dir, 'z_result_valid_%s.out' % fn2)
    plt_predict_cal(data_file1, data_file2)
