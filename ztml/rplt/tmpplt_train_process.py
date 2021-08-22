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
import scipy.stats


def plt_mse(data, outfn):
    # lv #3CAF6F
    fig = plt.figure()
    data = data[1:, :]
    x = data[:, 0]
    ytrain = data[:, -2]
    ytest = data[:, -1]

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(x, ytrain, c='#347FE2', linewidth=3.2)
    ax1.plot(x, ytest, c='#F37878' ,linewidth=3.2)
    ax1.set_xlim(-1, 300)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel("Mean Square Error (MSE)")
    left, bottom, width, height = 0.4, 0.4, 0.35, 0.35
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(x, ytrain, c='#347FE2', linewidth=2.2)
    ax2.plot(x, ytest, c='#F37878', linewidth=2.2)
    
    train_final_mean = np.mean(ytrain[3000:])
    test_final_mean = np.mean(ytest[3000:])
    ax2.plot(range(300, 5000), [train_final_mean]*(5000-300), 'r', linestyle='--', linewidth=2.2)
    ax2.text(2000, 0.004, 'MSE=%.5f' % train_final_mean)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('MSE')
    ax2.set_xlim(300, 5000)
    ax2.set_ylim(0, 0.01)
    ax2.set_xticks([300, 1000, 2000, 3000, 4000, 5000])
    
    plt.savefig(outfn)


def read_mse_2data(fn):
    with open(fn, 'r') as f:
        data =[[m.split(':')[-1].split() for m in i.split('|')] for i in f.readlines()]
        dd = []
        for m in data:
            aa = []
            for xx in m:
                for ii in xx:
                    aa.append(ii)
            dd.append(aa)
    return np.array(dd, dtype=np.float)


def plt_result(data, save_fn=None, show=False):
    
    x = data[:, 0]
    zttrain = data[:, 2]
    nttrain = data[:, 3]
    ztloss = data[:, 4]
    ntloss = data[:, 5]
    
    label_font = {"fontsize": 14, 'family': 'Times New Roman'}
    legend_font = {"fontsize": 12, 'family': 'Times New Roman'}
    tick_font_size = 12
    tick_font_dict = {"fontsize": 12, 'family': 'Times New Roman'}
    index_label_font = {"fontsize": 20, 'weight': 'bold', 'family': 'Times New Roman'}
    pindex = ['A', 'B', 'C', 'D', 'E', 'F']
    _xwd, _ywd = 0.118, 0.12
    sax = [[0.18098039215686275 + 0.020, 0.60, _xwd, _ywd],
           [0.49450980392156866 + 0.035, 0.60, _xwd, _ywd],
           [0.82803921568627460 + 0.035, 0.60, _xwd, _ywd],
           [0.18098039215686275 + 0.020, 0.11, _xwd, _ywd],
           [0.49450980392156866 + 0.035, 0.11, _xwd, _ywd],
           [0.82803921568627460 + 0.035, 0.11, _xwd, _ywd]]

    nrow = 1
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 8))
    plt.rc('font', family='Times New Roman', weight='normal')
    axes = axes.flatten()
    ax1, ax2 = axes[0], axes[1]
    
    ax1.plot(x, zttrain)
    # ax1.plot(x, ztloss)
    
    ax2.plot(x, nttrain)
    # ax2.plot(x, ntloss)
    
    ax1.set_ylabel('MSE')
    ax1.legend(fontsize=8)
    ax2.set_ylabel('MSE')
    ax2.legend(fontsize=8)
    plt.tight_layout()

    if save_fn:
        plt.savefig(save_fn, dpi=600)
    
    if show:
        plt.show()
        
    # assert axes.shape[0] == len(predict_data) == len(training_data)
    # if text is not None:
    #     assert axes.shape[0] == len(text)
    #
    # for i in range(axes.shape[0]):
    #     ax = axes[i]
    #     pd1 = predict_data[i]
    #     ax.scatter(pd1[:, 0], pd1[:, 1], edgecolors='white', color='#347FE2', linewidths=0.2)
    #     slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd1[:, 0], pd1[:, 1])
    #     slice_set = 0.0, 1.25
    #     _tmp_xy = np.linspace(slice_set, pd1.shape[0])
    #     ax.plot(_tmp_xy, _tmp_xy, '#F37878', linewidth=3, alpha=0.8)
    #     ax.set_xlim(slice_set)
    #     ax.set_ylim(slice_set)
    #     ax.set_xlabel("Calculated", fontdict=label_font)
    #     ax.set_ylabel("Predicted", fontdict=label_font)
    #     if i > 3:
    #         ax.text(0.05, 0.9, text[i] % r_value**2, fontdict=legend_font)
    #     else:
    #         ax.text(0.10, 0.9, text[i]% r_value**2, fontdict=legend_font)
    #
    #     ax.text(0.01, 1.3, pindex[i], fontdict=index_label_font)
    #     ax.set_xticklabels([round(i, 2) for i in ax.get_xticks()], tick_font_dict)
    #     ax.set_yticklabels([round(i, 2) for i in ax.get_yticks()], tick_font_dict)
    #     # ax.tick_params(axis='both', labelsize=tick_font_size)
    #     d = ax.get_position()
    #     print(i, d)
    #     tdata = training_data[i][1:, :]
    #     tx = tdata[:, 0]
    #     ytrain = tdata[:, -2]
    #     ytest = tdata[:, -1]
    #
    #     # left, bottom, width, height = 1/ncol*0.66 * (i+1), 1/nrow * 1.26 * (int(i / ncol) + 1), 0.125, 0.12
    #     # left, bottom, width, height = d.x0 + d.width * 1/ncol, d.y0+0.12/nrow, 0.125, 0.12
    #     left, bottom, width, height = sax[i]
    #
    #     if i == 3:
    #         left = left + 0.017
    #         width = width - 0.01
    #         train_final_mean = np.mean(ytrain[:4000])
    #         test_final_mean = np.mean(ytest[:4000])
    #     else:
    #         left = left
    #         width = width
    #         train_final_mean = np.mean(ytrain[2500:])
    #         test_final_mean = np.mean(ytest[2500:])
    #
    #     ax2 = fig.add_axes([left, bottom, width, height])
    #     ax2.plot(tx, ytrain, c='#347FE2', linewidth=1.2, label='train')
    #     ax2.plot(tx, ytest, c='#F37878', linewidth=1.2, label='test')
    #     if i == 3:
    #         ax2.set_xlim(-120, 5000)
    #         ax2.set_ylim(-0.001, 0.2)
    #         ax2.text(2000, 0.05, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean))
    #
    #     elif (i == 1) or (i == 0):
    #         ax2.set_xlim(-120, 3000)
    #         ax2.set_ylim(-0.001, 0.2)
    #         ax2.text(1000, 0.05, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean))
    #     else:
    #         ax2.set_xlim(-120, 3000)
    #         ax2.set_ylim(-0.001, 0.2)
    #         ax2.text(1000, 0.05, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean))

        


if __name__ == '__main__':
    fn = os.path.join(r"..\rtrain", "2training_module_2output", "running_3layer_100.log")
    data = read_mse_2data(fn)
    plt_result(data, save_fn=False, show=True)
