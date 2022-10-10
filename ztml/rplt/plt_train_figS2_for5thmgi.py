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

aa = lambda x: 0.05 if x < 0.5 else 0.95

def read_mse_data(fn):
    with open(fn, 'r') as f:
        data = np.array([[float(m.split(':')[-1]) for m in i.split('|')] for i in f.readlines()])
    return data


def read_cal_predit(fn):
    with open(fn, 'r') as f:
        data = np.array([i.split() for i in f.readlines()[0:]], dtype=np.float)
    newpredict = np.array([1.05 if i > 0.5 else 0.05 for i in data[:, 1]], dtype=np.float)
    return np.vstack((data[:, 0], newpredict)).transpose()


def plt_result(predict_data, training_data, text=None, save_fn=None, show=False, point_num=None):
    label_font = {"fontsize": 16, 'family': 'Times New Roman'}
    legend_font = {"fontsize": 12, 'family': 'Times New Roman'}
    tick_font_dict = {"fontsize": 14, 'family': 'Times New Roman'}
    index_label_font = {"fontsize": 18, 'weight': 'light', 'family': 'Times New Roman'}
    pindex = ['(g)', '(h)', '(i)', '(j)', '(k)', '(l)']
    _xwd, _ywd = 0.168, 0.08
    sax = [[0.305 + 0.020-0.02, 0.730+0.08, _xwd, _ywd],
           [0.800 + 0.004-0.02, 0.730+0.08, _xwd, _ywd],
           [0.305 + 0.010-0.02, 0.410+0.08, _xwd, _ywd],
           [0.800 + 0.019-0.02, 0.410+0.08, _xwd, _ywd],
           [0.305 + 0.000-0.00, 0.078+0.08, _xwd, _ywd],
           [0.800 + 0.000-0.00, 0.078+0.08, _xwd, _ywd]]

    nrow = 3
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 11))
    plt.rc('font', family='Times New Roman', weight='light')
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["ytick.direction"] = 'in'
    axes = axes.flatten()

    assert axes.shape[0] == len(predict_data) == len(training_data)
    if text is not None:
        assert axes.shape[0] == len(text)
    colors = {'Calculated': '#3CAF6F', 'Predicted': '#FF8C00'}

    for i in range(axes.shape[0]):
        ax = axes[i]
        pd1 = predict_data[i]
        ax.scatter(range(1, len(pd1)+1), pd1[:, 0], edgecolors='#53c482', color=colors['Calculated'], alpha=1, linewidths=0.01, s=90)
        ax.scatter(range(1, len(pd1)+1), pd1[:, 1], edgecolors='#ffa227', color=colors['Predicted'], alpha=1, linewidths=0.01, s=90)
        wrong_num = 0
        for nnn in range(pd1.shape[0]):
            if aa(pd1[nnn][0]) != aa(pd1[nnn][1]):
                wrong_num += 1
        
        # ax.text(10, 0.45, 'Accuracy: %.3f' % ((1 - wrong_num / pd1.shape[0]) * 100) + '%', fontdict=tick_font_dict)

        slice_set = -0.1, 1.15
        _tmp_xy = np.linspace(slice_set, pd1.shape[0])
        ax.set_xlim(-5, len(pd1)+5)
        ax.set_ylim(slice_set)
        ax.text(3, 0.55, text[i] % ((1 - wrong_num / pd1.shape[0]) * 100) + '%', fontdict=legend_font)

        ax.text(0.01, 1.2, pindex[i], fontdict=index_label_font)
        ax.set_xticks([0, 100, 200, 252])
        ax.set_xticklabels(ax.get_xticks(), tick_font_dict)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['N-type', 'P-type'], tick_font_dict)
        # ax.set_xlabel('Data point', tick_font_dict)
        ax.tick_params(axis='both', direction='in')

        if i == 0:
            labels = list(colors.keys())
            handles = [plt.Rectangle((0, 0), 1, 1, color=colors[label]) for label in labels]
            ax.legend(handles=handles, labels=labels, ncol=2, loc='upper left', bbox_to_anchor=[0.2, 0.3])

        d = ax.get_position()
        print(i, d)
        tdata = training_data[i][1:, :]
        tx = tdata[:, 0]
        ytrain = tdata[:, -2]
        ytest = tdata[:, -1]

        # tmp_data = np.abs(ytrain - ytest)
        # tmp_data_min = np.sort(tmp_data)
        # print(tmp_data_min[:5])
        # print([np.where(tmp_data == i) for i in tmp_data_min[:5]])
        # # print(tmp_data[np.where(tmp_data == tmp_data_min)])
        # print('\n')
        
        left, bottom, width, height = sax[i]
        
        if i == 3:
            train_final_mean = np.mean(ytrain[8000:])
            test_final_mean = np.mean(ytest[8000:])
        else:
            train_final_mean = np.mean(ytrain[2500:])
            test_final_mean = np.mean(ytest[2500:])

        ax2 = fig.add_axes([left, bottom, width, height])
        if i > 1:
            lw = 1.2
        else:
            lw = 2.2
        ax2.plot(tx, ytrain, c='#347FE2', linewidth=lw, label='train')
        ax2.plot(tx, ytest, c='#F37878', linewidth=1.2, label='test')
        if point_num is not None:
            num = point_num[i]
            train_final_mean = ytrain[num]
            test_final_mean = ytest[num]
            def dodododod():
                ax2.plot([num + 1] * 10, np.linspace(0, 1, 10), '-.', c='black')
                # ax2.plot(range(1, 1000), [ytrain[num]] * 999, '-.')
                tt = [i for i in ax2.get_xticks() if i > 0 and i != 200]
                tt.append(num + 1)
                tt = sorted(tt)
                ax2.set_xticks(tt)
                
        if i == 0:
            ax2.set_xlim(-1, 50)
            ax2.set_ylim(-0.001, 1)
            ax2.text(12, 0.6, 'train:%.5f\ntest:%.5f' % (float(train_final_mean), float(test_final_mean)), fontdict=legend_font)
        elif i == 1:
            ax2.set_xlim(-1, 100)
            ax2.set_ylim(-0.001, 1)
            ax2.text(30, 0.5, 'train:%.5f\ntest:%.5f' % (float(train_final_mean), float(test_final_mean)), fontdict=legend_font)
        elif i == 2:
            ax2.set_xlim(-1, 50)
            ax2.set_ylim(-0.001, 1)
            ax2.text(15, 0.5, 'train:%.5f\ntest:%.5f' % (float(train_final_mean), float(test_final_mean)), fontdict=legend_font)
        elif i == 3:
            ax2.set_xlim(-1, 700)
            ax2.set_ylim(-0.001, 1)
            ax2.text(180, 0.5, 'train:%.5f\ntest:%.5f' % (float(train_final_mean), float(test_final_mean)), fontdict=legend_font)
        elif i == 4:
            ax2.set_xlim(-120, 700)
            ax2.set_ylim(-0.001, 1)
            ax2.text(220, 0.5, 'train:%.5f\ntest:%.5f' % (float(train_final_mean), float(test_final_mean)), fontdict=legend_font)
        else:
            ax2.set_xlim(-120, 700)
            ax2.set_ylim(-0.001, 1)
            ax2.text(250, 0.5, 'train:%.5f\ntest:%.5f' % (float(train_final_mean), float(test_final_mean)), fontdict=legend_font)
        
        dodododod()
        ax2.set_ylabel('Cross Entropy')
        
        if i == 0:
            ax2.set_xlim(-1, 50)
        elif i == 1:
            ax2.set_xlim(-1, 100)
        elif i == 2:
            ax2.set_xlim(-1, 50)
        elif i == 3:
            ax2.set_xlim(-10, 700)
        elif i == 4:
            ax2.set_xlim(-10, 700)
        else:
            ax2.set_xlim(-10, 700)
        
    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.98, top=0.96, wspace=0.21, hspace=0.26)
    if save_fn is not None:
        plt.savefig(save_fn, dpi=600)
        
    if show:
        plt.show()

    plt.savefig('plt_figS2_for5thmgi.pdf', dpi=600)
    # plt.savefig('plt_figS2.jpg', dpi=600)


if __name__ == '__main__':
    # fn, ofn = r"training_module/out_run3.train", 'train.pdf'
    # fn, ofn = r"training_module/out_run3.test", 'test.pdf'
    label = 'run1'
    save_dir = r'..\rtrain\final_ntype_training_module'
    # run_mse(os.path.join(save_dir, 'running_%s.log' % label), 'training_%s.pdf' % label)
    text = ["Activation : Relu\nOptimizer : Adam\nHidden Layers :\n[100, 50, 20]\nAccuracy: %.3f",
            "Activation : Sigmod\nOptimizer : Adam\nHidden Layers :\n[100, 50, 20]\nAccuracy: %.3f",
            "Activation : Tanh\nOptimizer : Adam\nHidden Layers :\n[100, 50, 20]\nAccuracy: %.3f",
            "Activation : Tanh\nOptimizer : SGD\nHidden Layers :\n[100, 50, 20]\nAccuracy: %.3f",
            "Activation : Tanh\nOptimizer : SGD\nHidden Layers :\n[100, 100, 50, 20]\nAccuracy: %.3f",
            "Activation : Tanh\nOptimizer : SGD\nHidden Layers :\n[500, 100, 50, 20]\nAccuracy: %.3f"]
    labels = ["3layer_100_relu", "3layer_100_sigmoid", "3layer_100_tanh",
              "3layer_100_relu_sgd", "4layer_100", "4layer_500"]
    # nums = [8, 21, 8, 658, 558, 599]
    nums = [8, 20, 5, 138, 193, 208]

    for i in ['train_30_train.csv', 'train_30_test.csv', 'valid_40.csv']:
        predict_data, training_data = [], []
        # for label in ['3layer_100_Elu', '3layer_100_PRelu', '3layer_100_sigmod', '3layer_100_Tanh', '3layer_100', '4layer_100', '4layer_500']:
        for label in labels:
            training_fn = os.path.join(save_dir, 'running_%s.log' % label)
            training_data.append(read_mse_data(training_fn))

            output_fn = os.path.join(save_dir, 'result_%s_%s.out' % (i, label))
            predict_data.append(read_cal_predit(output_fn))
        
        save_fn = 'plt_%s_figS2.pdf' % i
        plt_result(predict_data, training_data, text, save_fn=None, show=False, point_num=nums)
        # plt_result(predict_data, training_data, text, save_fn=save_fn, show=False)

        exit()
