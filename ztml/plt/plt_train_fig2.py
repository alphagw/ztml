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


def read_mse_data(fn):
    with open(fn, 'r') as f:
        data = np.array([[float(m.split(':')[-1]) for m in i.split('|')] for i in f.readlines()])
    return data


def read_cal_predit(fn):
    with open(fn, 'r') as f:
        data = np.array([i.split() for i in f.readlines()[0:]], dtype=np.float)
    return data
    
    
def run_mse(fn, outfn):
    dd = read_mse_data(fn)
    plt_mse(dd, outfn)


def plt_result(predict_data, training_data, text=None, save_fn=None, show=False):
    
    a0 = predict_data.pop(0)
    t0 = training_data.pop(0)
    predict_data.insert(2, a0)
    training_data.insert(2, t0)
    
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

    nrow = 2
    ncol = 3
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 8))
    plt.rc('font', family='Times New Roman', weight='normal')
    axes = axes.flatten()
    
    assert axes.shape[0] == len(predict_data) == len(training_data)
    if text is not None:
        assert axes.shape[0] == len(text)
        
    for i in range(axes.shape[0]):
        ax = axes[i]
        pd1 = predict_data[i]
        ax.scatter(pd1[:, 0], pd1[:, 1], edgecolors='white', color='#347FE2', linewidths=0.2)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd1[:, 0], pd1[:, 1])
        slice_set = 0.0, 1.25
        _tmp_xy = np.linspace(slice_set, pd1.shape[0])
        ax.plot(_tmp_xy, _tmp_xy, '#F37878', linewidth=3, alpha=0.8)
        ax.set_xlim(slice_set)
        ax.set_ylim(slice_set)
        ax.set_xlabel("Calculated", fontdict=label_font)
        ax.set_ylabel("Predicted", fontdict=label_font)
        if i > 3:
            ax.text(0.05, 0.9, text[i] % r_value**2, fontdict=legend_font)
        else:
            ax.text(0.10, 0.9, text[i]% r_value**2, fontdict=legend_font)

        ax.text(0.01, 1.3, pindex[i], fontdict=index_label_font)
        ax.set_xticklabels([round(i, 2) for i in ax.get_xticks()], tick_font_dict)
        ax.set_yticklabels([round(i, 2) for i in ax.get_yticks()], tick_font_dict)
        # ax.tick_params(axis='both', labelsize=tick_font_size)
        d = ax.get_position()
        print(i, d)
        tdata = training_data[i][1:, :]
        tx = tdata[:, 0]
        ytrain = tdata[:, -2]
        ytest = tdata[:, -1]

        # left, bottom, width, height = 1/ncol*0.66 * (i+1), 1/nrow * 1.26 * (int(i / ncol) + 1), 0.125, 0.12
        # left, bottom, width, height = d.x0 + d.width * 1/ncol, d.y0+0.12/nrow, 0.125, 0.12
        left, bottom, width, height = sax[i]
        
        if i == 3:
            left = left + 0.017
            width = width - 0.01
            train_final_mean = np.mean(ytrain[8000:])
            test_final_mean = np.mean(ytest[8000:])
        else:
            left = left
            width = width
            train_final_mean = np.mean(ytrain[2500:])
            test_final_mean = np.mean(ytest[2500:])

        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(tx, ytrain, c='#347FE2', linewidth=1.2, label='train')
        ax2.plot(tx, ytest, c='#F37878', linewidth=1.2, label='test')
        if i == 3:
            ax2.set_xlim(-120, 8000)
            ax2.set_ylim(-0.001, 0.2)
            ax2.text(2000, 0.05, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean))

        elif (i == 1) or (i == 0):
            ax2.set_xlim(-120, 3000)
            ax2.set_ylim(-0.001, 0.08)
            ax2.text(1000, 0.02, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean))
        else:
            ax2.set_xlim(-120, 3000)
            ax2.set_ylim(-0.001, 0.04)
            ax2.text(1000, 0.01, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean))

        
        ax2.set_ylabel('MSE')
        # ax2.set_xlabel('Steps')
        ax2.legend(fontsize=8)
        # plt.xticks([])
        # plt.yticks([])
        plt.tight_layout()

    if save_fn is not None:
        plt.savefig(save_fn, dpi=600)
        
    if show:
        plt.show()

if __name__ == '__main__':
    # fn, ofn = r"training_module/out_run3.train", 'train.pdf'
    # fn, ofn = r"training_module/out_run3.test", 'test.pdf'
    label = 'run1'
    save_dir = r'..\train\training_module'
    # run_mse(os.path.join(save_dir, 'running_%s.log' % label), 'training_%s.pdf' % label)
    text = ["Activation        : Sigmod\nOptimizer        : Adam\nHidden Layers : [100, 50, 20]\nR-squared(R2) : %.5f",
            "Activation        : Tanh\nOptimizer        : Adam\nHidden Layers : [100, 50, 20]\nR-squared(R2) : %.5f",
            "Activation        : Relu\nOptimizer        : Adam\nHidden Layers : [100, 50, 20]\nR-squared(R2) : %.5f",
            "Activation        : Relu\nOptimizer        : SGD\nHidden Layers : [100, 50, 20]\nR-squared(R2) : %.5f",
            "Activation        : Relu\nOptimizer        : Adam\nHidden Layers : [100, 100, 50, 20]\nR-squared(R2) : %.5f",
            "Activation        : Relu\nOptimizer        : Adam\nHidden Layers : [500, 100, 50, 20]\nR-squared(R2) : %.5f"]
    for i in ['train_30_train.csv', 'train_30_test.csv', 'valid_40.csv']:
        predict_data, training_data = [], []
        # for label in ['3layer_100_Elu', '3layer_100_PRelu', '3layer_100_sigmod', '3layer_100_Tanh', '3layer_100', '4layer_100', '4layer_500']:
        for label in ['3layer_100', '3layer_100_sigmod', '3layer_100_Tanh',
                      '3layer_100_sgd', '4layer_100', '4layer_500']: #'3layer_100_Elu', '3layer_100_PRelu',
            training_fn = os.path.join(save_dir, 'running_%s.log' % label)
            training_data.append(read_mse_data(training_fn))

            output_fn = os.path.join(save_dir, 'result_%s_%s.out' % (i, label))
            predict_data.append(read_cal_predit(output_fn))
        
        save_fn = 'plt_%s_fig2.pdf' % i
        # plt_result(predict_data, training_data, text, save_fn=None, show=True)
        plt_result(predict_data, training_data, text, save_fn=save_fn, show=False)

        exit()
