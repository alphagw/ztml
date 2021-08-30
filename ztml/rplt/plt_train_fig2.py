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


def read_mse_data(fn):
    with open(fn, 'r') as f:
        data = np.array([[float(m.split(':')[-1]) for m in i.split('|')] for i in f.readlines()])
    return data


def read_cal_predit(fn):
    with open(fn, 'r') as f:
        data = np.array([i.split() for i in f.readlines()[0:]], dtype=np.float)
    return data
    

def plt_result(predict_data, training_data, text=None, save_fn=None, show=False):
    
    # a0 = predict_data.pop(0)
    # t0 = training_data.pop(0)
    # predict_data.insert(2, a0)
    # training_data.insert(2, t0)
    
    label_font = {"fontsize": 16, 'family': 'Times New Roman'}
    legend_font = {"fontsize": 12, 'family': 'Times New Roman'}
    tick_font_dict = {"fontsize": 14, 'family': 'Times New Roman'}
    index_label_font = {"fontsize": 18, 'weight': 'normal', 'family': 'Times New Roman'}
    pindex = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    _xwd, _ywd = 0.168, 0.08
    sax = [[0.305 + 0.02, 0.73, _xwd-0.015, _ywd],
           [0.800 + 0.004, 0.73, _xwd, _ywd],
           [0.305 + 0.01, 0.41, _xwd, _ywd],
           [0.800 + 0.019, 0.41, _xwd-0.01, _ywd],
           [0.305 + 0.0, 0.078, _xwd, _ywd],
           [0.800 + 0.0, 0.078, _xwd, _ywd]]

    nrow = 3
    ncol = 2
    fig, axes = plt.subplots(nrow, ncol, figsize=(9, 11))
    plt.rc('font', family='Times New Roman', weight='normal')
    plt.rcParams["xtick.direction"] = 'in'
    plt.rcParams["ytick.direction"] = 'in'
    axes = axes.flatten()
    
    assert axes.shape[0] == len(predict_data) == len(training_data)
    if text is not None:
        assert axes.shape[0] == len(text)
        
    for i in range(axes.shape[0]):
        ax = axes[i]
        pd1 = predict_data[i]
        ax.scatter(pd1[:, 0], pd1[:, 1], edgecolors='white', color='#347FE2', linewidths=0.2)
        slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd1[:, 0], pd1[:, 1])
        rmse = np.sqrt(np.mean(np.square(pd1[:, 0] - pd1[:, 1])))

        if i <= 1:
            slice_set = -0.3, 1.25
            ax.text(-0.25, 0.67, text[i] % (r_value**2, rmse, (1-rmse)*100) + "%", fontdict=legend_font)
            ax.text(-0.25, 1.3, pindex[i], fontdict=index_label_font)
        else:
            slice_set = -0.1, 1.25
            ax.text(-0.05, 0.75, text[i] % (r_value**2, rmse, (1-rmse)*100) + "%", fontdict=legend_font)
            ax.text(-0.05, 1.3, pindex[i], fontdict=index_label_font)

        _tmp_xy = np.linspace(slice_set, pd1.shape[0])
        ax.plot(_tmp_xy, _tmp_xy, '#F37878', linewidth=3, alpha=0.8)
        ax.set_xlim(slice_set)
        ax.set_ylim(slice_set)
        ax.set_xlabel("Calculated", fontdict=label_font)
        ax.set_ylabel("Predicted", fontdict=label_font)

        ax.set_xticklabels([round(i, 2) for i in ax.get_xticks()], tick_font_dict)
        ax.set_yticklabels([round(i, 2) for i in ax.get_yticks()], tick_font_dict)
        ax.tick_params(axis='both', direction='in')
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
        
        ax2 = fig.add_axes([left, bottom, width, height])
        ax2.plot(tx, ytrain, c='#347FE2', linewidth=1.2, label='train')
        ax2.plot(tx, ytest, c='#F37878', linewidth=1.2, label='test')
        if i == 2:
            train_final_mean = np.mean(ytrain[200:2500])
            test_final_mean = np.mean(ytest[200:2500])
            ax2.set_xlim(-80, 3000)
            ax2.set_ylim(-0.001, 0.05)
            ax2.text(800, 0.05, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean), fontdict=legend_font)
            ax2.set_yticks([0.0, 0.1])
        elif i == 0:
            train_final_mean = np.mean(ytrain[500:2500])
            test_final_mean = np.mean(ytest[500:2500])
            ax2.set_xlim(-80, 3000)
            ax2.set_ylim(-0.001, 0.1)
            ax2.text(800, 0.05, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean), fontdict=legend_font)
            ax2.set_yticks([0.0, 0.1])
        elif i == 1:
            train_final_mean = np.mean(ytrain[500:2500])
            test_final_mean = np.mean(ytest[500:2500])
            ax2.set_xlim(-80, 3000)
            ax2.set_ylim(-0.001, 0.2)
            ax2.text(800, 0.09, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean), fontdict=legend_font)
            ax2.set_yticks([0.0, 0.2])
        elif i >= 3:
            train_final_mean = np.mean(ytrain[10000:12000])
            test_final_mean = np.mean(ytest[10000:12000])
            ax2.set_xlim(-200, 12000)
            ax2.set_ylim(-0.001, 0.15)
            ax2.text(3000, 0.08, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean), fontdict=legend_font)
            ax2.set_yticks([0.0, 0.15])

        # else:
        #     train_final_mean = np.mean(ytrain[:4000])
        #     test_final_mean = np.mean(ytest[:4000])
        #     ax2.set_xlim(-120, 3000)
        #     ax2.set_ylim(-0.001, 0.2)
        #     ax2.text(1000, 0.05, 'train:%.5f\ntest :%.5f' % (train_final_mean, test_final_mean))

        
        ax2.set_ylabel('MSE', labelpad=-12, fontdict=legend_font)
        # ax2.set_xlabel('Steps')
        # ax2.legend(fontsize=8)
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.06, right=0.98, top=0.96, wspace=0.21, hspace=0.26)

    if save_fn is not None:
        plt.savefig(save_fn, dpi=600)
        
    if show:
        plt.show()
    plt.savefig('plt_fig2.pdf', dpi=600)
    plt.savefig('plt_fig2.png', dpi=600)
    

if __name__ == '__main__':
    # fn, ofn = r"training_module/out_run3.train", 'train.pdf'
    # fn, ofn = r"training_module/out_run3.test", 'test.pdf'
    # label = 'run1'
    save_dir = r'..\rtrain\final_training_module'
    # run_mse(os.path.join(save_dir, 'running_%s.log' % label), 'training_%s.pdf' % label)
    text = ["Hidden Layers : [100, 50, 20]\nR-squared(R2) : %.5f\nActivation        : Sigmod\nOptimizer     : Adam\nRMSE: %.5f\nAccuracy: %.3f",
            "Hidden Layers : [100, 50, 20]\nR-squared(R2) : %.5f\nActivation        : Tanh\nOptimizer     : Adam\nRMSE: %.5f\nAccuracy: %.3f",
            "Hidden Layers : [100, 50, 20]\nR-squared(R2) : %.5f\nActivation        : Relu\nOptimizer     : Adam\nRMSE: %.5f\nAccuracy: %.3f",
            "Hidden Layers : [100, 50, 20]\nR-squared(R2) : %.5f\nActivation        : Relu\nOptimizer     : SGD\nRMSE: %.5f\nAccuracy: %.3f",
            "Hidden Layers : [100, 100, 50, 20]\nR-squared(R2) : %.5f\nActivation        : Relu\nOptimizer     : SGD\nRMSE: %.5f\nAccuracy: %.3f",
            "Hidden Layers : [500, 100, 50, 20]\nR-squared(R2) : %.5f\nActivation        : Relu\nOptimizer     : SGD\nRMSE: %.5f\nAccuracy: %.3f"]
    for i in ['train_30_train.csv', 'train_30_test.csv', 'valid_40.csv']:
    # for i in ['train_30_test.csv', 'train_30_train.csv', 'valid_40.csv']:
        predict_data, training_data = [], []
        # for label in ['3layer_100_Elu', '3layer_100_PRelu', '3layer_100_sigmod', '3layer_100_Tanh', '3layer_100', '4layer_100', '4layer_500']:
        # for label in ['3layer_100', '3layer_100_sigmod', '3layer_100_Tanh',
        #               '3layer_100_sgd', '4layer_100', '4layer_500']: #'3layer_100_Elu', '3layer_100_PRelu',
        # labels = ["3layer_100_adam", "3layer_100_sgd", "3layer_100_sgd_Sigmod", "3layer_100_sgd_Tanh",
        #           "4layer_100_sgd", "4layer_500_sgd"]

        labels = ["3layer_100_sigmoid", "3layer_100_tanh",
                  "3layer_100_relu",  "3layer_100_relu_sgd",
                  "4layer_100", "4layer_500"]

        for label in labels: #'3layer_100_Elu', '3layer_100_PRelu',
            training_fn = os.path.join(save_dir, 'running_%s.log' % label)
            training_data.append(read_mse_data(training_fn))

            output_fn = os.path.join(save_dir, 'result_%s_%s.out' % (i, label))
            predict_data.append(read_cal_predit(output_fn))
            print(training_fn, output_fn)
        save_fn = 'plt_%s_fig2train.pdf' % i
        plt_result(predict_data, training_data, text, save_fn='plt_fig2.jpg', show=False)
        # plt_result(predict_data, training_data, text, save_fn=save_fn, show=False)

        exit()
