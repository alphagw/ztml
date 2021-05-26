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


def plt_mse(data):
    fig = plt.figure()
    data = data[1:, :]
    x = data[:, 0]
    y = data[:, -1]

    left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
    ax1 = fig.add_axes([left, bottom, width, height])
    ax1.plot(x, y, linewidth=3.2)
    ax1.set_xlim(-1, 300)
    ax1.set_xlabel('Steps')
    ax1.set_ylabel("Mean Square Error (MSE)")
    left, bottom, width, height = 0.4, 0.4, 0.35, 0.35
    ax2 = fig.add_axes([left, bottom, width, height])
    ax2.plot(x, y, linewidth=2.2)
    final_mean = np.mean(y[3000:])
    ax2.plot(range(300, 5000), [final_mean]*(5000-300), 'r', linestyle='--', linewidth=2.2)
    ax2.text(2000, 0.004, 'MSE=%.5f' % final_mean)
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('MSE')
    ax2.set_xlim(300, 5000)
    ax2.set_ylim(0, 0.01)
    ax2.set_xticks([300, 1000, 2000, 3000, 4000, 5000])
    
    plt.savefig('1.pdf')


def read_mse_data(fn):
    with open(fn, 'r') as f:
        data = np.array([[float(m.split(':')[-1]) for m in i.split('|')] for i in f.readlines()])
    return data
    
    
def run_mse():
    fn = r"training_module/test1"
    dd = read_mse_data(fn)
    plt_mse(dd)


def plt_result():
    fn = r"training_module/results.out"
    with open(fn, 'r') as f:
        data = np.array([i.split() for i in f.readlines()[1:]], dtype=np.float)
    actual = data[:, 0]
    predict =  data[:, 1]
    # print(actual, predict)
    plt.figure(1, (8, 8))
    plt.scatter(actual, predict, edgecolors='white', linewidths=0.2)
    slice_set = 0.2, 0.7
    _tmp_xy = np.linspace(slice_set, data.shape[0])
    plt.plot(_tmp_xy, _tmp_xy, 'r', linewidth=3)
    plt.xlim(slice_set)
    plt.ylim(slice_set)
    plt.xlabel("Calculated")
    plt.ylabel("Predicted")
    # plt.xticks([])
    # plt.yticks([])
    plt.savefig('1.pdf')


def plt_features():
    csv_file = r'G:\ztml\ztml\data\clean_data.csv'
    data = pd.read_csv(csv_file).values
    train_data = data[:, :-1]
    label = data[:, -1]
    a = np.corrcoef(data, rowvar=0)
    b = np.triu(a)
    auto_get_corelated_group(data=b, is_triu=True)
    # print(b[2, 67])
    # for ix, iy in ii:
    #     if ix != iy:
    #         print(a[ix, iy])
    # sns.heatmap(b, vmin=0, vmax=1)
    # plt.savefig('1.pdf')


def auto_get_corelated_group(data, coref_val=0.9, is_triu=False):
    if is_triu:
        pass
    else:
        data = np.triu(data)
        
    double_index = np.where(data > coref_val)
    index_final = [[double_index[0][m], double_index[1][m]]
                   for m in range(len(double_index[0]))
                   if double_index[0][m] != double_index[1][m]]
    print(index_final)
    fd = [[]]
    _tmp_data = index_final.pop(0)
    print(np.where(index_final[0][0] == fd[0][0]))
    # fd[0].append(index_final[][:, 0])
    print(fd)
    # for m in fn:
    

if __name__ == '__main__':
    # plt_result()
    plt_features()