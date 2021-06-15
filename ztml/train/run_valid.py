#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/06/15 22:01:37'

import os
from ztml.train.train import ttest
from ztml.train.train_Ntype import ntype_ttest


def use_ml_to_predict_zt(head_dir, fname, has_t=True):
    save_dir = r'..\train\training_module'
    nfeature = 28
    hidden_layer = [500, 100, 50, 20]  # [100, 50, 20]  [100, 100, 50, 20]
    label = '4layer_500'  # '3layer_100_Elu', '3layer_100_PRelu', '3layer_100_sigmod', '3layer_100_Tanh', '3layer_100', '4layer_100', '4layer_500'
    
    ttest(test_csv_fn=os.path.join(head_dir, fname),
          mp_fn=os.path.join(save_dir, 'dnn_params_3000_%s.pkl' % label),
          output_fn='z_result_valid_has_t_%s.out' % fname,
          save_dir=save_dir, n_feature=nfeature, HIDDEN_NODES=hidden_layer,
          batch_size=500, shuffle=False,
          has_t=has_t)


def use_ml_to_predict_ntype(head_dir, fname, has_t=True):
    save_dir = r'..\train\training_module'
    nfeature = 28
    hidden_layer = [500, 100, 50, 20]  # [100, 50, 20]  [100, 100, 50, 20]
    label = 'N_type_4layer_500'  # '3layer_100_Elu', '3layer_100_PRelu', '3layer_100_sigmod', '3layer_100_Tanh', '3layer_100', '4layer_100', '4layer_500'
    
    ntype_ttest(test_csv_fn=os.path.join(head_dir, fname),
          mp_fn=os.path.join(save_dir, 'dnn_params_5000_%s.pkl' % label),
          output_fn='ntype_z_result_valid_has_t_%s.out' % fname, shuffle=False,
          save_dir=save_dir, n_feature=nfeature, HIDDEN_NODES=hidden_layer,
          batch_size=500, zt=False, n_output=1, has_t=has_t)


if __name__ == '__main__':
    head_dir = r'..\data'
    fn2 = r'30_for_predict.csv'
    fn1 = r'10_for_check.csv'
    
    # has_t 指定想要获取那一列特征并且输出到结果中，-5列是温度(必须去掉label列)，第3列是原子总数
    has_t = [-3, 2, 12]
    use_ml_to_predict_zt(head_dir, fn1, has_t=has_t)
    use_ml_to_predict_zt(head_dir, fn2, has_t=has_t)
    use_ml_to_predict_ntype(head_dir, fn1, has_t=has_t)
    use_ml_to_predict_ntype(head_dir, fn2, has_t=has_t)
