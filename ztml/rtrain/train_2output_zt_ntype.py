#   coding:utf-8
#   This file is part of potentialmind.
#
#   potentialmind is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2019/11/08 15:36:27'

import os
import math
import numpy as np
import torch
import torch.nn as nn
from ztml.read_data import load_pmdata
from matfleet.utilities import now_time
from copy import deepcopy


class DNN2(nn.Module):
    DP_RADIO = 0.5
    B_ININT = -0.0
    
    def __init__(self, n_feature, n_hidden, zt_hidden, noptype_hidden,
                 batch_normalize=True, dropout=True, activation=nn.ReLU(), zt_act=nn.ReLU(), noptype_act=nn.Sigmoid()):
        super(DNN2, self).__init__()
        assert isinstance(n_hidden, (list, tuple))
        
        self.ACTIVATION = activation
        self.zt_ACTIVATION = zt_act
        self.nt_ACTIVATION = noptype_act
        self.do_bn = batch_normalize
        self.do_dp = dropout
        self.fcs, self.bns, self.dps = [], [], []
        self.ztfcs, self.ztbns, self.ztdps = [], [], []
        self.ntfcs, self.ntbns, self.ntdps = [], [], []
        self.bn_input = nn.BatchNorm1d(n_feature, momentum=0.5)
        self.n_hidden = [n_feature] + n_hidden
        self.zthidden = [self.n_hidden[-1]] + zt_hidden
        self.noptype_hidden = [self.n_hidden[-1]] + noptype_hidden
        
        for hid_index in range(1, len(self.n_hidden)):
            fc = torch.nn.Linear(self.n_hidden[hid_index - 1], self.n_hidden[hid_index])
            setattr(self, 'fc%d' % hid_index, fc)
            self._set_init(fc)
            self.fcs.append(fc)
            
            if self.do_bn:
                bn = nn.BatchNorm1d(self.n_hidden[hid_index], momentum=0.5)
                setattr(self, 'bn%d' % hid_index, bn)
                self.bns.append(bn)
            
            if self.do_dp:
                dp = torch.nn.Dropout(self.DP_RADIO)
                setattr(self, 'dp%s' % hid_index, dp)
                self.dps.append(dp)
        
        for zt_index in range(1, len(self.zthidden)):
            fc = torch.nn.Linear(self.zthidden[zt_index - 1], self.zthidden[zt_index])
            setattr(self, 'ztfc%d' % zt_index, fc)
            self._set_init(fc)
            self.ztfcs.append(fc)

            if self.do_bn:
                bn = nn.BatchNorm1d(self.zthidden[zt_index], momentum=0.5)
                setattr(self, 'ztbn%d' % zt_index, bn)
                self.ztbns.append(bn)

            if self.do_dp:
                dp = torch.nn.Dropout(self.DP_RADIO)
                setattr(self, 'ztdp%s' % zt_index, dp)
                self.ztdps.append(dp)
        
        for noptype_index in range(1, len(self.noptype_hidden)):
            fc = torch.nn.Linear(self.noptype_hidden[noptype_index - 1], self.noptype_hidden[noptype_index])
            setattr(self, 'ntfc%d' % noptype_index, fc)
            self._set_init(fc)
            self.ntfcs.append(fc)

            if self.do_bn:
                bn = nn.BatchNorm1d(self.noptype_hidden[noptype_index], momentum=0.5)
                setattr(self, 'ntbn%d' % noptype_index, bn)
                self.ntbns.append(bn)

            if self.do_dp:
                dp = torch.nn.Dropout(self.DP_RADIO)
                setattr(self, 'ntdp%s' % noptype_index, dp)
                self.ntdps.append(dp)
        
        self.zt_predict = torch.nn.Linear(self.zthidden[-1], 1)
        self._set_init(self.zt_predict)

        self.nt_predict = torch.nn.Linear(self.noptype_hidden[-1], 1)
        self._set_init(self.nt_predict)

    def _set_init(self, fc):
        nn.init.normal_(fc.weight, mean=0, std=0.1)
        nn.init.constant_(fc.bias, self.B_ININT)
    
    def forward(self, x):
        # if self.do_bn: x = self.bn_input(x)
        for i in range(len(self.n_hidden) - 1):
            x = self.fcs[i](x)
            if self.do_bn: x = self.bns[i](x)
            if self.do_dp: x = self.dps[i](x)
            x = self.ACTIVATION(x)
        share_x = torch.tensor(deepcopy(x.cpu().data.numpy())).cuda()
        
        for zi in range(len(self.zthidden) - 1):
            x = self.ztfcs[zi](x)
            if self.do_bn: x = self.ztbns[zi](x)
            if self.do_dp: x = self.ztdps[zi](x)
            x = self.zt_ACTIVATION(x)
        zt = self.zt_predict(x)
        
        for ni in range(len(self.noptype_hidden) - 1):
            share_x = self.ntfcs[ni](share_x)
            if self.do_bn: share_x = self.ntbns[ni](share_x)
            if self.do_dp: share_x = self.ntdps[ni](share_x)
            share_x = self.nt_ACTIVATION(share_x)
        nt = self.nt_predict(share_x)
        
        return torch.cat((zt, nt), 1)


class DNN(nn.Module):
    DP_RADIO = 0.3
    B_ININT = -0.0

    def __init__(self, n_feature, n_hidden, n_output, batch_normalize=True, dropout=True, activation=nn.ReLU()):
        super(DNN, self).__init__()
        assert isinstance(n_hidden, (list, tuple))

        self.ACTIVATION = activation
        self.do_bn = batch_normalize
        self.do_dp = dropout
        self.fcs , self.bns, self.dps = [], [], []
        
        self.bn_input = nn.BatchNorm1d(n_feature, momentum=0.5)
        self.n_hidden = [n_feature] + n_hidden
        
        for hid_index in range(1, len(self.n_hidden)):
            fc = torch.nn.Linear(self.n_hidden[hid_index - 1], self.n_hidden[hid_index])
            setattr(self, 'fc%d' % hid_index, fc)
            self._set_init(fc)
            self.fcs.append(fc)
            
            if self.do_bn:
                bn = nn.BatchNorm1d(self.n_hidden[hid_index], momentum=0.5)
                setattr(self, 'bn%d' % hid_index, bn)
                self.bns.append(bn)
            
            if self.do_dp:
                dp = torch.nn.Dropout(self.DP_RADIO)
                setattr(self, 'dp%s' % hid_index, dp)
                self.dps.append(dp)
        
        self.predict = torch.nn.Linear(self.n_hidden[-1], n_output)
        self._set_init(self.predict)
    
    def _set_init(self, fc):
        nn.init.normal_(fc.weight, mean=0, std=0.1)
        nn.init.constant_(fc.bias, self.B_ININT)
    
    def forward(self, x):
        # if self.do_bn: x = self.bn_input(x)
        for i in range(len(self.n_hidden) - 1):
            x = self.fcs[i](x)
            if self.do_bn: x = self.bns[i](x)
            if self.do_dp: x = self.dps[i](x)
            x = self.ACTIVATION(x)
        
        x = self.predict(x)
        return x

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def do_time():
    return now_time().replace(' ', '_').replace('-', '_').replace(':', '_')


def train(restore=False, module_params_fn=None, epoch=10000, cuda=True, save_dir='', zt=True, label='1',
          lr=0.01, is_lr_adjust=False, lr_adjust_step=1500, lr_weight_deacy=0.9,
          n_feature=34,
          HIDDEN_NODES=None,
          activation=nn.ReLU(),
          optimizer='Adam',
          now_head_dir = "2_rmcoref_data"):
    
    zt_hidden = [50, 30, 20]
    nt_hidden = [50, 30, 20]
    
    if HIDDEN_NODES is None:
        HIDDEN_NODES = [100, 80, 50]
        
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    
    train_csv_fn = os.path.join(r'G:\ztml\ztml\rdata', now_head_dir, 'train_30_train.csv')
    test_csv_fn = os.path.join(r'G:\ztml\ztml\rdata', now_head_dir, 'train_30_test.csv')
    train_pmdata_loader = load_pmdata(csv_file=train_csv_fn, shuffle=True, zt=zt, batch_size=588, output2=True)
    test_pmdata_loader = load_pmdata(csv_file=test_csv_fn, shuffle=True, zt=zt, batch_size=252, output2=True)
    
    if cuda:
        dnn = DNN2(n_feature=n_feature, n_hidden=HIDDEN_NODES, zt_hidden=zt_hidden, noptype_hidden=nt_hidden,
                   batch_normalize=True, dropout=True,
                   activation=nn.ReLU(), zt_act=nn.ReLU(), noptype_act=nn.Sigmoid()).cuda()
    else:
        dnn = DNN2(n_feature=n_feature, n_hidden=HIDDEN_NODES, zt_hidden=zt_hidden, noptype_hidden=nt_hidden,
                   batch_normalize=True, dropout=True,
                   activation=nn.ReLU(), zt_act=nn.ReLU(), noptype_act=nn.Sigmoid())
    
    # if cuda:
    #     dnn = DNN(n_feature=n_feature, n_hidden=HIDDEN_NODES, n_output=2,
    #               batch_normalize=True, dropout=True, activation=activation).cuda()
    # else:
    #     dnn = DNN(n_feature=n_feature, n_hidden=HIDDEN_NODES, n_output=2,
    #               batch_normalize=True, dropout=True, activation=activation)
    
    if restore:
        dnn.load_state_dict(torch.load(module_params_fn))
    print(dnn)
    # dnn = DNN(660, 1000, 4)
    # optimizer = torch.optim.Adam(dnn.parameters(), lr)  # weight_decay=0.01
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(dnn.parameters(), lr)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(dnn.parameters(), lr)
    else:
        raise ValueError("Only support Adam and SGD")
    loss_func = nn.MSELoss()
    floss_func = nn.MSELoss(reduction='none')
    # l1_crit = nn.L1Loss(size_average=False)
    # reg_loss = 0
    # for param in dnn.parameters():
    #     reg_loss += l1_crit(param, 100)
    tfn = os.path.join(save_dir, 'running_%s.log' % label)

    for ep in range(epoch):
        epoch = ep + 1
        if is_lr_adjust:
            if epoch % lr_adjust_step == 0:
                lr = lr * lr_weight_deacy
                adjust_learning_rate(optimizer, lr)

        for step, (b_x, b_y) in enumerate(train_pmdata_loader):
            # input_data = torch.DoubleTensor(b_x)
            # print(input_data)
            # print(input_data.shape)
            # print(dnn.fc1.weight.grad)
            if cuda:
                b_x, b_y = b_x.cuda(), b_y.cuda()
            else:
                b_x, b_y = b_x, b_y
            output = dnn(b_x.float())
            label_y = b_y.reshape(-1, 2)
            loss = loss_func(output, label_y.float())  # + 0.005 * reg_loss
            floss = floss_func(output, label_y.float())  # + 0.005 * reg_loss
            floss = np.mean(floss.cpu().data.numpy(), axis=1)

            dnn.eval()
            for _, (test_x, test_y) in enumerate(test_pmdata_loader):
                
                if cuda:
                    test_x, test_y = test_x.cuda(), test_y.cuda()

                test_output = dnn(test_x.float())
                testlabel_y = test_y.reshape(-1, 2)
                ftest_loss = floss_func(test_output, testlabel_y.float())  # + 0.005 * reg_loss
                ftest_loss = np.mean(ftest_loss.cpu().data.numpy(), axis=1)
            # print(output.cpu().data.numpy().shape, label_y.cpu().data.numpy().shape)
            
            dnn.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            txt_temple = 'Epoch: {0} | Step: {1} | train loss: {2:.18s} | test loss: {3:.18s}'.format(epoch,
                                                                                                      step,
                                                                                                      ' '.join(['%.6f' % nn for nn in floss.tolist()]),
                                                                                                      ' '.join(['%.6f' % nn for nn in ftest_loss.tolist()]))
            print(txt_temple)
            # now_step = step + epoch * math.ceil(TOTAL_LINE / BATCH_SIZE)
            save_module = True
            save_step = 1000


            if epoch == 0:
                write(tfn, txt_temple, 'w')
            else:
                # if now_step % SAVE_STEP == 0:
                write(tfn, txt_temple, 'a')
                
            if save_module:
                if epoch  % save_step == 0:
                    torch.save(dnn, os.path.join(save_dir, 'dnn_%d_%s.pkl' % (epoch, label)))
                    torch.save(dnn.state_dict(), os.path.join(save_dir, 'dnn_params_%d_%s.pkl' % (epoch, label)))


def ttest(test_csv_fn, mp_fn, save_dir='', output_fn='', n_feature=34, shuffle=False,
          HIDDEN_NODES=[100, 50, 50, 20], activation=nn.ReLU(), batch_size=252, has_t=None, n_output=1):
    # csv_fn = r'G:\ztml\ztml\data\clean_data_normalized.csv'
    # test_csv_fn = r'G:\ztml\ztml\data\test_data_from_normalized_data.csv'
    train_pmdata_loader = load_pmdata(csv_file=test_csv_fn, shuffle=shuffle, batch_size=batch_size)

    dnn = DNN(n_feature=n_feature, n_hidden=HIDDEN_NODES, n_output=n_output, batch_normalize=True, dropout=True, activation=activation)

    dnn.load_state_dict(torch.load(mp_fn))
    loss_func = nn.MSELoss()
    
    for step, (b_x, b_y) in enumerate(train_pmdata_loader):
        b_x, b_y = b_x, b_y
        dnn.eval()
        output = dnn(b_x.float())
        
        label_y = b_y.reshape(-1, 1)
        loss = loss_func(output, label_y.float())
        # print('loss: ', loss.data.numpy(), 'label_y: ', label_y.data.numpy(), 'predict_y: ', output.data.numpy())
        print('loss: ', loss.data.numpy())
        with open(os.path.join(save_dir, output_fn), 'w') as f:
            for i in range(len(label_y.data.numpy())):
                if has_t is not None:
                    f.write("%.7f     %.7f      %s\n" % (label_y.data.numpy()[i][0], output.data.numpy()[i][0], '  '.join([str(b_x.data.numpy()[i][m]) for m in has_t])))
                else:
                    f.write("%.7f     %.7f\n" % (label_y.data.numpy()[i][0], output.data.numpy()[i][0]))
                print(label_y.data.numpy()[i][0], '   ', output.data.numpy()[i][0])
        
        # if step == 10:
        #     break


def write(fn, content, mode='w'):
    with open(fn, mode) as f:
        f.write(content + '\n')



def run_train():
    # hid    den_layer = [500, 100, 50, 20]  # [100, 50, 20]  [100, 100, 50, 20]
    # epoch = 5000
    # # '3layer_100_Elu', '3layer_100_PRelu', '3layer_100_sigmod', '3layer_100_Tanh', '3layer_100', '4layer_100', '4layer_500'
    # label = '4layer_500'
    # activation = nn.ReLU()
    # optimizer = 'Adam'
    # # label = ["3layer_100", "3layer_100_sigmod", "3layer_100_Tanh", "3layer_100_sgd", "4layer_100", "4layer_500"]
    # # activation = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    # # hidden_layer= [[100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 100, 50, 20], [500, 100, 50, 20]]
    labels = ["3layer_100", "3layer_100_sigmod", "3layer_100_Tanh", "3layer_100_PReLU"]
    # activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.Tanh(), nn.Tanh(), nn.Tanh()]
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.PReLU()]
    hidden_layers= [[100, 50], [100, 50], [100, 50], [100, 50]]
    optimizers = ['Adam', 'Adam', 'Adam', 'Adam']

    # hidden_layer = [100, 50, 20]  # [100, 50, 20]  [100, 100, 50, 20]
    epoch = 8000

    for i in range(0, len(labels)):
        label = labels[i]
        activation = activations[i]
        hidden_layer = hidden_layers[i]
        optimizer = optimizers[i]
        
        train(cuda=True,
              epoch=epoch,
              save_dir=save_dir,
              label=label,
              n_feature=nfeature,
              HIDDEN_NODES=hidden_layer,
              activation=activation,
              optimizer=optimizer,
              now_head_dir="2_rmcoref_data",
              lr=0.01, is_lr_adjust=True, lr_adjust_step=2000, lr_weight_deacy=0.9)
        exit()
        

def run_test():
    
    # label = ["3layer_100", "3layer_100_sigmod", "3layer_100_Tanh", "3layer_100_sgd", "4layer_100", "4layer_500"]
    # activation = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    # hidden_layer= [[100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 100, 50, 20], [500, 100, 50, 20]]
    labels = ["3layer_100", "3layer_100_sigmod", "3layer_100_Tanh", "3layer_100_sgd", "4layer_100", "4layer_500"]
    # activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.Sigmoid(), nn.Sigmoid(), nn.Sigmoid()]
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    hidden_layers= [[100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 100, 50, 20], [500, 100, 50, 20]]

    for m in ['train_30_train.csv', 'train_30_test.csv']:
        for i in range(len(labels)):
            nlabel = labels[i]
            nactivation = activations[i]
            nhidden_layer = hidden_layers[i]
            
            # if i == 3:
            #     num = 12000
            # else:
            #     num = 5000
            num = 8000
            ttest(test_csv_fn=os.path.join(r'..\\rdata', m),
                  mp_fn=os.path.join(save_dir, 'dnn_params_%d_%s.pkl'%(num, nlabel)),
                  output_fn='result_%s_%s.out' % (m, nlabel), activation=nactivation,
                  save_dir=save_dir,  n_feature=nfeature, HIDDEN_NODES=nhidden_layer)
            
            
if __name__ == '__main__':
    save_dir = '2training_module_2output'
    nfeature = 11
    run_train()
    # run_test()
    # loss = nn.MSELoss(reduction='none')
    # input = torch.randn(3, 5, requires_grad=True)
    # target = torch.randn(3, 5)
    # output = loss(input, target)
    # print(output.data.numpy())
    # print(np.mean(output.data.numpy(), axis=0))
