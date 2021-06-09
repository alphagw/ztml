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
import torch
import torch.nn as nn
from ztml.read_data import load_pmdata
from matfleet.utilities import now_time


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

def train(restore=False, module_params_fn=None, lr=0.01, epoch=10000, cuda=True, save_dir=''):
    csv_fn = r'G:\ztml\ztml\data\clean_data_normalized.csv'
    train_pmdata_loader = load_pmdata(csv_file=csv_fn, shuffle=True, zt=True, batch_size=840)

    n_feature = 34
    
    HIDDEN_NODES = [100, 50, 50, 20]
    
    if cuda:
        dnn = DNN(n_feature=n_feature, n_hidden=HIDDEN_NODES, n_output=1, batch_normalize=True, dropout=True).cuda()
    else:
        dnn = DNN(n_feature=n_feature, n_hidden=HIDDEN_NODES, n_output=1, batch_normalize=True, dropout=True)
    
    if restore:
        dnn.load_state_dict(torch.load(module_params_fn))
    print(dnn)
    # dnn = DNN(660, 1000, 4)
    optimizer = torch.optim.Adam(dnn.parameters(), lr)  # weight_decay=0.01
    loss_func = nn.MSELoss()
    # l1_crit = nn.L1Loss(size_average=False)
    # reg_loss = 0
    # for param in dnn.parameters():
    #     reg_loss += l1_crit(param, 100)
    tfn = os.path.join(save_dir, 'running_%s.log' % do_time())

    for ep in range(epoch):
        epoch = ep + 1
        # if epoch % 1500 == 0:
        #     lr = lr * 0.5
        #     adjust_learning_rate(optimizer, lr)

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
            
            label_y = b_y.reshape(-1, 1)
            loss = loss_func(output, label_y.float())  # + 0.005 * reg_loss
            # print(output.cpu().data.numpy().shape, label_y.cpu().data.numpy().shape)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            txt_temple = 'Epoch: {0} | Step: {1} | train loss: {2:.4f}'.format(epoch, step, loss.cpu().data.numpy())
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
                if os.path.isdir(save_dir):
                    pass
                else:
                    os.mkdir(save_dir)

                if epoch  % save_step == 0:
                    torch.save(dnn, os.path.join(save_dir, 'dnn_%d_%s.pkl' % (epoch, do_time())))
                    torch.save(dnn.state_dict(), os.path.join(save_dir, 'dnn_params_%d_%s.pkl' % (epoch, do_time())))


def ttest(mp_fn=r'training_module/dnn_params_10000.pkl', save_dir=''):
    csv_fn = r'G:\ztml\ztml\data\clean_data_normalized.csv'
    train_pmdata_loader = load_pmdata(csv_file=csv_fn, shuffle=True)
    HIDDEN_NODES = [100, 50, 50, 20]

    dnn = DNN(n_feature=34, n_hidden=HIDDEN_NODES, n_output=1, batch_normalize=True, dropout=True)

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
        with open(os.path.join(save_dir, "results.out"), 'w') as f:
            for i in range(len(label_y.data.numpy())):
                f.write("%.7f     %.7f\n" % (label_y.data.numpy()[i][0], output.data.numpy()[i][0]) )
                print(label_y.data.numpy()[i][0], '   ', output.data.numpy()[i][0])
        
        # if step == 10:
        #     break


def write(fn, content, mode='w'):
    with open(fn, mode) as f:
        f.write(content + '\n')


if __name__ == '__main__':
    save_dir = 'training_module'
    # train(cuda=False, save_dir=save_dir)
    ttest(mp_fn=os.path.join(save_dir, 'dnn_params_1000_2021_06_09_16_53_29.694827.pkl'), save_dir=save_dir)