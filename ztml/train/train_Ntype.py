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


def train(restore=False, module_params_fn=None, lr=0.01, epoch=10000, cuda=True, save_dir='', zt=False, label='1',
          n_feature = 34, HIDDEN_NODES = [100, 50, 50, 20], activation=nn.ReLU(), optimizer='Adam', n_output=1):
    
    if os.path.isdir(save_dir):
        pass
    else:
        os.mkdir(save_dir)
    
    train_csv_fn = r'..\\data\\train_30_train.csv'
    test_csv_fn = r'..\\data\\train_30_test.csv'
    train_pmdata_loader = load_pmdata(csv_file=train_csv_fn, shuffle=True, zt=zt, batch_size=588)
    test_pmdata_loader = load_pmdata(csv_file=test_csv_fn, shuffle=True, zt=zt, batch_size=252)
    
    if cuda:
        dnn = DNN(n_feature=n_feature, n_hidden=HIDDEN_NODES, n_output=n_output, batch_normalize=True, dropout=True, activation=activation).cuda()
    else:
        dnn = DNN(n_feature=n_feature, n_hidden=HIDDEN_NODES, n_output=n_output, batch_normalize=True, dropout=True, activation=activation)
    
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
    # loss_func = nn.Softmax()
    loss_func = nn.MSELoss()
    # l1_crit = nn.L1Loss(size_average=False)
    # reg_loss = 0
    # for param in dnn.parameters():
    #     reg_loss += l1_crit(param, 100)
    tfn = os.path.join(save_dir, 'running_%s.log' % label)

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
            # for i in range(output.shape[0]):
            #     if output[i] < 0:
            #         output[i] = 0
            #     else:
            #         output[i] = 1
            label_y = b_y.reshape(-1, 1)
            loss = loss_func(output, label_y.float())  # + 0.005 * reg_loss

            dnn.eval()
            for _, (test_x, test_y) in enumerate(test_pmdata_loader):
                
                if cuda:
                    test_x, test_y = test_x.cuda(), test_y.cuda()

                test_output = dnn(test_x.float())
                testlabel_y = test_y.reshape(-1, 1)
                test_loss = loss_func(test_output, testlabel_y.float())  # + 0.005 * reg_loss
            # print(output.cpu().data.numpy().shape, label_y.cpu().data.numpy().shape)
            
            dnn.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            txt_temple = 'Epoch: {0} | Step: {1} | train loss: {2:.6f} | test loss: {3:.6f}'.format(epoch, step, loss.cpu().data.numpy(), test_loss.cpu().data.numpy())
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


def ntype_ttest(test_csv_fn, mp_fn, save_dir='', output_fn='', n_feature=34, shuffle=False,
          HIDDEN_NODES=[100, 50, 50, 20], activation=nn.ReLU(), batch_size=252, zt=False, n_output=1, has_t=None):
    # csv_fn = r'G:\ztml\ztml\data\clean_data_normalized.csv'
    # test_csv_fn = r'G:\ztml\ztml\data\test_data_from_normalized_data.csv'
    train_pmdata_loader = load_pmdata(csv_file=test_csv_fn, shuffle=shuffle, batch_size=batch_size, zt=zt)

    dnn = DNN(n_feature=n_feature, n_hidden=HIDDEN_NODES, n_output=n_output, batch_normalize=True, dropout=True,
              activation=activation)

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
                if has_t:
                    if has_t is not None:
                        f.write("%.7f     %.7f      %s\n" % (label_y.data.numpy()[i][0], output.data.numpy()[i][0],
                                                             '  '.join([str(b_x.data.numpy()[i][m]) for m in has_t])))
                    else:
                        f.write("%.7f     %.7f\n" % (label_y.data.numpy()[i][0], output.data.numpy()[i][0]))
                else:
                    f.write("%.7f     %.7f\n" % (label_y.data.numpy()[i][0], output.data.numpy()[i][0]))
                print(label_y.data.numpy()[i][0], '   ', output.data.numpy()[i][0])
        
        # if step == 10:
        #     break


def write(fn, content, mode='w'):
    with open(fn, mode) as f:
        f.write(content + '\n')



def run_train():
    
    labels = ["3layer_100", "3layer_100_sigmod", "3layer_100_Tanh", "3layer_100_sgd", "4layer_100", "4layer_500"]
    activations = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    hidden_layers= [[100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 100, 50, 20], [500, 100, 50, 20]]
    optimizers = ['Adam', 'Adam', 'Adam', 'SGD', 'Adam', 'Adam']

    # hidden_layer = [100, 50, 20]  # [100, 50, 20]  [100, 100, 50, 20]
    epoch = 5000
    #'3layer_100_Elu', '3layer_100_PRelu', '3layer_100_sigmod', '3layer_100_Tanh', '3layer_100', '4layer_100', '4layer_500'
    # label = '3layer_100'
    # activation = nn.ReLU()
    # optimizer = 'Adam'
    
    for i in range(len(labels)):
        label = labels[i]
        activation = activations[i]
        hidden_layer = hidden_layers[i]
        optimizer = optimizers[i]
        
        train(cuda=True,
              epoch=epoch,
              save_dir=save_dir,
              label=label,
              zt=False,
              n_feature=nfeature,
              HIDDEN_NODES=hidden_layer,
              activation=activation,
              optimizer=optimizer,
              n_output=1)


def run_test():
    label = ['N_type_4layer_500', "3layer_100", "3layer_100_sigmod", "3layer_100_Tanh", "3layer_100_sgd", "4layer_100", "4layer_500"]
    activation = [nn.ReLU(), nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.ReLU(), nn.ReLU()]
    hidden_layer= [[100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 50, 20], [100, 100, 50, 20], [500, 100, 50, 20]]

    for m in ['train_30_train.csv', 'train_30_test.csv', 'valid_40.csv']:
        
        nlabel = label[0]
        nactivation = activation[0]
        nhidden_layer = hidden_layer[-1]

        ntype_ttest(test_csv_fn=os.path.join(r'..\\data', m),
              mp_fn=os.path.join(save_dir, 'dnn_params_5000_%s.pkl'%nlabel),
              output_fn='result_%s_%s.out' % (m, nlabel), activation=nactivation,
              save_dir=save_dir,  n_feature=nfeature, HIDDEN_NODES=nhidden_layer, zt=False, n_output=1)
        break
            
if __name__ == '__main__':
    save_dir = 'ntype_training_module'
    nfeature = 28
    run_train()
    # run_test()
