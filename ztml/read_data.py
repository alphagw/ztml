#   coding:utf-8
#   This file is part of Alkemiems.
#
#   Alkemiems is free software: you can redistribute it and/or modify
#   it under the terms of the MIT License.

__author__ = 'Guanjie Wang'
__version__ = 1.0
__maintainer__ = 'Guanjie Wang'
__email__ = "gjwang@buaa.edu.cn"
__date__ = '2021/05/24 15:54:00'

from torch.utils.data.dataset import Dataset
import pandas as pd
import torch.utils.data as Data


class PmData(Dataset):
    
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file).values
        self.train_data = self.data[:, :-1]
        self.label = self.data[:, -1]
        
    def __getitem__(self, item):
        # train_val, label = self.data[item], self.targets[item]
        return self.train_data[item, :], self.label[item]

    def __len__(self):
        return len(self.data)


def load_pmdata(csv_file, batch_size=840, shuffle=True):
    pmdata = PmData(csv_file)
    train_loader = Data.DataLoader(dataset=pmdata, batch_size=batch_size, shuffle=shuffle)
    return train_loader


if __name__ == '__main__':
    
    root_path = r'G:\ztml\ztml\data\clean_data.csv'
    # a = PmData(root=root_path, ele='Sb')
    a = PmData(root_path)
    print(a[1])
    # print(a[10894])
    # print(len(a))