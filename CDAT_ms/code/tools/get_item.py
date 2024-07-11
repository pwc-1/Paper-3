# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 09:13:09 2019

@author: HP
"""


import os
import numpy as np
from tools.pickle_use import pickle_load
import mindspore
import x2ms_adapter
class my_dataset:
    
    def __init__(self,data_pkl,target_pkl):
        
        self.data_path=data_pkl
        self.target_path=target_pkl
        self.pickle_files=os.listdir(self.data_path)
        self.targets=x2ms_adapter.from_numpy(pickle_load(self.target_path).astype(np.int64))
        
    def __getitem__(self,index):
        
        pickle_path=os.path.join(self.data_path,
            'sample{}.pickle'.format(index))
        x2ms_adapter.from_numpy(pickle_load(pickle_path).astype(np.float32))
        return x2ms_adapter.from_numpy(pickle_load(pickle_path).astype(np.float32)),self.targets[index]
    def __len__(self):
        return len(self.pickle_files)
        
        
        
    