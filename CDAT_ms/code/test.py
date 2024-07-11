#coding=utf-8
import os
import sys
import torch
import mindspore
from pathlib import Path
sys.path.append(os.path.abspath((__file__)))
sys.path.append(os.path.abspath(Path(__file__).resolve().parent.parent))
import x2ms_adapter
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
from options_train import global_variables as opt_conf
from tools.terminal_log import create_log_file_terminal,save_opt,create_exp_dir
from tools.progress.bar import Bar
import tools.godblessdbg as godblessdbg
from tools.log import AverageMeter
from tools.progress.bar import Bar
import time
import numpy as np
import random
import glob
import utils
from utils import earlystop,lr_decay,AverageMeter_array,AverageMeter_array_mask
import scipy.io as io
import shutil
from matplotlib import pyplot as plt
import cv2
import json
import math
from vision import save_keypoints_img

def draw_loss_line(loss_list,index_map,save_dir):
    loss=np.array(loss_list)
    x=range(1,len(loss_list[0])+1)
    plt.figure()
    plt.xlabel("Epoch",fontsize=25) 
    plt.ylabel("MAE",fontsize=25)
    j=0
    for i in index_map:
        plt.plot(x,loss[j],label=i,markersize=10,linewidth=2.5) 
        j+=1
    plt.yticks(size = 22)
    plt.xticks(size = 22)
    plt.grid(ls='--')
    plt.legend(fontsize=17)
    plt.savefig(os.path.join(save_dir,'mae.pdf'),bbox_inches = 'tight')
    plt.figure() 
    plt.close('all')



def check_mem(cuda_device):
    devices_info = os.popen('"/usr/bin/nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader').read().strip().split("\n")
    total, used = devices_info[int(cuda_device)].split(',')
    return total,used

def occumpy_mem(cuda_device):
    total, used = check_mem(cuda_device)
    total = int(total)
    used = int(used)
    max_mem = int(total * 0.9)
    block_mem = max_mem - used
    #float占用四个字节，所以会占用所有未使用内存
    x = x2ms_adapter.FloatTensor(256,1024,block_mem)
    del x


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    mindspore.set_seed(seed)  # if you are using multi-GPU.


def test(epoch,split,test_vision,model,test_loader,opt,log):
    model.set_train(False)
    results = {}
    batch_time = AverageMeter()
    data_time = AverageMeter()
    mae = AverageMeter()
    mse = AverageMeter()
    bar = Bar('Testing', max=len(test_loader))
    end = time.time()

    for step, data in enumerate(test_loader):
        data_time.update(time.time() - end)
        end = time.time()
        res=model.inference(data,0,epoch,0,step,test_vision,len(test_loader),log)
        batch_time.update(time.time() - end)
        end = time.time()
        str_plus=''
        for k,v in res.items():
            str_plus+=' | {key:}:{value:.4f}'.format(key=k,value=v.asnumpy())
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:}'.format(
            batch=step + 1,
            size=len(test_loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
        )+str_plus
        bar.next()

    bar.finish()
    return res


if __name__ == '__main__':
    # print(sys.argv)
    print(sys.argv)
    option = dict([arg.split('=') for arg in sys.argv[1:]])
    # t=['./code/test.py', 'model=CDAT', 'comment=SHA_results_vision', 'dataset=GCC_weak_strong_flip_just_weak_fix', 'tar_dataset=SHA_weak_strong_flip_just_weak', 'src_scale_factor=[1.0]', 'tar_scale_factor=[1.0]', 'tar_fix_scale_factor=1.0', 'src_train_crop_size=[576,768]', 'train_crop_size=[576,768]', 'search_space=[grey,scale,perspective_transform]', 'searched_domain=[0.785,0.779,0.272]', 'model_for_load_MT=./code/pre_models/SHA_92.3.pth', 'dataroot_SHA=./data/SHA', 'vision_each_epoch=300', 'num_workers=1']
    # option = dict([arg.split('=') for arg in t[1:]])
    opt=opt_conf(**option)
    # occumpy_mem(0)
    seed_torch(opt.seed)
    log_output=create_log_file_terminal(opt.log_txt_path)
    save_opt(opt,opt.log_txt_path)
    scripts_to_save=glob.glob('./code/*')
    create_exp_dir(opt.log_root_path,scripts_to_save)
    log_output.info(os.path.realpath(__file__))
    log_output.info(sys.argv)
    
    from importlib import import_module
    net=import_module('models.{}'.format(opt.model))
    exec('model=net.{}(opt)'.format(opt.model))
    if opt.model_for_load_MT:
        checkpoint=torch.load(opt.model_for_load_MT,map_location=torch.device('cpu'))
        model_state_disk=checkpoint['net']
        dict1=model.parameters_and_names()
        for key, parameter in dict1:                # 获取ms模型的参数名和数值
            if 'embedding_table' in key:                                    # 参数名中的embedding_table替换为weight
                key = key.replace('embedding_table', 'weight')
            elif 'gamma' in key:
                key = key.replace('gamma', 'weight')                        #// 参数名中的gamma替换为weight
            elif 'beta' in key:
                key = key.replace('beta', 'bias')                           #// 参数名中的beta替换为bias
            elif 'moving_mean' in key:
                key = key.replace('moving_mean', 'running_mean')
            elif 'moving_variance' in key:
                key = key.replace('moving_variance', 'running_var')
            elif 'weight' in key:
                temp = mindspore.Tensor(model_state_disk.get(key).detach().numpy())
                if len(parameter.shape)>len(temp.shape):
                    temp = temp.unsqueeze(len(temp.shape))
                    parameter.set_data(temp)
                    continue
                #依据key获取pytorch中相应参数的数值并赋给mindspore当前参数parameter，上面替换参数名就是为了get(key)的时候不会找不到
            temp=mindspore.Tensor(model_state_disk.get(key).detach().numpy())
            parameter.set_data(temp)
            
    # dataloader=import_module('datasets.{}'.format(opt.dataset))
    # exec('train_loader,train_data,_,_=dataloader.{}(opt)'.format(opt.dataset))
    # if opt.dataset!='GCC_DR':
    #     train_data.mix_domain_set(opt.searched_domain)
    dataloader=import_module('datasets.{}'.format(opt.tar_dataset))
    exec('_,_,test_loader_tar,test_data_tar=dataloader.{}(opt)'.format(opt.tar_dataset))

    sample_num=min(len(test_loader_tar),opt.vision_each_epoch)
    test_vision_tar=random.sample(list(range(len(test_loader_tar))), sample_num)
    test_vision_tar.append(0)

    # assert opt.model_for_load
    # checkpoint=torch.load(opt.model_for_load)
    # model.load_state_dict(checkpoint['net'])
    model.next_epoch()
    results=test(0,'test', test_vision_tar, model, test_loader_tar,opt,log_output)
    log_output.info('Results in {}'.format(opt.tar_dataset))
    for k,v in results.items():
        log_output.info('{}: {}'.format(k,v))
    log_output.info(godblessdbg.end)
    log_output.info('log location {0}'.format(opt.log_root_path))
