# -*- coding: utf-8
import time
import os
import sys
import cv2
import warnings
import random
import mindspore
import x2ms_adapter
class global_variables(object):

    num_gpu=1
    local_rank=0
    iter_each_train_epoch=0#训练时按照tar图片数量设置epoch
    # iter_each_test_epoch=0
    #global variables
    seed=1234
    
    time_=time.localtime()
    time_=time.strftime('%Y-%m-%d-%H-%M',time_)
    log_root_path=''
    log_txt_path=''
    
    '''
    data process
    '''
    comment=''
    dataset='SHA'
    gcc2sha_ratio=1.0

    dataroot_GCC='/opt/data/common/GCC'
    dataroot_GCC_anno='/dataset/GCC_dataset/GCC'
    dataroot_SHA='/opt/data/common/shanghaitechA'


    weak_aug=['pass']
    strong_aug=['pass']
    blur_sigma=[0.1,2.0]
    contrast_severity=1
    brightness_severity=1

    tar_dataset='SHA'
    UCF50_val_choose=1
    train_crop_size=[0,0]
    tar_scale_factor=[1.0]
    tar_fix_scale_factor=1.0
    WE_dilate_kernel=1

    # only for GCC
    src_train_crop_size=[0,0]
    src_scale_factor=[1.0]
    split_method='random'# location camera
    split_txt_path='/dataset/GCC_dataset/description_and_split'
    level_regularization=[0,1,2,3,4,5,6,7,8]
    time_regularization=[0,25]
    weather_regularization=[0,1,2,3,4,5,6]
    count_range=[0,100000]
    radio_range=[0.,100000.]
    level_capacity=[10.,25.,50.,100.,300.,600.,1000.,2000.,4000.]

    # Paper: Learning from Synthetic Data for Crowd Counting in the Wild
    # Target Dataset level time weather count range ratio range
    # SHT A 4,5,6,7,8 6:00∼19:59 0,1,3,5,6 25∼4000 0.5∼1
    # SHT B 1,2,3,4,5 6:00∼19:59 0,1,5,6 10∼600 0.3∼1
    # UCF CC 50 5,6,7,8 8:00∼17:59 0,1,5,6 400∼4000 0.6∼1
    # UCF-QNRF 4,5,6,7,8 5:00∼20:59 0,1,5,6 400∼4000 0.6∼1
    # WorldExpo’10 2,3,4,5,6 6:00∼18:59 0,1,5,6 0∼1000 0∼1

    search_space=['grey','scale','perspective_transform']

    scale_range_max=1.0
    attributes_range={
    'grey':[0.,1.],
    'scale':[0.,1.],#default=1.
    'perspective_transform':[0.,45.],
    }
    searched_domain=[0,0,0]
    # Paper: Bi-lever Alignment for Cross domain crowd counting
    # SHT A:[0.785,0.779,0.272]
    # SHT B:[0.166,0.45,0.082]
    # QNRF B:[0.67,0.417,0.558]
    # UCF-CC-50:[0.855,0.231,0.109] 
    img_interpolate_mode='bilinear'# 'nearest' 'bicubic'
    mean_std=([0.5, 0.5, 0.5], [1.0, 1.0, 1.0])
    gt_factor=100.0
    sigma=4
    kernel_size=15

    num_workers=16
    train_batch_size   = 4
    test_batch_size   = 1

    #model
    model=''
    # model_pre_train=''
    model_resume_path=''
    model_for_load=''
    model_for_load_MT=''
    model_scale_factor=8# CSRNet 8  Res50 8 Hrnet  32  
    model_interpolate_mode='bilinear'


    # mean teacher
    teacher_alpha=0.99

    # domain loss
    gradient_scalar=-1.0
    domain_loss_trade_off=[1.0]
    gradient_scalar_1=-1.0
    dist_loss_trade_off=[1.0]

    # gass map
    den_map_thd_factor=0.1
    maxima_ksize=3
    p_label_gass_loss_trade_off=[1.0]

    # out map
    p_label_loss_trade_off=[1.0]

    #ada mix
    ####################### FYH OPTIONS FOR MIX PESUDO LABEL #######################
    concat_way = "only_density_map"  # ["only_density_map","only_feature_map","all"]
    ada_mix_loss_choose = "bce"  # ["bce","softmax"]
    mix_mode = 'soft'
    mix_grid=[8,8]
    mix_loss_trade_off = [1.0]
    p_label_mix_loss_trade_off=[1.0]
    mix_mask_dilate_kernel=1
    cnt_thd=100
    ################################################################################
    # loss
    loss_choose='loss1'
    den_loss_trade_off=[1.0]
    mask_den_choose=''

    ###BF_cls
    BF_grid=[8,8]
    BF_thd=[0.005,10000]
    BF_loss_choose='BCE'#softmax
    src_BF_cls_weight=[1.0]
    tar_BF_cls_weight=[1.0]

    # local_density_loss
    src_local_density_loss_trade_off=[1.0]
    tar_local_density_loss_trade_off=[1.0]
    local_density_grid=[8,8]

    mask_den_loss_trade_off=[0.0]


    # vision
    save_start_epochs=3
    vision_each_epoch=10
    vision_frequency=30

    #training set
    frozen_layers=['pass']
    secondary_layers=['pass']

    main_lr_init = 1e-5
    main_weight_decay=1e-4

    secondary_lr_init = 1e-5
    secondary_weight_decay=1e-4

    optimizer='Adam'
    train_mode='step'
    num_epochs   = 100
    decay_iter_freq=100
    decay_gamma=0.95

    epoch_stage=[0,50,100]
    # 0.95**100=0.006
    # 0.94**100=0.002
    # 0.93**100=7.051e-4
    # 0.92**100=2.392e-4
    # 0.91**100=8.019e-5
    # 0.90**100=2.656e-5

    def __init__(self, **kwself):
        for k, v in kwself.items():
            # print(k)
            # print('\n')
            if k=='--local_rank':
                k='local_rank'
            if not hasattr(self, k):
                print("Warning: opt has not attribut {}".format(k))
                import pdb
                pdb.set_trace()
                self.__dict__.update({k: v})
            tp = eval('type(self.{0})'.format(k))
            if tp == type(''):
                setattr(self, k, tp(v))
            elif tp == type([]):
                tp=eval('type(self.{0}[0])'.format(k))
                if tp==type('1'):
                    v=x2ms_adapter.tensor_api.split(v[1:-1], ',')
                    setattr(self, k, v)
                else:
                    setattr(self, k, eval(v))
            else:
                setattr(self, k, eval(v))

        if self.comment:
            self.log_root_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log/{}_{}_{}'.format(self.time_,self.model,self.comment))
            self.log_txt_path=os.path.join(os.path.dirname(self.log_root_path),'{}_{}_{}.txt'.format(self.time_,self.model,self.comment))
        else:
            self.log_root_path=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),'log/{}_{}'.format(self.time_,self.model))
            self.log_txt_path=os.path.join(os.path.dirname(self.log_root_path),'{}_{}.txt'.format(self.time_,self.model))
        
        if self.num_gpu>1:
            self.log_root_path+='_'+str(self.local_rank)
            self.log_txt_path=x2ms_adapter.tensor_api.split(self.log_txt_path, '.txt')[0]+'_'+str(self.local_rank)+'.txt'

        if 'pass' in self.frozen_layers:
            self.frozen_layers=[]

        if 'pass' in self.secondary_layers:
            self.secondary_layers=[]

        # if 'pass' in self.per_transform:
        #     self.per_transform=[]
        
        if 'GCC' not in self.dataset:
            self.split_method=None
            self.split_txt_path=None
            self.level_regularization=None
            self.time_regularization=None
            self.weather_regularization=None
            self.count_range=None
            self.radio_range=None
            self.level_capacity=None

        self.attributes_range['scale']=[max(self.src_train_crop_size[0]/1080,self.src_train_crop_size[1]/1920),self.scale_range_max]


        assert self.test_batch_size==1
        if self.num_gpu>1:
            if not os.path.exists(self.log_root_path) and self.local_rank==0:
                os.makedirs(self.log_root_path)
        elif not os.path.exists(self.log_root_path):
            os.makedirs(self.log_root_path)

if __name__ == '__main__':
    option = dict([x2ms_adapter.tensor_api.split(arg, '=') for arg in sys.argv[1:]])
    opt = global_variables(**option)
    for k, v in opt.__class__.__dict__.items():
        if not k.startswith('__'):
            print(k, getattr(opt, k))

    '''
    k_size=15
>>> sigma=4
>>> H = np.multiply(cv2.getGaussianKernel(k_size, sigma), (cv2.getGaussianKernel(k_size, sigma)).T)
>>> H
array([[0.00052666, 0.00079061, 0.00111494, 0.00147705, 0.00183822,
        0.0021491 , 0.00236032, 0.00243525, 0.00236032, 0.0021491 ,
        0.00183822, 0.00147705, 0.00111494, 0.00079061, 0.00052666],
       [0.00079061, 0.00118685, 0.00167372, 0.00221732, 0.0027595 ,
        0.00322618, 0.00354327, 0.00365574, 0.00354327, 0.00322618,
        0.0027595 , 0.00221732, 0.00167372, 0.00118685, 0.00079061],
       [0.00111494, 0.00167372, 0.00236032, 0.00312692, 0.00389152,
        0.00454964, 0.00499681, 0.00515542, 0.00499681, 0.00454964,
        0.00389152, 0.00312692, 0.00236032, 0.00167372, 0.00111494],
       [0.00147705, 0.00221732, 0.00312692, 0.0041425 , 0.00515542,
        0.0060273 , 0.00661969, 0.00682983, 0.00661969, 0.0060273 ,
        0.00515542, 0.0041425 , 0.00312692, 0.00221732, 0.00147705],
       [0.00183822, 0.0027595 , 0.00389152, 0.00515542, 0.00641603,
        0.0075011 , 0.00823834, 0.00849985, 0.00823834, 0.0075011 ,
        0.00641603, 0.00515542, 0.00389152, 0.0027595 , 0.00183822],
       [0.0021491 , 0.00322618, 0.00454964, 0.0060273 , 0.0075011 ,
        0.00876967, 0.0096316 , 0.00993734, 0.0096316 , 0.00876967,
        0.0075011 , 0.0060273 , 0.00454964, 0.00322618, 0.0021491 ],
       [0.00236032, 0.00354327, 0.00499681, 0.00661969, 0.00823834,
        0.0096316 , 0.01057824, 0.01091403, 0.01057824, 0.0096316 ,
        0.00823834, 0.00661969, 0.00499681, 0.00354327, 0.00236032],
       [0.00243525, 0.00365574, 0.00515542, 0.00682983, 0.00849985,
        0.00993734, 0.01091403, 0.01126048, 0.01091403, 0.00993734,
        0.00849985, 0.00682983, 0.00515542, 0.00365574, 0.00243525],
       [0.00236032, 0.00354327, 0.00499681, 0.00661969, 0.00823834,
        0.0096316 , 0.01057824, 0.01091403, 0.01057824, 0.0096316 ,
        0.00823834, 0.00661969, 0.00499681, 0.00354327, 0.00236032],
       [0.0021491 , 0.00322618, 0.00454964, 0.0060273 , 0.0075011 ,
        0.00876967, 0.0096316 , 0.00993734, 0.0096316 , 0.00876967,
        0.0075011 , 0.0060273 , 0.00454964, 0.00322618, 0.0021491 ],
       [0.00183822, 0.0027595 , 0.00389152, 0.00515542, 0.00641603,
        0.0075011 , 0.00823834, 0.00849985, 0.00823834, 0.0075011 ,
        0.00641603, 0.00515542, 0.00389152, 0.0027595 , 0.00183822],
       [0.00147705, 0.00221732, 0.00312692, 0.0041425 , 0.00515542,
        0.0060273 , 0.00661969, 0.00682983, 0.00661969, 0.0060273 ,
        0.00515542, 0.0041425 , 0.00312692, 0.00221732, 0.00147705],
       [0.00111494, 0.00167372, 0.00236032, 0.00312692, 0.00389152,
        0.00454964, 0.00499681, 0.00515542, 0.00499681, 0.00454964,
        0.00389152, 0.00312692, 0.00236032, 0.00167372, 0.00111494],
       [0.00079061, 0.00118685, 0.00167372, 0.00221732, 0.0027595 ,
        0.00322618, 0.00354327, 0.00365574, 0.00354327, 0.00322618,
        0.0027595 , 0.00221732, 0.00167372, 0.00118685, 0.00079061],
       [0.00052666, 0.00079061, 0.00111494, 0.00147705, 0.00183822,
        0.0021491 , 0.00236032, 0.00243525, 0.00236032, 0.0021491 ,
        0.00183822, 0.00147705, 0.00111494, 0.00079061, 0.00052666]])
    '''
