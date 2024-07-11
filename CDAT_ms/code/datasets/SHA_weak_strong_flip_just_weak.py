import os
import sys
import numpy as np
from torch.utils import data
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from scipy import sparse
import scipy
import torch.nn.functional as F
import math
import cv2
from tools.progress.bar import Bar
import pickle
import json
import pdb
import torch.nn as nn
from torch.autograd import Variable
import math
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from importlib import import_module
# https://github.com/bethgelab/imagecorruptions

# from imagecorruptions import corrupt
def rad(x):
    return x*np.pi/180


def perspective_transformation(img, transform,ann_point):
    h,w=img.shape[0:2]
    anglex=0
    angley = 0
    anglez = 0
    fov = 42

    for i in transform.split(' '):
        if i.split('_')[0]=='x':
            anglex=int(i.split('_')[1])
        elif i.split('_')[0]=='y':
            angley=int(i.split('_')[1])
        elif i.split('_')[0]=='z':
            anglez=int(i.split('_')[1])

    if anglex==0 and angley==0 and anglez==0:
        return img,ann_point

    #镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z=np.sqrt(w**2 + h**2)/2/np.tan(rad(fov/2))
    #齐次变换矩阵
    rx = np.array([[1,                  0,                          0,                          0],
                   [0,                  np.cos(rad(anglex)),        -np.sin(rad(anglex)),       0],
                   [0,                 -np.sin(rad(anglex)),        np.cos(rad(anglex)),        0,],
                   [0,                  0,                          0,                          1]], np.float32)
 
    ry = np.array([[np.cos(rad(angley)), 0,                         np.sin(rad(angley)),       0],
                   [0,                   1,                         0,                          0],
                   [-np.sin(rad(angley)),0,                         np.cos(rad(angley)),        0,],
                   [0,                   0,                         0,                          1]], np.float32)
 
    rz = np.array([[np.cos(rad(anglez)), np.sin(rad(anglez)),      0,                          0],
                   [-np.sin(rad(anglez)), np.cos(rad(anglez)),      0,                          0],
                   [0,                  0,                          1,                          0],
                   [0,                  0,                          0,                          1]], np.float32)
 
    r = rx.dot(ry).dot(rz)
 
    #四对点的生成
    pcenter = np.array([h/2, w/2, 0, 0], np.float32)
    
    p1 = np.array([0,0,  0,0], np.float32) - pcenter
    p2 = np.array([w,0,  0,0], np.float32) - pcenter
    p3 = np.array([0,h,  0,0], np.float32) - pcenter
    p4 = np.array([w,h,  0,0], np.float32) - pcenter
    
    dst1 = r.dot(p1)
    dst2 = r.dot(p2)
    dst3 = r.dot(p3)
    dst4 = r.dot(p4)
 
    list_dst = [dst1, dst2, dst3, dst4]
 
    org = np.array([[0,0],
                    [w,0],
                    [0,h],
                    [w,h]], np.float32)
    
    dst = np.zeros((4,2), np.float32)
 
    #投影至成像平面
    for i in range(4):
        dst[i,0] = list_dst[i][0]*z/(z-list_dst[i][2]) + pcenter[0]
        dst[i,1] = list_dst[i][1]*z/(z-list_dst[i][2]) + pcenter[1]
 
    warpR = cv2.getPerspectiveTransform(org, dst)
    result = cv2.warpPerspective(img, warpR, (w,h))


    x0=max(dst[0,0],dst[2,0],0)
    x1=min(dst[1,0],dst[3,0],w)

    y0=max(dst[0,1],dst[1,1],0)
    y1=min(dst[2,1],dst[3,1],h)

    tmp=result[int(y0):int(y1),int(x0):int(x1)]
    resize = cv2.resize(tmp, (w,h), interpolation = cv2.INTER_LINEAR)

    h,w=resize.shape[:2]
    scale_h=h/tmp.shape[0]
    scale_w=w/tmp.shape[1]


    p=np.zeros([4,len(ann_point)], np.float32) - pcenter[:,np.newaxis]
    ann_point=np.array(ann_point)
    p[0]+=ann_point[:,1]
    p[1]+=ann_point[:,0]

    ann_dst=r.dot(p).T# number*4
    points_dst=np.zeros((len(ann_dst),2), np.float32)
    points_dst[:,0] = ann_dst[:,0]*z/(z-ann_dst[:,2]) + pcenter[0]
    points_dst[:,1] = ann_dst[:,1]*z/(z-ann_dst[:,2]) + pcenter[1]

    list_points_dst=[]
    for det_p in points_dst:
        if det_p[0]>=x0 and det_p[0]<x1 and det_p[1]>=y0 and det_p[1]<y1:
            tmp=[(det_p[1]-y0)*scale_h,(det_p[0]-x0)*scale_w]
            list_points_dst.append(tmp)

    return resize,list_points_dst



class Gaussian(nn.Module):
    def __init__(self, in_channels, sigmalist, kernel_size=64, stride=1, padding=0, froze=True):
        super(Gaussian, self).__init__()
        out_channels = len(sigmalist) * in_channels
        # gaussian kernel
        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma: math.exp(-(x - mu) ** 2 / float(2 * sigma ** 2)))
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            gauss = torch.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss /= gauss.sum()
            _1D_window = gauss.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = Variable(_2D_window.expand(in_channels, 1, kernel_size, kernel_size).contiguous())
            windows.append(window)
        kernels = torch.stack(windows)
        kernels = kernels.permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)
        
        self.gkernel = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.gkernel.weight = torch.nn.Parameter(weight)
        # # Froze this Gaussian net
        if froze: self.frozePara()

    def forward(self, dotmaps):
        gaussianmaps = self.gkernel(dotmaps)
        return gaussianmaps
    
    def frozePara(self):
        for para in self.parameters():
            para.requires_grad = False

class Gaussianlayer(nn.Module):
    def __init__(self, sigma=None, kernel_size=15):
        super(Gaussianlayer, self).__init__()
        if sigma == None:
            sigma = [4]
            # sigma = [kernel_size * 0.3]
        self.gaussian = Gaussian(1, sigma, kernel_size=kernel_size, padding=kernel_size // 2, froze=True)

    def forward(self, dotmaps):
        denmaps = self.gaussian(dotmaps)
        return denmaps

class gen_den_map(nn.Module):
    def __init__(self,sigma,kernel_size):
        super(gen_den_map, self).__init__()
        self.gs = Gaussianlayer(sigma=[sigma],kernel_size=kernel_size)
    
    def forward(self, dot_map):
        # b*1*h*w
        gt_map = self.gs(dot_map)
        return gt_map

def sha_sort_img(filename):
    return int(filename.split('_')[1].split('.')[0])

def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    factor=torch.empty(1).uniform_(-c, c)
    x=torch.clamp(x+c,0,1)
    return x

def gaussian_blur(img,sigma=[0.1,2.0]):
    sigma=torch.empty(1).uniform_(sigma[0], sigma[1]).item()
    kernel_size=int(8*sigma+1)
    if kernel_size==1:
        return img
    if kernel_size%2==0:
        kernel_size+=1
    gass_kernel=gen_den_map(sigma, kernel_size)
    img=img.unsqueeze(0)
    img_cloned=img.clone()
    img[:,0:1]=gass_kernel(img_cloned[:,0:1])
    img[:,1:2]=gass_kernel(img_cloned[:,1:2])
    img[:,2:3]=gass_kernel(img_cloned[:,2:3])
    img=img.squeeze()
    return img


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    factor=torch.empty(1).uniform_(1-c, 1+c)
    means = torch.mean(x, [1,2], keepdim=True)
    x=torch.clamp((x-means)*factor+means,0,1)
    return x

class SHA_dataloader(data.Dataset):
    def __init__(self,split,opt):
        self.split=split
        self.opt=opt
        self.mean=opt.mean_std[0]
        self.std=opt.mean_std[1]
        self.scale_factor=opt.tar_scale_factor
        self.fix_scale_factor=opt.tar_fix_scale_factor
        self.weak_aug=self.opt.weak_aug
        self.strong_aug=self.opt.strong_aug
        self.crop_size=self.opt.train_crop_size

        self.gen_map=gen_den_map(opt.sigma, opt.kernel_size)

        if self.split=='train':
            self.img_root=os.path.join(opt.dataroot_SHA,'train','pre_load_img')
            self.ann_root=os.path.join(opt.dataroot_SHA,'train','dot_map_03_535_3seg')
        else:
            self.img_root=os.path.join(opt.dataroot_SHA,'val','pre_load_img')
            self.ann_root=os.path.join(opt.dataroot_SHA,'val','dot_map_03_535_3seg')

        imgfiles=[filename for filename in os.listdir(self.img_root) \
                       if os.path.isfile(os.path.join(self.img_root,filename)) and filename.split('.')[1] in ['jpg','png']]
        imgfiles.sort(key=sha_sort_img)

        if not os.path.exists(os.path.join(opt.dataroot_SHA,'{}_dot_pre_load_img.json'.format(self.split))):
            self.points_data={}
            for img_file in imgfiles:
                self.points_data[img_file]=[]
                ann_path1=os.path.join(self.ann_root,img_file.split('.')[0]+'_seg1.png')
                ann_path2=os.path.join(self.ann_root,img_file.split('.')[0]+'_seg2.png')
                ann_path3=os.path.join(self.ann_root,img_file.split('.')[0]+'_seg3.png')
                den1=cv2.imread(ann_path1,0)
                den2=cv2.imread(ann_path2,0)
                den3=cv2.imread(ann_path3,0)
                den=den1+den2+den3
                cnt=np.sum(den)

                den=torch.from_numpy(den).float()
                h,w=den.size()

                x_grid=torch.tensor(list(range(w)),dtype=torch.int16).unsqueeze(0)
                x_grid=x_grid.expand(h,w)

                y_grid=torch.tensor(list(range(h)),dtype=torch.int16).unsqueeze(1)
                y_grid=y_grid.expand(h,w)

                x=x_grid[den>0].tolist()
                y=y_grid[den>0].tolist()

                for i in range(len(y)):
                    assert den[y[i],x[i]]>=1
                    for j in range(int(den[y[i],x[i]])):
                        self.points_data[img_file].append([y[i],x[i]])
                assert len(self.points_data[img_file])==cnt
            res=json.dumps(self.points_data)
            json_save=os.path.join(os.path.join(opt.dataroot_SHA,'{}_dot_pre_load_img.json'.format(self.split)))
            with open(json_save, 'w') as f:  # ‘a’表示在不删除原数据的情况下在文件末尾写入数据
                f.write(res)
                f.close()
        else:
            with open(os.path.join(os.path.join(opt.dataroot_SHA,'{}_dot_pre_load_img.json'.format(self.split))),'r') as f:
                self.points_data=json.load(f)
        self.imgfiles=imgfiles

        print('SHA {}'.format(len(self.imgfiles)))

    def __getitem__(self,index):
        img_file=self.imgfiles[index]
        img_path=os.path.join(self.img_root,img_file)
        ann_point=self.points_data[img_file]
        img=cv2.imread(img_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_strong=np.copy(img)

        #strong aug
        # if 'jpeg_compression' in self.strong_aug and random.random() < 0.5 and 'train' in self.split:
        #     img_strong = corrupt(img_strong, corruption_name='jpeg_compression', severity=1)

        if 'grey' in self.strong_aug and random.random() < 0.5 and 'train' in self.split:
            img_strong=cv2.cvtColor(img_strong, cv2.COLOR_RGB2GRAY)
            img_strong=img_strong[:,:,np.newaxis]
            img_strong=np.concatenate((img_strong,img_strong,img_strong),axis=-1)

        #weak
        # if 'jpeg_compression' in self.weak_aug and random.random() < 0.5 and 'train' in self.split:
        #     img = corrupt(img, corruption_name='jpeg_compression', severity=1)


        if 'grey' in self.weak_aug and random.random() < 0.5 and 'train' in self.split:
            img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img=img[:,:,np.newaxis]
            img=np.concatenate((img,img,img),axis=-1)

        

        img=img.transpose([2,0,1])# c h w
        img=torch.from_numpy(img).float()
        img=img.div(255)


        img_strong=img_strong.transpose([2,0,1])# c h w
        img_strong=torch.from_numpy(img_strong).float()
        img_strong=img_strong.div(255)



        src_h,src_w=img.size(1),img.size(2)

        scale_factor=1.0
        if 'train' in self.split and 'scale' in self.weak_aug:
            scale_factor=self.scale_factor[np.random.randint(0,len(self.scale_factor),1).tolist()[0]]
        elif not 'train' in self.split:
            scale_factor=self.fix_scale_factor

        h=int(src_h*scale_factor)
        h=h if h%self.opt.model_scale_factor==0 else h+self.opt.model_scale_factor-h%self.opt.model_scale_factor
        w=int(src_w*scale_factor)
        w=w if w%self.opt.model_scale_factor==0 else w+self.opt.model_scale_factor-w%self.opt.model_scale_factor
        size=(h,w)
        scale_factor_h=h/src_h
        scale_factor_w=w/src_w

        if 'train' in self.split:
            if self.crop_size[0]!=0 and self.crop_size[1]!=0:
                if h>=self.crop_size[0] and w>=self.crop_size[1]:
                    pass
                else:
                    scale_factor=max(self.crop_size[0]/h,self.crop_size[1]/w)
                    h=math.ceil(h*scale_factor)
                    h=h if h%self.opt.model_scale_factor==0 else h+self.opt.model_scale_factor-h%self.opt.model_scale_factor
                    w=math.ceil(w*scale_factor)
                    w=w if w%self.opt.model_scale_factor==0 else w+self.opt.model_scale_factor-w%self.opt.model_scale_factor
                    size=(h,w)
                    scale_factor_h=h/src_h
                    scale_factor_w=w/src_w
            if scale_factor_h==1. and scale_factor_w==1.:
                pass
            else:
                img = F.interpolate(img.unsqueeze(0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
                img = img.squeeze(0)

                img_strong = F.interpolate(img_strong.unsqueeze(0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
                img_strong = img_strong.squeeze(0)

        else:
            h=int(src_h*scale_factor)
            h=h if h%self.opt.model_scale_factor==0 else h+self.opt.model_scale_factor-h%self.opt.model_scale_factor
            w=int(src_w*scale_factor)
            w=w if w%self.opt.model_scale_factor==0 else w+self.opt.model_scale_factor-w%self.opt.model_scale_factor
            size=(h,w)
            scale_factor_h=h/src_h
            scale_factor_w=w/src_w
            if scale_factor_h==1. and scale_factor_w==1.:
                pass
            else:
                img = F.interpolate(img.unsqueeze(0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
                img = img.squeeze(0)

                img_strong = F.interpolate(img_strong.unsqueeze(0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
                img_strong = img_strong.squeeze(0)

        den=torch.zeros(h,w)
        if 'train' in self.split and self.crop_size[0]!=0 and self.crop_size[1]!=0 and 'crop' in self.weak_aug:
            start_y=random.randint(0,h-self.crop_size[0]) if h>self.crop_size[0] else 0
            start_x=random.randint(0,w-self.crop_size[1]) if h>self.crop_size[1] else 0
            end_y=h if h<=self.crop_size[0] else start_y+self.crop_size[0]
            end_x=w if w<=self.crop_size[1] else start_x+self.crop_size[1]
            img=img[:,start_y:end_y,start_x:end_x]
            img_strong=img_strong[:,start_y:end_y,start_x:end_x]
            cnt=0
            for i in range(len(ann_point)):
                y=int((ann_point[i][0])*scale_factor_h)
                x=int((ann_point[i][1])*scale_factor_w)
                if y>=start_y and y<end_y and x>=start_x and x<end_x:
                    den[y,x]+=1.
                    cnt+=1
            den=den[start_y:end_y,start_x:end_x]
        else:
            cnt=0
            for i in range(len(ann_point)):
                y=min(int((ann_point[i][0])*scale_factor_h),h-1)
                x=min(int((ann_point[i][1])*scale_factor_w),w-1)
                den[y,x]+=1.
                cnt+=1

        den=self.gen_map(den.unsqueeze(0).unsqueeze(0))
        den*=self.opt.gt_factor
        den=den.squeeze()

        # if random.random() < 0.5 and 'HorizontallyFlip' in self.weak_aug and 'train' in self.split:
        #     img = torch.flip(img, [2])
        #     img_strong = torch.flip(img_strong, [2])
        #     den = torch.flip(den, [1])
        # if random.random() < 0.5 and 'VerticallyFlip' in self.weak_aug and 'train' in self.split:
        #     img = torch.flip(img, [1])
        #     img_strong = torch.flip(img_strong, [1])
        #     den = torch.flip(den, [0])

        HorizontallyFlip=False
        VerticallyFlip=False
        if random.random() < 0.5 and 'HorizontallyFlip' in self.weak_aug and 'train' in self.split:
            img = torch.flip(img, [2])
            # img_strong = torch.flip(img_strong, [2])
            # den = torch.flip(den, [1])
            HorizontallyFlip=True
        if random.random() < 0.5 and 'VerticallyFlip' in self.weak_aug and 'train' in self.split:
            img = torch.flip(img, [1])
            # img_strong = torch.flip(img_strong, [1])
            # den = torch.flip(den, [0])
            VerticallyFlip=True

        if 'brightness' in self.strong_aug and random.random() < 0.5 and 'train' in self.split:
            img_strong = brightness(img_strong, severity=self.opt.brightness_severity)

        if 'contrast' in self.strong_aug and random.random() < 0.5 and 'train' in self.split:
            img_strong = contrast(img_strong, severity=self.opt.contrast_severity)

        if 'glass_blur' in self.strong_aug and random.random() < 0.5 and 'train' in self.split:
            img_strong = gaussian_blur(img_strong,sigma=self.opt.blur_sigma)

        if 'brightness' in self.weak_aug and random.random() < 0.5 and 'train' in self.split:
            img = brightness(img, severity=self.opt.brightness_severity)

        if 'contrast' in self.weak_aug and random.random() < 0.5 and 'train' in self.split:
            img = contrast(img, severity=self.opt.contrast_severity)

        if 'glass_blur' in self.weak_aug and random.random() < 0.5 and 'train' in self.split:
            img = gaussian_blur(img,sigma=self.opt.blur_sigma)


        for t, m, s in zip(img, self.mean, self.std):
            t.sub_(m).div_(s)

        for t, m, s in zip(img_strong, self.mean, self.std):
            t.sub_(m).div_(s)

        info={'dataset':'SHA','img_file':img_file,'img_path':img_path,'gt_cnt':cnt,'HorizontallyFlip':HorizontallyFlip,'VerticallyFlip':VerticallyFlip}

        return img,img_strong,den,info

    def __len__(self):
        return len(self.imgfiles)
        
def SHA_weak_strong_flip_just_weak(opt):
    # data_train=SHA_dataloader('train',opt)
    # train_loader = DataLoader(data_train, batch_size=opt.train_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,
    #     shuffle=True,drop_last=True,worker_init_fn=np.random.seed(opt.seed))

    # data_test=SHA_dataloader('val',opt)
    # test_loader = DataLoader(data_test, batch_size=opt.test_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,shuffle=False,worker_init_fn=np.random.seed(opt.seed))

    # return train_loader,data_train, test_loader,data_test



    data_train=SHA_dataloader('train',opt)

    if opt.num_gpu>1:
        sampler = DistributedSampler(data_train)
        train_loader = DataLoader(data_train, batch_size=opt.train_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,
            drop_last=True,
            worker_init_fn=np.random.seed(opt.seed),sampler=sampler)
    else:
        train_loader = DataLoader(data_train, batch_size=opt.train_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,
            shuffle=True,drop_last=True,
            worker_init_fn=np.random.seed(opt.seed))
    data_test=SHA_dataloader('val',opt)
    # if opt.num_gpu>1:
    #     sampler = DistributedSampler(data_test)
    #     test_loader = DataLoader(data_test, batch_size=opt.test_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,shuffle=False,
    #     worker_init_fn=np.random.seed(opt.seed),sampler=sampler)
    # else:
    #     test_loader = DataLoader(data_test, batch_size=opt.test_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,shuffle=False,
    #     worker_init_fn=np.random.seed(opt.seed))

    test_loader = DataLoader(data_test, batch_size=opt.test_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,shuffle=False,
        worker_init_fn=np.random.seed(opt.seed))

    # cnt=[]
    # for index in range(len(data_train)):
    #     img_file=data_train.imgfiles[index]
    #     ann_point=data_train.points_data[img_file]
    #     cnt.append(len(ann_point))

    # for index in range(len(data_test)):
    #     img_file=data_test.imgfiles[index]
    #     ann_point=data_test.points_data[img_file]
    #     cnt.append(len(ann_point))

    # cnt_array=np.array(cnt)
    # print(np.mean(cnt_array))
    # print(np.std(cnt_array))

    # avg_s=[]
    # avg_v=[]
    # for index in range(len(data_train)):
    #     img_file=data_train.imgfiles[index]
    #     ann_point=data_train.points_data[img_file]
    #     img_path=os.path.join(data_train.img_root,img_file)

    #     img=cv2.imread(img_path)
    #     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #     H, S, V = cv2.split(hsv)

    #     avg_s.append(np.mean(S))
    #     avg_v.append(np.mean(V))

    # for index in range(len(data_test)):
    #     img_file=data_test.imgfiles[index]
    #     ann_point=data_test.points_data[img_file]
    #     img_path=os.path.join(data_test.img_root,img_file)

    #     img=cv2.imread(img_path)
    #     hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    #     H, S, V = cv2.split(hsv)

    #     avg_s.append(np.mean(S))
    #     avg_v.append(np.mean(V))

    # import pickle
    # f = open('SHA_avg_s.pickle', 'wb')
    # pickle.dump(avg_s,f,protocol = 4)
    # f.close()
    # f = open('SHA_avg_v.pickle', 'wb')
    # pickle.dump(avg_v,f,protocol = 4)
    # f.close()

    # with open('save_b_line.pickle', 'rb') as fid:
    #     try:
    #         data2 = pickle.load(fid)
    #     except:
    #         data2 = pickle.load(fid, encoding='bytes')


    return train_loader,data_train, test_loader,data_test

def data_collate(data):
    img,strong_img,den,info= zip(*data)
    return img,strong_img,den,info
