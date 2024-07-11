import os
import sys
import numpy as np
import random
from scipy import sparse
import scipy
import math
import cv2
from tools.progress.bar import Bar
import pickle
import json
import pdb
from scipy.io import loadmat
from PIL import Image
from vision import save_keypoints_img
from importlib import import_module
# https://github.com/bethgelab/imagecorruptions

from imagecorruptions import corrupt
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.third_party_adapter.numpy_api as x2ms_np
import x2ms_adapter.torch_api.datasets as x2ms_datasets
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn
def rad(x):
    return x*np.pi/180

def perspective_transformation(img, transform,ann_point):
    h,w=img.shape[0:2]
    anglex=0
    angley = 0
    anglez = 0
    fov = 42

    for i in x2ms_adapter.tensor_api.split(transform, ' '):
        if x2ms_adapter.tensor_api.split(i, '_')[0]=='x':
            anglex=int(x2ms_adapter.tensor_api.split(i, '_')[1])
        elif x2ms_adapter.tensor_api.split(i, '_')[0]=='y':
            angley=int(x2ms_adapter.tensor_api.split(i, '_')[1])
        elif x2ms_adapter.tensor_api.split(i, '_')[0]=='z':
            anglez=int(x2ms_adapter.tensor_api.split(i, '_')[1])

    if anglex==0 and angley==0 and anglez==0:
        return img,ann_point

    #镜头与图像间的距离，21为半可视角，算z的距离是为了保证在此可视角度下恰好显示整幅图像
    z=x2ms_adapter.tensor_api.sqrt(np, w**2 + h**2)/2/np.tan(rad(fov/2))
    #齐次变换矩阵
    rx = np.array([[1,                  0,                          0,                          0],
                   [0,                  x2ms_adapter.tensor_api.cos(np, rad(anglex)),        -x2ms_adapter.tensor_api.sin(np, rad(anglex)),       0],
                   [0,                 -x2ms_adapter.tensor_api.sin(np, rad(anglex)),        x2ms_adapter.tensor_api.cos(np, rad(anglex)),        0,],
                   [0,                  0,                          0,                          1]], np.float32)
 
    ry = np.array([[x2ms_adapter.tensor_api.cos(np, rad(angley)), 0,                         x2ms_adapter.tensor_api.sin(np, rad(angley)),       0],
                   [0,                   1,                         0,                          0],
                   [-x2ms_adapter.tensor_api.sin(np, rad(angley)),0,                         x2ms_adapter.tensor_api.cos(np, rad(angley)),        0,],
                   [0,                   0,                         0,                          1]], np.float32)
 
    rz = np.array([[x2ms_adapter.tensor_api.cos(np, rad(anglez)), x2ms_adapter.tensor_api.sin(np, rad(anglez)),      0,                          0],
                   [-x2ms_adapter.tensor_api.sin(np, rad(anglez)), x2ms_adapter.tensor_api.cos(np, rad(anglez)),      0,                          0],
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

    if len(ann_point)==0:
        return resize,[]
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



class Gaussian(nn.Cell):
    def __init__(self, in_channels, sigmalist, kernel_size=64, stride=1, padding=0, froze=True):
        super(Gaussian, self).__init__()
        out_channels = len(sigmalist) * in_channels
        # gaussian kernel
        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma: x2ms_adapter.tensor_api.exp(math, -(x - mu) ** 2 / float(2 * sigma ** 2)))
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            gauss = x2ms_adapter.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss /= x2ms_adapter.tensor_api.x2ms_sum(gauss)
            _1D_window = x2ms_adapter.tensor_api.unsqueeze(gauss, 1)
            _2D_window = x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.mm(_1D_window, x2ms_adapter.tensor_api.t(_1D_window))), 0), 0)
            window = x2ms_adapter.autograd.Variable(x2ms_adapter.tensor_api.contiguous(x2ms_adapter.tensor_api.expand(_2D_window, in_channels, 1, kernel_size, kernel_size)))
            windows.append(window)
        kernels = x2ms_adapter.stack(windows)
        kernels = x2ms_adapter.tensor_api.permute(kernels, 1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)
        
        self.gkernel = x2ms_nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.gkernel.weight = mindspore.Parameter(weight)
        # # Froze this Gaussian net
        if froze: self.frozePara()

    def construct(self, dotmaps):
        gaussianmaps = self.gkernel(dotmaps)
        return gaussianmaps
    
    def frozePara(self):
        for para in x2ms_adapter.parameters(self):
            para.requires_grad = False

class Gaussianlayer(nn.Cell):
    def __init__(self, sigma=None, kernel_size=15):
        super(Gaussianlayer, self).__init__()
        if sigma == None:
            sigma = [4]
            # sigma = [kernel_size * 0.3]
        self.gaussian = Gaussian(1, sigma, kernel_size=kernel_size, padding=kernel_size // 2, froze=True)

    def construct(self, dotmaps):
        denmaps = self.gaussian(dotmaps)
        return denmaps

class gen_den_map(nn.Cell):
    def __init__(self,sigma,kernel_size):
        super(gen_den_map, self).__init__()
        self.gs = Gaussianlayer(sigma=[sigma],kernel_size=kernel_size)
    
    def construct(self, dot_map):
        # b*1*h*w
        gt_map = self.gs(dot_map)
        return gt_map

def fdst_sort_img(filename):
    return int(x2ms_adapter.tensor_api.split(filename, '.')[0])
def fdst_sort_video(filename):
    return int(filename)


def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]
    factor=x2ms_adapter.tensor_api.uniform_(x2ms_adapter.empty(1), -c, c)
    x=x2ms_adapter.clamp(x+c,0,1)
    return x

def gaussian_blur(img,sigma=[0.1,2.0]):
    sigma=x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.uniform_(x2ms_adapter.empty(1), sigma[0], sigma[1]))
    kernel_size=int(8*sigma+1)
    if kernel_size==1:
        return img
    if kernel_size%2==0:
        kernel_size+=1
    gass_kernel=gen_den_map(sigma, kernel_size)
    img=x2ms_adapter.tensor_api.unsqueeze(img, 0)
    img_cloned=x2ms_adapter.tensor_api.clone(img)
    img[:,0:1]=gass_kernel(img_cloned[:,0:1])
    img[:,1:2]=gass_kernel(img_cloned[:,1:2])
    img[:,2:3]=gass_kernel(img_cloned[:,2:3])
    img=x2ms_adapter.tensor_api.squeeze(img)
    return img


def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]
    factor=x2ms_adapter.tensor_api.uniform_(x2ms_adapter.empty(1), 1-c, 1+c)
    means = x2ms_adapter.x2ms_mean(x, [1,2], keepdim=True)
    x=x2ms_adapter.clamp((x-means)*factor+means,0,1)
    return x



class GCC_dataloader:
    def __init__(self,split,opt):

        self.split=split
        self.opt=opt
        self.mean=opt.mean_std[0]
        self.std=opt.mean_std[1]
        self.gen_map=gen_den_map(opt.sigma, opt.kernel_size)
        self.crop_size=opt.src_train_crop_size
        self.data_list=[]
        self.dataroot=opt.dataroot_GCC
        self.weak_aug=self.opt.weak_aug
        self.strong_aug=self.opt.strong_aug
        self.scale_factor=opt.src_scale_factor


        if opt.split_method=='location':
            prefix='cross_location_'
        elif opt.split_method=='camera':
            prefix='cross_camera_'
        elif opt.split_method=='random':
            prefix=''
        split_txt=os.path.join(opt.split_txt_path,prefix+'{}_list.txt'.format(self.split))

        with open(split_txt, "r") as f:
            if 'debug' in opt.comment:
                lines = f.readlines()[:500]
            else:
                lines = f.readlines()

        
        for line in lines:
            level,time,weather,path,imgfile,count=x2ms_adapter.tensor_api.split(x2ms_adapter.tensor_api.split(line, '\n')[0], ' ')
            level=int(level)
            time=int(time)
            weather=int(weather)
            scene_camera=x2ms_adapter.tensor_api.split(path, '/')[-1]
            count=int(count)
            radio=count/self.opt.level_capacity[level]
            if 'train' in self.split:
                if not level in self.opt.level_regularization:
                    continue
                if not weather in self.opt.weather_regularization:
                    continue
                if time<self.opt.time_regularization[0] or time>=self.opt.time_regularization[1]:
                    continue
                if count<self.opt.count_range[0] or time>self.opt.count_range[1]:
                    continue
                if radio<self.opt.radio_range[0] or radio>self.opt.radio_range[1]:
                    continue

            # ann_point_json=os.path.join(self.opt.dataroot_GCC_anno,scene_camera,'jsons',imgfile+'.json')

            ann_point_json=os.path.join(self.dataroot,scene_camera,'jsons',imgfile+'.json')
            with open(ann_point_json,'r') as f:
                ann_point=json.load(f)
            ann_point_list=ann_point['image_info'] #[[y,x],[y,x]]

            data_dict={
                    'dataset':'GCC',
                    'video_name':scene_camera,
                    'video_path':os.path.join(self.dataroot,scene_camera),
                    'img_path':os.path.join(self.dataroot,scene_camera,'pngs',imgfile+'.png'),
                    'img_file':imgfile+'.png',
                    'ann_point':ann_point_list,
                }

            self.data_list.append(data_dict)
        print('the number of images for {}img: {}'.format(self.split,len(self.data_list)))

        self.mix_domain_set(domain=self.opt.searched_domain)

        dataloader=import_module('datasets.{}'.format(opt.tar_dataset))
        exec('_,self.train_data_tar,_,_=dataloader.{}(opt)'.format(opt.tar_dataset))

    def mix_domain_set(self,domain):
        data_list_copy=self.data_list.copy()
        seed_cnt=0
        for i in range(len(self.opt.search_space)):
            temp=self.opt.attributes_range[self.opt.search_space[i]]
            search_space_range=temp[1]-temp[0]
            domain_set=domain[i]*search_space_range+temp[0]
            if self.opt.search_space[i]=='grey':
                random.seed(self.opt.seed+seed_cnt)
                seed_cnt+=1
                set_list=random.sample(list(range(len(data_list_copy))), int(domain[i]*len(data_list_copy)))
                for j in range(len(data_list_copy)):
                    if j in set_list:
                        self.data_list[j]['grey']=True
                    else:
                        self.data_list[j]['grey']=False

            elif self.opt.search_space[i]=='scale':
                random.seed(self.opt.seed+seed_cnt)
                seed_cnt+=1
                set_list=random.sample(list(range(len(data_list_copy))), int(0.5*len(data_list_copy)))
                for j in range(len(data_list_copy)):
                    if j in set_list:
                        self.data_list[j]['scale']=domain_set
                    else:
                        self.data_list[j]['scale']=1.0
            elif self.opt.search_space[i]=='perspective_transform':
                random.seed(self.opt.seed+seed_cnt)
                seed_cnt+=1
                set_list=random.sample(list(range(len(data_list_copy))), int(0.5*len(data_list_copy)))

                for j in range(len(data_list_copy)):
                    if j not in set_list:
                        self.data_list[j]['perspective_transform']='x_0'

                random.seed(self.opt.seed+seed_cnt)
                seed_cnt+=1
                t1=random.sample(set_list, int(0.125*len(data_list_copy)))
                for ind in t1:
                    self.data_list[ind]['perspective_transform']='x_-{}'.format(int(domain_set))

                set_list=list(set(set_list)-set(t1))

                random.seed(self.opt.seed+seed_cnt)
                seed_cnt+=1
                t2=random.sample(set_list, int(0.125*len(data_list_copy)))
                for ind in t2:
                    self.data_list[ind]['perspective_transform']='x_{}'.format(int(domain_set))

                set_list=list(set(set_list)-set(t2))
                random.seed(self.opt.seed+seed_cnt)
                seed_cnt+=1
                t3=random.sample(set_list, int(0.125*len(data_list_copy)))
                for ind in t3:
                    self.data_list[ind]['perspective_transform']='y_-{}'.format(int(domain_set))

                t4=list(set(set_list)-set(t3))
                for ind in t4:
                    self.data_list[ind]['perspective_transform']='y_{}'.format(int(domain_set))

    def __getitem__(self,index):
        data_dict = self.data_list[index]
        key_list=data_dict.keys()
        dataset=data_dict['dataset']
        video_name=data_dict['video_name']
        video_path=data_dict['video_path']
        img_path=data_dict['img_path']
        img_file=data_dict['img_file']
        ann_point=data_dict['ann_point']
        try:
            img=cv2.imread(img_path)
        except:
            print(img_path)

        if 'perspective_transform' in key_list:
            img,ann_point=perspective_transformation(img, data_dict['perspective_transform'],ann_point)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_strong=np.copy(img)


        #strong aug
        # if 'jpeg_compression' in self.strong_aug and random.random() < 0.5 and 'train' in self.split:
        #     img_strong = corrupt(img_strong, corruption_name='jpeg_compression', severity=1)

        if 'train' in self.split and data_dict['grey']:
            img_strong=cv2.cvtColor(img_strong, cv2.COLOR_RGB2GRAY)
            img_strong=img_strong[:,:,np.newaxis]
            img_strong=x2ms_np.concatenate((img_strong,img_strong,img_strong),axis=-1)
        #weak
        # if 'jpeg_compression' in self.weak_aug and random.random() < 0.5 and 'train' in self.split:
        #     img = corrupt(img, corruption_name='jpeg_compression', severity=1)

        if 'grey' in self.weak_aug and random.random() < 0.5 and 'train' in self.split:
            img=cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img=img[:,:,np.newaxis]
            img=x2ms_np.concatenate((img,img,img),axis=-1)

        img=x2ms_adapter.tensor_api.transpose(img, [2,0,1])# c h w
        img=x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.from_numpy(img))
        img=x2ms_adapter.tensor_api.div(img, 255)

        img_strong=x2ms_adapter.tensor_api.transpose(img_strong, [2,0,1])# c h w
        img_strong=x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.from_numpy(img_strong))
        img_strong=x2ms_adapter.tensor_api.div(img_strong, 255)
        
        src_h,src_w=x2ms_adapter.tensor_api.x2ms_size(img, 1),x2ms_adapter.tensor_api.x2ms_size(img, 2)
        scale_factor=1.0
        if 'train' in self.split:
            # scale_factor=self.scale_factor[np.random.randint(0,len(self.scale_factor),1).tolist()[0]]
            scale_factor=data_dict['scale']

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
                img = x2ms_adapter.nn_functional.interpolate(x2ms_adapter.tensor_api.unsqueeze(img, 0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
                img = x2ms_adapter.tensor_api.squeeze(img, 0)

                img_strong = x2ms_adapter.nn_functional.interpolate(x2ms_adapter.tensor_api.unsqueeze(img_strong, 0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
                img_strong = x2ms_adapter.tensor_api.squeeze(img_strong, 0)

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
                img = x2ms_adapter.nn_functional.interpolate(x2ms_adapter.tensor_api.unsqueeze(img, 0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
                img = x2ms_adapter.tensor_api.squeeze(img, 0)

                img_strong = x2ms_adapter.nn_functional.interpolate(x2ms_adapter.tensor_api.unsqueeze(img_strong, 0),size=size,mode=self.opt.img_interpolate_mode,align_corners = True)
                img_strong = x2ms_adapter.tensor_api.squeeze(img_strong, 0)

        den=x2ms_adapter.zeros(h,w)
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

        den=self.gen_map(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.unsqueeze(den, 0), 0))
        den*=self.opt.gt_factor
        den=x2ms_adapter.tensor_api.squeeze(den)

        HorizontallyFlip=False
        VerticallyFlip=False
        if random.random() < 0.5 and 'HorizontallyFlip' in self.weak_aug and 'train' in self.split:
            img = x2ms_adapter.flip(img, [2])
            # img_strong = torch.flip(img_strong, [2])
            # den = torch.flip(den, [1])
            HorizontallyFlip=True
        if random.random() < 0.5 and 'VerticallyFlip' in self.weak_aug and 'train' in self.split:
            img = x2ms_adapter.flip(img, [1])
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
            x2ms_adapter.tensor_api.div_(x2ms_adapter.tensor_api.sub_(t, m), s)

        for t, m, s in zip(img_strong, self.mean, self.std):
            x2ms_adapter.tensor_api.div_(x2ms_adapter.tensor_api.sub_(t, m), s)

        img_file=video_name+'_'+img_file
        info={'dataset':'GCC','img_file':img_file,'img_path':img_path,'gt_cnt':cnt,'data_dict':data_dict,'HorizontallyFlip':HorizontallyFlip,'VerticallyFlip':VerticallyFlip}
        tar_index=random.sample(list(range(len(self.train_data_tar.imgfiles))),1)[0]
        weak_tar_img,strong_tar_img,tar_den,tar_info=self.train_data_tar.__getitem__(tar_index)

        return img,img_strong,den,info,weak_tar_img,strong_tar_img,tar_den,tar_info


    def tensor_to_numpy(self,img_tensor):
        new_img=x2ms_adapter.zeros(x2ms_adapter.tensor_api.x2ms_size(img_tensor))
        for i in range(3):
            new_img[i]=img_tensor[i]*self.std[i]+self.mean[i]
        new_img*=255
        new_img=x2ms_adapter.tensor_api.numpy(new_img)
        new_img=new_img.astype(np.uint8)
        return new_img

    def __len__(self):
        return len(self.data_list)

def GCC_weak_strong_flip_just_weak_fix(opt):
    data_train=GCC_dataloader('train',opt)

    if opt.num_gpu>1:
        sampler = x2ms_datasets.DistributedSampler(data_train)
        train_loader = x2ms_datasets.DataLoader(data_train, batch_size=opt.train_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,
            drop_last=True,
            worker_init_fn=np.random.seed(opt.seed),sampler=sampler)
    else:
        train_loader = x2ms_datasets.DataLoader(data_train, batch_size=opt.train_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,
            shuffle=True,drop_last=True,
            worker_init_fn=np.random.seed(opt.seed))
    data_test=GCC_dataloader('test',opt)
    # if opt.num_gpu>1:
    #     sampler = DistributedSampler(data_test)
    #     test_loader = DataLoader(data_test, batch_size=opt.test_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,shuffle=False,
    #     worker_init_fn=np.random.seed(opt.seed),sampler=sampler)
    # else:
    #     test_loader = DataLoader(data_test, batch_size=opt.test_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,shuffle=False,
    #     worker_init_fn=np.random.seed(opt.seed))

    
    # test_loader = DataLoader(data_test, batch_size=opt.test_batch_size, num_workers=opt.num_workers,collate_fn=data_collate,shuffle=False,
    #     worker_init_fn=np.random.seed(opt.seed))

    # cnt=[]
    # for index in range(len(data_train)):
    #     data_dict = data_train.data_list[index]
    #     img_file=data_dict['img_file']
    #     ann_point=data_dict['ann_point']

    #     cnt.append(len(ann_point))

    # for index in range(len(data_test)):


    #     data_dict = data_test.data_list[index]
    #     img_file=data_dict['img_file']
    #     ann_point=data_dict['ann_point']
        
    #     cnt.append(len(ann_point))

    # cnt_array=np.array(cnt)
    # print(np.mean(cnt_array))
    # print(np.std(cnt_array))

    avg_s=[]
    avg_v=[]
    data_dict_save=[]
    for index in range(len(data_train)):
        data_dict = data_train.data_list[index]
        # img_path=data_dict['img_path']
        # img_file=data_dict['img_file']
        # img=cv2.imread(img_path)
        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # H, S, V = cv2.split(hsv)

        data_dict_save.append(data_dict)
        # avg_s.append(np.mean(S))
        # avg_v.append(np.mean(V))

    for index in range(len(data_test)):
        data_dict = data_test.data_list[index]
        # img_path=data_dict['img_path']
        # img_file=data_dict['img_file']
        # img=cv2.imread(img_path)
        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        # H, S, V = cv2.split(hsv)
        data_dict_save.append(data_dict)
        # avg_s.append(np.mean(S))
        # avg_v.append(np.mean(V))
    
    # import pickle
    # f = open('GCC_avg_s.pickle', 'wb')
    # pickle.dump(avg_s,f,protocol = 4)
    # f.close()
    # f = open('GCC_avg_v.pickle', 'wb')
    # pickle.dump(avg_v,f,protocol = 4)
    # f.close()
    f = open('GCC_data_dict.pickle', 'wb')
    pickle.dump(data_dict_save,f,protocol = 4)
    f.close()

    return train_loader,data_train, None,None





def data_collate(data):
    img,strong_img,den,info,weak_tar_img,strong_tar_img,tar_den,tar_info= zip(*data)
    return img,strong_img,den,info,weak_tar_img,strong_tar_img,tar_den,tar_info
