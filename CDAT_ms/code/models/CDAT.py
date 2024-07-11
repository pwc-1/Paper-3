import numpy as np
import cv2
import random
import os
from tools.log import AverageMeter
from utils import AverageMeter_acc_map
from vision import save_keypoints_img,save_img_tensor
import pdb
import time
import torch
import math
from x2ms_adapter.torch_api.optimizers import optim_register
import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.torch_api.lr_schedulers as lr_schedule_wrapper
import x2ms_adapter.torch_api.nn_api.loss as loss_wrapper
import x2ms_adapter.torch_api.nn_api.nn as x2ms_nn


def get_optimizer(model,opt):
    params = []
    frozen_layers = []
    main_layers=[]
    secondary_layers=[]
    for pname, p in x2ms_adapter.named_parameters(model):
        if p.requires_grad:
            params.append({'params': p})
        elif not p.requires_grad:
            print(pname)
        if pname.split('.')[0] in opt.frozen_layers:
            frozen_layers.append(p)
            continue
        if pname.split('.')[0] in opt.secondary_layers:
            secondary_layers.append(p)
            continue
        main_layers.append(p)

    # # # Fix frozen_layers ---------------
    for param in frozen_layers:
        param.requires_grad = False
    # # # ----------------------------------

    if opt.optimizer.lower()=='adam':
        group=[{'params':main_layers, 'lr': opt.main_lr_init,'weight_decay': opt.main_weight_decay}]
        optimizer = nn.Adam(params=group)
    elif opt.optimizer.lower()=='sgd':
        optimizer = nn.SGD(params=[{'params':main_layers, 'lr': opt.main_lr_init,'weight_decay': opt.main_weight_decay},
                                     {'params':secondary_layers, 'lr': opt.secondary_lr_init,'weight_decay': opt.secondary_weight_decay}])
    return optimizer


class Conv2d(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='relu', same_padding=False, bn=False, dilation=1):
        super(Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0
        self.conv = []
        if dilation==1:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation,pad_mode='pad',has_bias=True)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=dilation, dilation=dilation,pad_mode='pad',has_bias=True)
        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu' :
            self.relu = nn.ReLU()
        elif NL == 'prelu':
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def construct(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x



'''MLP model'''  #https://github.com/HarshayuGirase/PECNet/blob/master/utils/models.py
class MLP(nn.Cell):
    def __init__(self, input_dim, output_dim=None, hidden_size=(1024, 512), activation='relu', discrim=False, dropout=-1):
        super(MLP, self).__init__()
        dims = []
        dims.append(input_dim)
        dims.extend(hidden_size)
        if output_dim:
            dims.append(output_dim)
        self.layers = x2ms_nn.ModuleList()
        for i in range(len(dims)-1):
            self.layers.append(x2ms_nn.Linear(dims[i], dims[i+1]))

        if activation == 'relu':
            self.activation = x2ms_nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = x2ms_nn.Sigmoid()

        self.sigmoid = x2ms_nn.Sigmoid() if discrim else None
        self.dropout = dropout

    def construct(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i != len(self.layers)-1:
                x = self.activation(x)
                if self.dropout != -1:
                    x = x2ms_nn.Dropout(min(0.1, self.dropout/3) if i == 1 else self.dropout)(x)
            elif self.sigmoid:
                x = x2ms_adapter.tensor_api.sigmoid(self, x)
        return x

# class GradientReversalLayer(torch.autograd.Function):
#     """
#     Implement the gradient reversal layer for the convenience of domain adaptation neural network.
#     The forward part is the identity function while the backward part is the negative function.
#     """
#     def forward(self, inputs):
#         return inputs

#     def backward(self, grad_output):
#         grad_input = grad_output.clone()
#         grad_input = -grad_input
#         return grad_input

class _GradientScalarLayer(x2ms_adapter.autograd.Function):
    @staticmethod
    def construct(ctx, input, weight):
        ctx.weight = weight
        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.copy()
        return ctx.weight*grad_input, None

gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(nn.Cell):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def construct(self, input):
        return gradient_scalar(input, self.weight)

    def __repr__(self):
        tmpstr = self.__class__.__name__ + "("
        tmpstr += "weight=" + str(self.weight)
        tmpstr += ")"
        return tmpstr

class Domain_classifier(nn.Cell):
    def __init__(self,in_channels,grad_scalar,opt):
        super(Domain_classifier, self).__init__()
        channel_list=[in_channels]+opt.MLP_layer+[1]

        self.conv=x2ms_nn.ModuleList()
        for i in range(len(channel_list)-1):
            self.conv.append(Conv2d(channel_list[i], channel_list[i+1], 5, NL='', same_padding=False))
        # self.conv=Conv2d(in_channels, opt.MLP_layer[0], 3, same_padding=True, NL='relu')
        # self.fc=MLP(opt.MLP_layer[0],output_dim=1,hidden_size=opt.MLP_layer[1:])
        self.grl=GradientScalarLayer(grad_scalar)
    def construct(self,x):
        b=x2ms_adapter.tensor_api.x2ms_size(x, 0)
        x=self.grl(x)
        for m in self.conv:
            x=m(x)
        # b*2*h*w
        return x


class WeightEMA (object):
    """
    Exponential moving average weight optimizer for mean teacher model
    """
    def __init__(self, params, src_params, alpha=0.999):
        self.params = list(params)
        self.src_params = list(src_params)
        self.alpha = alpha

        for p, src_p in zip(self.params, self.src_params):
            p.data[:] = src_p.data[:]

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for p, src_p in zip(self.params, self.src_params):
            x2ms_adapter.tensor_api.mul_(p.data, self.alpha)
            x2ms_adapter.tensor_api.add_(p.data, src_p.data * one_minus_alpha)


class all_local_maximum_location_get(nn.Cell):
    def __init__(self,k_size,thd):
        super(all_local_maximum_location_get, self).__init__()
        assert k_size%2==1
        self.max_pool=nn.MaxPool2d(kernel_size=k_size, stride=1,padding=int(k_size/2),pad_mode='pad')
        # self.avg_pool=nn.AvgPool2d(kernel_size=k_size, stride=1,padding=int(k_size/2))
        self.thd=thd
    def construct(self,x):
        # x: channel*h*w   or b*channel*h*w
        
        max_val=mindspore.ops.max(x)
        min_val=mindspore.ops.min(x)
        y=self.max_pool(x)
        # y_avg=self.avg_pool(x)
        mask=(y>self.thd*max_val)
        y_=(y==x)*(y!=0)*mask
        y_=y_.squeeze()

        pred_cnt=x2ms_adapter.tensor_api.item(x2ms_adapter.x2ms_sum(y_,(-2,-1)))
        return y_,pred_cnt

class BasicDeconv(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn=False):
        super(BasicDeconv, self).__init__()
        self.use_bn = use_bn
        self.tconv = mindspore.nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride=stride, has_bias=not self.use_bn)
        self.bn = nn.BatchNorm2d(out_channels, affine=True) if self.use_bn else None

    def construct(self, x):
        # pdb.set_trace()
        x = self.tconv(x)
        if self.use_bn:
            x = self.bn(x)
        return mindspore.ops.relu(x)


class den_level_Discriminator(nn.Cell):
    def __init__(self, input_nc, ndf=512, num_classes=1,opt=None):
        super(den_level_Discriminator, self).__init__()
        self.grl=GradientScalarLayer(opt.gradient_scalar)
        # self.D = nn.Sequential(
        #     nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)
        # self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)


        self.D = nn.SequentialCell(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1,pad_mode='pad'),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1,pad_mode='pad'),
            nn.LeakyReLU(alpha=0.2)
        )
        self.cls = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1,pad_mode='pad')

    def construct(self, x, size=None):
        # out = self.D(x)
        # src_out = self.cls1(out)
        # tgt_out = self.cls2(out)
        # out = torch.cat((src_out, tgt_out), dim=1)
        # if size is not None:
        #     out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        # return out
        x=self.grl(x)
        out = self.D(x)
        out = self.cls(out)
        if size is not None:
            out = mindspore.ops.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out



class get_den_level_label(nn.Cell):
    def __init__(self, opt):
        super(get_den_level_label, self).__init__()
        self.opt=opt
        self.split_den_level=opt.split_den_level
        self.k1= x2ms_nn.AvgPool2d(opt.grid, stride=opt.grid)

    def construct(self, den_map):
        batch=len(den_map)
        label_list=[]
        for i in range(batch):
            local_cnt=self.k1(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.detach(den_map[i])), 0), 0))*self.opt.grid[0]*self.opt.grid[1]/self.opt.gt_factor
            local_cnt=x2ms_adapter.tensor_api.squeeze(local_cnt)
            den_level_map=[]
            for j in range(len(self.split_den_level)-1):
                tmp=x2ms_adapter.tensor_api.x2ms_float(((local_cnt>=self.split_den_level[j])*(local_cnt<self.split_den_level[j+1])))
                den_level_map.append(tmp)
            label=x2ms_adapter.stack(den_level_map,0)#level*h*w
            if len(x2ms_adapter.tensor_api.x2ms_size(label))==1:
                label=x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.unsqueeze(label, 1), 1)
            label_list.append(label)

        return label_list



class Gaussian(nn.Cell):
    def __init__(self, in_channels, sigmalist, kernel_size=64, stride=1, padding=0, froze=True):
        super(Gaussian, self).__init__()
        out_channels = len(sigmalist) * in_channels
        # gaussian kernel
        mu = kernel_size // 2
        gaussFuncTemp = lambda x: (lambda sigma: math.exp(-(x - mu) ** 2 / float(2 * sigma ** 2)))
        gaussFuncs = [gaussFuncTemp(x) for x in range(kernel_size)]
        windows = []
        for sigma in sigmalist:
            gauss = mindspore.Tensor([gaussFunc(sigma) for gaussFunc in gaussFuncs])
            gauss /= gauss.sum()
            _1D_window = gauss.unsqueeze(1)
            _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
            window = _2D_window.broadcast_to((in_channels, 1, kernel_size, kernel_size))
            windows.append(window)
        kernels = mindspore.ops.stack(windows)
        kernels = kernels.permute(1, 0, 2, 3, 4)
        weight = kernels.reshape(out_channels, in_channels, kernel_size, kernel_size)
        
        self.gkernel = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, group=in_channels, has_bias=False,pad_mode='pad')
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


class CDAT(nn.Cell):
    def __init__(self, opt):
        super(CDAT, self).__init__()

        self.student_model=VGG_16_deconv_DA_den_level_fix_BCE_BF_cls(opt)
        self.teacher_model=VGG_16_deconv_DA_den_level_fix_BCE_BF_cls(opt)

        # self.scheduler=self.student_model.scheduler

        # teacher_params = []
        # for key, value in dict(x2ms_adapter.named_parameters(self.teacher_model)).items():
        #     if value.requires_grad:
        #         teacher_params += [value]
        #         value.requires_grad = False

        # student_params = []
        # for key, value in dict(x2ms_adapter.named_parameters(self.student_model)).items():
        #     if value.requires_grad:
        #         student_params += [value]


        # self.teacher_optimizer = WeightEMA(
        # teacher_params, student_params, alpha=opt.teacher_alpha
        # )

        

        


    def next_epoch(self):
        self.student_model.train_reset()
        self.student_model.test_reset()
        self.teacher_model.train_reset()


    def construct(self, data,search_time,epoch,global_step,step,vision_list):
        tar_teacher_den_pred,tar_teacher_den_pred_gass,tar_teacher_den_mix,BFs,lds,mae_src_teacher,mae_src_teacher_points,mae_src_teacher_mix=self.teacher_model.teacher_inference(data,search_time,epoch,global_step,step,vision_list)
        res=self.student_model(data,search_time,epoch,global_step,step,vision_list,tar_teacher_den_pred,tar_teacher_den_pred_gass,tar_teacher_den_mix,BFs,lds,mae_src_teacher,mae_src_teacher_points,mae_src_teacher_mix)
        # self.teacher_optimizer.step()

        return res

    def inference(self,data,search_time,epoch,global_step,step,vision_list,batch_num_each_epoch,logger):
        res=self.student_model.inference(data,search_time,epoch,global_step,step,vision_list,batch_num_each_epoch,logger)
        return res


class BF_classifier(nn.Cell):
    def __init__(self, input_nc, ndf=512, num_classes=1,opt=None):
        super(BF_classifier, self).__init__()
        # self.D = nn.Sequential(
        #     nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True),
        #     nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
        #     nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # )
        # self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)
        # self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)


        self.D = nn.SequentialCell(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1,pad_mode='pad'),
            nn.LeakyReLU(alpha=0.2),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1,pad_mode='pad'),
            nn.LeakyReLU(alpha=0.2)
        )
        self.cls = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1,pad_mode='pad')

    def construct(self, x, size=None):
        # out = self.D(x)
        # src_out = self.cls1(out)
        # tgt_out = self.cls2(out)
        # out = torch.cat((src_out, tgt_out), dim=1)
        # if size is not None:
        #     out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        # return out
        out = self.D(x)
        out = self.cls(out)
        if size is not None:
            out = mindspore.ops.interpolate(out, size=size, mode='bilinear', align_corners=True)
        return out

class get_den_level_label_1(nn.Cell):
    def __init__(self, grid,split,opt):
        super(get_den_level_label_1, self).__init__()
        self.opt=opt
        self.split_den_level=split
        self.grid=grid
        self.k1= nn.AvgPool2d(tuple(self.grid), stride=tuple(self.grid))

    def construct(self, den_map):
        batch=len(den_map)
        label_list=[]
        for i in range(batch):
            local_cnt=self.k1(den_map[i].squeeze().unsqueeze(0).unsqueeze(0))*self.grid[0]*self.grid[1]/self.opt.gt_factor
            local_cnt=local_cnt.squeeze()
            den_level_map=[]
            for j in range(len(self.split_den_level)-1):
                tmp=((local_cnt>self.split_den_level[j])*(local_cnt<self.split_den_level[j+1])).float()
                den_level_map.append(tmp)
            label=mindspore.ops.stack(den_level_map,0)#level*h*w
            if len(label.size())==1:
                label=label.unsqueeze(1).unsqueeze(1)
            label_list.append(label)

        return label_list

class VGG16(nn.Cell):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=3, padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(512, 512, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3,padding=1,pad_mode='pad'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )
 
    def forward(self, x):
        x = self.features(x)
        x = mindspore.ops.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


class VGG_16_deconv_DA_den_level_fix_BCE_BF_cls(nn.Cell):
    def __init__(self, opt):
        super(VGG_16_deconv_DA_den_level_fix_BCE_BF_cls, self).__init__()
        self.gradient_scalar=opt.gradient_scalar
        self.opt=opt
        self.step_epoch=0
        self.get_BF_label=get_den_level_label_1(opt.BF_grid,opt.BF_thd,opt)
        self.get_max_point=all_local_maximum_location_get(opt.maxima_ksize,opt.den_map_thd_factor)
        self.gen_map=gen_den_map(opt.sigma, opt.kernel_size)


        self.mix_grid_avg_pool=nn.AvgPool2d(tuple(opt.mix_grid),tuple(opt.mix_grid))
        self.local_density_grid_avg_pool=nn.AvgPool2d(tuple(opt.local_density_grid),tuple(opt.local_density_grid))
        self.avg_pool8=nn.AvgPool2d((8,8),(8,8))

        exec('self.loss_fcn={}(opt)'.format(opt.loss_choose))
        # self.get_local_maximum=all_local_maximum_location_get(3)
        # exec('self.loss_fcn={}(opt)'.format(opt.loss_choose))
        # vgg = VGG16()

        ## Load the pre-trained VGG model parameters
        # vgg.load_state_dict(torch.load(model_path))

        # features = list(x2ms_adapter.nn_cell.children(vgg.features))


        self.backbone = nn.SequentialCell(
            nn.Conv2d(3, 64, kernel_size=3, padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3,padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3,padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3,padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3,padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3,padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3,padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3,padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3,padding=1,pad_mode='pad',has_bias=True),
            nn.ReLU()
        )

        self.den_pred = nn.SequentialCell(
                                     Conv2d(512, 128, 3, same_padding=True, bn=True, NL='relu'),
                                     BasicDeconv(128, 128, 2, 2, True),
                                     Conv2d(128, 64, 3, same_padding=True, bn=True, NL='relu'),
                                     BasicDeconv(64, 64, 2, 2, True),
                                     Conv2d(64, 32, 3, same_padding=True, bn=True, NL='relu'),
                                     BasicDeconv(32, 32, 2, 2, True),
                                     Conv2d(32, 1, 1, same_padding=True, NL=''),
                                     nn.ReLU(),
                                     )
        # init_weights(self.den_pred)
        self.domain_cls=den_level_Discriminator(512,512,1,opt)
        # init_weights(self.domain_cls)
        if self.opt.BF_loss_choose.lower()=='bce':
            self.BF_cls=BF_classifier(512,256,1,opt)
        elif self.opt.BF_loss_choose.lower()=='softmax':
            self.BF_cls=BF_classifier(512,256,2,opt)
        # init_weights(self.BF_cls)

        self.Local_den_predictor=BF_classifier(512,256,1,opt)
        # init_weights(self.Local_den_predictor)


        #"only_density_map"  # ["only_density_map","only_feature_map","all"]
        if self.opt.concat_way=='only_density_map':
            channel=2
        elif self.opt.concat_way=='only_feature_map':
            channel=512
        elif self.opt.concat_way=='all':
            channel=514

        if self.opt.ada_mix_loss_choose.lower()=='bce':
            self.mix_cls=BF_classifier(channel,256,1,opt)
        elif self.opt.ada_mix_loss_choose.lower()=='softmax':
            self.mix_cls=BF_classifier(channel,256,2,opt)
        # init_weights(self.mix_cls)


        if opt.model_for_load:
            checkpoint=x2ms_adapter.load(opt.model_for_load)
            model_dict =  x2ms_adapter.nn_cell.state_dict(self)
            state_dict = {k:v for k,v in checkpoint['net'].items() if k in model_dict.keys()}
            print(state_dict.keys())  # dict_keys(['w', 'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'])
            model_dict.update(state_dict)
            x2ms_adapter.load_state_dict(self, model_dict)

        # self.optimizer=get_optimizer(self,self.opt)
        # if self.opt.train_mode=='step':
        #     # self.scheduler = mindspore.experimental.optim.lr_scheduler.StepLR(self.optimizer, self.opt.decay_iter_freq, gamma=self.opt.decay_gamma)
        #     self.scheduler=[0.1]
    def next_epoch(self):
        self.train_reset()
        self.test_reset()

    def train_reset(self):
    
        self.mae_src=AverageMeter()
        self.mae_tar=AverageMeter()
        self.mse_src=AverageMeter()
        self.mse_tar=AverageMeter()
        self.mae_src_teacher=AverageMeter()
        self.mae_src_teacher_points=AverageMeter()
        self.mae_src_teacher_mix=AverageMeter()
        self.loss=AverageMeter()
        self.den_loss=AverageMeter()
        self.D_src_loss=AverageMeter()
        self.D_tar_loss=AverageMeter()
        self.p_label_loss=AverageMeter()
        self.p_label_gass_loss=AverageMeter()
        self.p_label_mix_loss=AverageMeter()
        self.src_BF_loss=AverageMeter()
        self.src_local_density_loss=AverageMeter()
        self.tar_BF_loss=AverageMeter()
        self.tar_local_density_loss=AverageMeter()
        self.mix_loss=AverageMeter()
        self.mask_den_loss=AverageMeter()
        self.BF_acc_src=AverageMeter()
        self.BF_acc_tar=AverageMeter()


    def get_train_res(self):
        res={}
        res['mae_s']=self.mae_src.avg
        res['mse_s']=self.mse_src.avg**0.5
        res['mae_t']=self.mae_tar.avg
        res['mse_t']=self.mse_tar.avg**0.5
        res['mae_tea']=self.mae_src_teacher.avg
        res['mae_tea_p']=self.mae_src_teacher_points.avg
        res['mae_tea_m']=self.mae_src_teacher_mix.avg
        res['loss']=self.loss.avg
        res['d']=self.den_loss.avg
        res['Ds']=self.D_src_loss.avg
        res['Dt']=self.D_tar_loss.avg
        res['p']=self.p_label_loss.avg
        res['pg']=self.p_label_gass_loss.avg
        res['pm']=self.p_label_mix_loss.avg
        res['sBF']=self.src_BF_loss.avg
        res['sldl']=self.src_local_density_loss.avg
        res['tBF']=self.tar_BF_loss.avg
        res['tldl']=self.tar_local_density_loss.avg
        res['m']=self.mix_loss.avg
        res['md']=self.mask_den_loss.avg
        res['BF_sa']=self.BF_acc_src.avg*100
        res['BF_ta']=self.BF_acc_tar.avg*100


        return res

    def test_reset(self):
        self.mae=AverageMeter()
        self.mse=AverageMeter()
        if 'WE' in self.opt.tar_dataset:
            self.mae1=AverageMeter()
            self.mae2=AverageMeter()
            self.mae3=AverageMeter()
            self.mae4=AverageMeter()
            self.mae5=AverageMeter()

        if self.opt.tar_dataset in ['WE_weak_strong_flip_just_weak', 'WE_blurred_weak_strong_flip_just_weak','WE_blurred_mask_weak_strong_flip_just_weak']:
            self.mae1_mask=AverageMeter()
            self.mae2_mask=AverageMeter()
            self.mae3_mask=AverageMeter()
            self.mae4_mask=AverageMeter()
            self.mae5_mask=AverageMeter()

            self.mae1_mask_dilate=AverageMeter()
            self.mae2_mask_dilate=AverageMeter()
            self.mae3_mask_dilate=AverageMeter()
            self.mae4_mask_dilate=AverageMeter()
            self.mae5_mask_dilate=AverageMeter()

    def get_test_res(self):
        res={}
        res['mae']=self.mae.avg
        res['mse']=self.mse.avg**0.5

        if 'WE' in self.opt.tar_dataset:
            res['mae1']=self.mae1.avg
            res['mae2']=self.mae2.avg
            res['mae3']=self.mae3.avg
            res['mae4']=self.mae4.avg
            res['mae5']=self.mae5.avg
        if self.opt.tar_dataset in ['WE_weak_strong_flip_just_weak', 'WE_blurred_mask_weak_strong_flip_just_weak']:
            res['1_m']=self.mae1_mask.avg
            res['2_m']=self.mae2_mask.avg
            res['3_m']=self.mae3_mask.avg
            res['4_m']=self.mae4_mask.avg
            res['5_m']=self.mae5_mask.avg

            res['1_md']=self.mae1_mask_dilate.avg
            res['2_md']=self.mae2_mask_dilate.avg
            res['3_md']=self.mae3_mask_dilate.avg
            res['4_md']=self.mae4_mask_dilate.avg
            res['5_md']=self.mae5_mask_dilate.avg

        return res


    def get_mix_gt(self,src_den_map,src_gt_map):
        batch=len(src_den_map)
        den_maps_gaussian=[]
        for i in range(batch):
            points_map,num_points=self.get_max_point(x2ms_adapter.tensor_api.detach(src_den_map[i:i+1]))
            points_map=x2ms_adapter.tensor_api.x2ms_float(points_map)
            points_gass_map=self.gen_map(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.unsqueeze(points_map, 0), 0))*self.opt.gt_factor
            den_maps_gaussian.append(points_gass_map)

        den_maps_gaussian=x2ms_adapter.cat(den_maps_gaussian,0)
        src_local_den_map=self.mix_grid_avg_pool(x2ms_adapter.tensor_api.detach(src_den_map))
        src_gt_local_den_map=self.mix_grid_avg_pool(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.stack(src_gt_map,0), 1))
        src_local_gass_den_map=self.mix_grid_avg_pool(den_maps_gaussian)

        error1=x2ms_adapter.x2ms_abs(src_local_den_map-src_gt_local_den_map)
        error2=x2ms_adapter.x2ms_abs(src_local_gass_den_map-src_gt_local_den_map)

        mix_label = x2ms_adapter.zeros(size=x2ms_adapter.tensor_api.x2ms_size(src_gt_local_den_map))
        mix_label[error1 <= error2] = 1.0

        src_local_den_map=self.avg_pool8(x2ms_adapter.tensor_api.detach(src_den_map))*64
        src_local_gass_den_map=self.avg_pool8(den_maps_gaussian)*64

        return mix_label,src_local_den_map,src_local_gass_den_map


    def get_local_for_teacher_inference(self,den_map,gass_den_map):

        local_den_map=self.mix_grid_avg_pool(x2ms_adapter.tensor_api.detach(den_map))
        out_shape=local_den_map.shape[-2:]

        local_den_map=self.avg_pool8(x2ms_adapter.tensor_api.detach(den_map))*64
        local_gass_den_map=self.avg_pool8(gass_den_map)*64

        return local_den_map,local_gass_den_map,out_shape

    def construct(self, data,search_time,epoch,global_step,step,vision_list,tar_teacher_den_pred,tar_teacher_den_pred_gass,tar_teacher_den_mix,tar_BF_plabel,tar_ld_plabel,mae_src_teacher,mae_src_teacher_points,mae_src_teacher_mix):

        self.mae_src_teacher=mae_src_teacher
        self.mae_src_teacher_points=mae_src_teacher_points
        self.mae_src_teacher_mix=mae_src_teacher_mix

        self.scheduler.step()
        str_prefix=''
        # model(src_data,tar_data,global_step,step,train_vision)
        src_data=data[:4]
        tar_data=data[4:8]

        batch=len(src_data[0])
        if batch>1:
            src_img=x2ms_adapter.stack(src_data[1])
            tar_img=x2ms_adapter.stack(tar_data[1])
        else:
            src_img=x2ms_adapter.tensor_api.unsqueeze(src_data[1][0], 0)
            tar_img=x2ms_adapter.tensor_api.unsqueeze(tar_data[1][0], 0)


        src_BF_gt=self.get_BF_label(src_data[2])
        src_BF_gt=x2ms_adapter.tensor_api.detach(x2ms_adapter.stack(src_BF_gt,0))

        src_fea=self.backbone(src_img)
        src_den_map=self.den_pred(src_fea)
        src_BF=self.BF_cls(src_fea,src_BF_gt.shape[-2:])

        src_ld_gt=self.local_density_grid_avg_pool(x2ms_adapter.stack(src_data[2],0))*self.opt.local_density_grid[0]*self.opt.local_density_grid[1]
        src_local_den=self.Local_den_predictor(src_fea,src_ld_gt.shape[-2:])
        src_D_pred = self.domain_cls(src_fea,src_fea.shape[-2:])

        #############

        mix_gt,src_local_den_map,src_local_gass_den_map=self.get_mix_gt(src_den_map,src_data[2])
        if self.opt.concat_way=='only_density_map':
            inp=x2ms_adapter.cat([src_local_den_map,src_local_gass_den_map],dim=1)
        elif self.opt.concat_way=='only_feature_map':
            inp=src_fea
        elif self.opt.concat_way=='all':
            inp=x2ms_adapter.cat([src_fea,src_local_den_map,src_local_gass_den_map],dim=1)
        mix_pred=self.mix_cls(inp,mix_gt.shape[-2:])

        #############
        '''
        tar
        '''
        tar_BF_gt=self.get_BF_label(tar_data[2])
        tar_BF_gt=x2ms_adapter.tensor_api.detach(x2ms_adapter.stack(tar_BF_gt,0))

        tar_fea=self.backbone(tar_img)
        tar_den_map=self.den_pred(tar_fea)
        tar_BF=self.BF_cls(tar_fea,tar_BF_gt.shape[-2:])
        tar_local_den=self.Local_den_predictor(tar_fea,src_ld_gt.shape[-2:])
        tar_D_pred=self.domain_cls(tar_fea,tar_fea.shape[-2:])

        loss,loss_D_src,loss_D_tgt,den_loss,p_label_den_loss,p_label_gass_den_loss,p_label_mix_den_loss,src_BF_loss,tar_BF_loss,src_local_density_loss,tar_local_density_loss,mix_loss,mask_den_loss=self.loss_fcn(
            epoch,src_den_map,tar_den_map,src_D_pred,tar_D_pred,src_data[2],tar_teacher_den_pred,tar_teacher_den_pred_gass,tar_teacher_den_mix,
            src_BF,src_BF_gt,tar_BF,tar_BF_plabel,src_local_den,src_ld_gt,tar_local_den,tar_ld_plabel,mix_pred,mix_gt
            )


        self.loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(loss)),batch)
        self.D_src_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(loss_D_src)),batch)
        self.D_tar_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(loss_D_tgt)),batch)
        self.den_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(den_loss)),batch)
        self.p_label_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(p_label_den_loss)),batch)
        self.p_label_gass_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(p_label_gass_den_loss)),batch)
        self.p_label_mix_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(p_label_mix_den_loss)),batch)
        self.src_local_density_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(src_local_density_loss)),batch)
        self.src_BF_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(src_BF_loss)),batch)
        self.tar_local_density_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(tar_local_density_loss)),batch)
        self.tar_BF_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(tar_BF_loss)),batch)
        self.mix_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(mix_loss)),batch)
        self.mask_den_loss.update(x2ms_adapter.tensor_api.item(x2ms_adapter.tensor_api.detach(mask_den_loss)),batch)
        # x2ms_adapter.nn_cell.zero_grad(self.optimizer)
        # loss.backward()
        # self.optimizer.step()

        if self.opt.BF_loss_choose.lower()=='bce':
            src_mask_pred=x2ms_adapter.tensor_api.x2ms_float((x2ms_adapter.nn_functional.sigmoid(src_BF)>0.5))
            tar_mask_pred=x2ms_adapter.tensor_api.x2ms_float((x2ms_adapter.nn_functional.sigmoid(tar_BF)>0.5))
        elif self.opt.BF_loss_choose.lower()=='softmax':
            src_BF_softmax=x2ms_adapter.nn_functional.softmax(src_BF,dim=1)
            src_mask_pred=x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.argmax(src_BF_softmax,dim=1), 1))
            tar_BF_softmax=x2ms_adapter.nn_functional.softmax(tar_BF,dim=1)
            tar_mask_pred=x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.argmax(tar_BF_softmax,dim=1), 1))

        self.BF_acc_src.update(x2ms_adapter.tensor_api.item(x2ms_adapter.x2ms_sum(src_BF_gt[src_BF_gt>0]==src_mask_pred[src_BF_gt>0]))/x2ms_adapter.tensor_api.numel(src_BF_gt[src_BF_gt>0]),x2ms_adapter.tensor_api.numel(src_BF_gt[src_BF_gt>0]))
        self.BF_acc_tar.update(x2ms_adapter.tensor_api.item(x2ms_adapter.x2ms_sum(tar_BF_gt[tar_BF_gt>0]==tar_mask_pred[tar_BF_gt>0]))/x2ms_adapter.tensor_api.numel(tar_BF_gt[tar_BF_gt>0]),x2ms_adapter.tensor_api.numel(tar_BF_gt[tar_BF_gt>0]))

        for i in range(batch):

            sum_map_src=x2ms_adapter.tensor_api.item((x2ms_adapter.x2ms_sum(x2ms_adapter.tensor_api.detach(src_den_map[i]))/self.opt.gt_factor))
            self.mae_src.update(abs(sum_map_src-src_data[3][i]['gt_cnt']))
            self.mse_src.update((abs(sum_map_src-src_data[3][i]['gt_cnt'])**2))

            sum_map_tar=x2ms_adapter.tensor_api.item((x2ms_adapter.x2ms_sum(x2ms_adapter.tensor_api.detach(tar_den_map[i]))/self.opt.gt_factor))
            self.mae_tar.update(abs(sum_map_tar-tar_data[3][i]['gt_cnt']))
            self.mse_tar.update((abs(sum_map_tar-tar_data[3][i]['gt_cnt'])**2))

            
            if epoch<=self.opt.save_start_epochs or epoch%self.opt.vision_frequency==0:
                if step in vision_list and i==0:

                    video_name=os.path.basename(src_data[3][i]['dataset'])
                    img_name=src_data[3][i]['img_file']
                    save_root=os.path.join(self.opt.log_root_path,'train',str(epoch),str_prefix,src_data[3][i]['dataset'])
                    save_filename=os.path.join(save_root,video_name+'gt_point_',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'_cnt_'+str(src_data[3][i]['gt_cnt'])+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_keypoints_img(src_img[i],src_data[2][i], self.opt, save_filename)
                    save_filename=os.path.join(save_root,video_name+'out_point_',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'_cnt_'+str(sum_map_src)+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_keypoints_img(src_img[i],x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.detach(src_den_map[i])), self.opt, save_filename)

                    video_name=os.path.basename(tar_data[3][i]['dataset'])
                    img_name=tar_data[3][i]['img_file']
                    save_root=os.path.join(self.opt.log_root_path,'train',str(epoch),str_prefix,tar_data[3][i]['dataset'])
                    save_filename=os.path.join(save_root,video_name+'gt_point_',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'_cnt_'+str(tar_data[3][i]['gt_cnt'])+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_keypoints_img(tar_img[i],tar_data[2][i], self.opt, save_filename)
                    save_filename=os.path.join(save_root,video_name+'out_point_',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'_cnt_'+str(sum_map_tar)+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_keypoints_img(tar_img[i],x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.detach(tar_den_map[i])), self.opt, save_filename)


                    video_name=os.path.basename(tar_data[3][i]['dataset'])
                    img_name=tar_data[3][i]['img_file']

                    save_root=os.path.join(self.opt.log_root_path,'train',str(epoch),str_prefix,tar_data[3][i]['dataset'])

                    save_filename=os.path.join(save_root,video_name+'out_point_teacher',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'_cnt_'+str(x2ms_adapter.x2ms_sum(tar_teacher_den_pred[i])/self.opt.gt_factor)+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_keypoints_img(tar_img[i],x2ms_adapter.tensor_api.squeeze(tar_teacher_den_pred[i]), self.opt, save_filename)

                    save_filename=os.path.join(save_root,video_name+'out_point_teacher',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'_cnt_points_'+str(x2ms_adapter.x2ms_sum(tar_teacher_den_pred_gass[i])/self.opt.gt_factor)+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_keypoints_img(tar_img[i],x2ms_adapter.tensor_api.squeeze(tar_teacher_den_pred_gass[i]), self.opt, save_filename)

                    save_filename=os.path.join(save_root,video_name+'out_point_teacher',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'_cnt_mix_'+str(x2ms_adapter.x2ms_sum(tar_teacher_den_mix[i])/self.opt.gt_factor)+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_keypoints_img(tar_img[i],x2ms_adapter.tensor_api.squeeze(tar_teacher_den_mix[i]), self.opt, save_filename)

        res=self.get_train_res()

        return res
    

    def teacher_inference(self, data,search_time,epoch,global_step,step,vision_list):
        data=data[4:8]
        batch=len(data[0])
        den_maps=[]
        den_maps_gaussian=[]
        den_maps_mix=[]
        BFs=[]
        lds=[]
        for i in range(batch):
            c,h,w=x2ms_adapter.tensor_api.x2ms_size(data[0][i])
            x=self.backbone(x2ms_adapter.tensor_api.unsqueeze(data[0][i], 0))
            den_map=self.den_pred(x)
            BF=self.BF_cls(x)

            if self.opt.BF_loss_choose.lower()=='bce':
                BF = x2ms_adapter.tensor_api.x2ms_float((x2ms_adapter.sigmoid(x2ms_adapter.tensor_api.detach(BF)) > 0.5))

            if self.opt.BF_loss_choose.lower()=='softmax':
                BF = x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.detach(BF), dim=1)
                BF = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.argmax(BF, dim=1), 1))

            ld=self.Local_den_predictor(x,(int(x2ms_adapter.tensor_api.x2ms_size(x, -2)*8/self.opt.local_density_grid[0]),int(x2ms_adapter.tensor_api.x2ms_size(x, -1)*8/self.opt.local_density_grid[1])))

            den_map_detach=x2ms_adapter.tensor_api.detach(den_map)
            sum_map=x2ms_adapter.x2ms_sum(x2ms_adapter.tensor_api.detach(den_map))/self.opt.gt_factor
            
            self.mae_src_teacher.update(abs(sum_map-data[3][i]['gt_cnt']))
            # den_maps.append(den_map.detach())

            points_map,num_points=self.get_max_point(x2ms_adapter.tensor_api.detach(den_map))
            points_map=x2ms_adapter.tensor_api.x2ms_float(points_map)
            self.mae_src_teacher_points.update(abs(num_points-data[3][i]['gt_cnt']))
            points_gass_map=self.gen_map(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.unsqueeze(points_map, 0), 0))*self.opt.gt_factor
            # den_maps_gaussian.append(points_gass_map)

            ######
            local_den_map,local_gass_map,out_shape=self.get_local_for_teacher_inference(den_map_detach,points_gass_map)
            
            if self.opt.concat_way=='only_density_map':
                inp=x2ms_adapter.cat([local_den_map,local_gass_map],dim=1)
            elif self.opt.concat_way=='only_feature_map':
                inp=x
            elif self.opt.concat_way=='all':
                inp=x2ms_adapter.cat([x,local_den_map,local_gass_map],dim=1)
            mix_pred=self.mix_cls(inp,out_shape)

            mix_map=x2ms_adapter.zeros(points_gass_map.shape)
            mix_pred=x2ms_adapter.nn_functional.interpolate(mix_pred,size=mix_map.shape[-2:],mode='bilinear', align_corners=True)

            if self.opt.ada_mix_loss_choose.lower()=='bce' and self.opt.mix_mode=='soft':
                mix_pred=x2ms_adapter.sigmoid(x2ms_adapter.tensor_api.detach(mix_pred))
                mix_map=mix_pred*den_map+(1-mix_pred)*points_gass_map
            if self.opt.ada_mix_loss_choose.lower()=='bce' and self.opt.mix_mode=='hard':
                mix_pred = x2ms_adapter.tensor_api.x2ms_float((x2ms_adapter.sigmoid(x2ms_adapter.tensor_api.detach(mix_pred)) > 0.5))
                mix_map=mix_pred*den_map+(1-mix_pred)*points_gass_map

            if self.opt.ada_mix_loss_choose.lower()=='softmax' and self.opt.mix_mode=='soft':
                mix_pred = x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.detach(mix_pred), dim=1)
                mix_map=mix_pred[:,0:1,:,:]*points_gass_map+mix_pred[:,1:2,:,:]*den_map
            if self.opt.ada_mix_loss_choose.lower()=='softmax' and self.opt.mix_mode=='hard':
                mix_pred = x2ms_adapter.nn_functional.softmax(x2ms_adapter.tensor_api.detach(mix_pred), dim=1)
                mix_pred = x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.argmax(mix_pred, dim=1), 1))
                mix_map=mix_pred*den_map+(1-mix_pred)*points_gass_map

            sum_mix_map=x2ms_adapter.x2ms_sum(mix_map)/self.opt.gt_factor
            self.mae_src_teacher_mix.update(abs(sum_mix_map-data[3][i]['gt_cnt']))


            if 'HorizontallyFlip' in list(data[3][i].keys()):
                if data[3][i]['HorizontallyFlip']:
                    den_map_detach = x2ms_adapter.flip(den_map_detach, [3])
                    mix_map = x2ms_adapter.flip(mix_map, [3])
                    points_gass_map = x2ms_adapter.flip(points_gass_map, [3])
                    BF = x2ms_adapter.flip(BF, [3])
                    ld = x2ms_adapter.flip(ld, [3])

            if 'VerticallyFlip' in list(data[3][i].keys()):
                if data[3][i]['VerticallyFlip']:
                    den_map_detach = x2ms_adapter.flip(den_map_detach, [2])
                    mix_map = x2ms_adapter.flip(mix_map, [2])
                    points_gass_map = x2ms_adapter.flip(points_gass_map, [2])
                    BF = x2ms_adapter.flip(BF, [2])
                    ld = x2ms_adapter.flip(ld, [2])
            den_maps.append(den_map_detach)
            den_maps_gaussian.append(points_gass_map)
            den_maps_mix.append(mix_map)
            BFs.append(BF)
            lds.append(ld)

        return den_maps,den_maps_gaussian,den_maps_mix,BFs,lds,self.mae_src_teacher,self.mae_src_teacher_points,self.mae_src_teacher_mix

    def inference(self,data,search_time,epoch,global_step,step,vision_list,batch_num_each_epoch,logger):
        batch=len(data[0])
        for i in range(batch):
            data=list(data)
            data[0]=list(data[0])
            data[1]=list(data[1])
            data[2]=list(data[2])
            data[0][i]=mindspore.tensor(data[0][i].numpy())
            data[1][i]=mindspore.tensor(data[1][i].numpy())
            data[2][i]=mindspore.tensor(data[2][i].numpy())
            c,h,w=data[0][i].shape
            x=self.backbone(data[0][i].unsqueeze(0))
            den_map=self.den_pred(x)
            sum_map=mindspore.ops.sum(den_map[i])/self.opt.gt_factor
          
            self.mae.update(abs(sum_map-data[3][i]['gt_cnt']))
            self.mse.update((abs(sum_map-data[3][i]['gt_cnt'])**2))

            if 'WE' in self.opt.tar_dataset:
                scene=data[3][i]['img_file'][:6]
                if scene=='104207':
                    self.mae1.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='200608':
                    self.mae2.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='200702':
                    self.mae3.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='202201':
                    self.mae4.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='500717':
                    self.mae5.update(abs(sum_map-data[3][i]['gt_cnt']))
            if len(data)==5:
                scene=data[3][i]['img_file'][:6]
                mask=x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.from_numpy(data[4][i]))
                mask = x2ms_adapter.nn_functional.interpolate(x2ms_adapter.tensor_api.unsqueeze(x2ms_adapter.tensor_api.unsqueeze(mask, 0), 0),size=(h,w),mode=self.opt.img_interpolate_mode,align_corners = True)
                mask = x2ms_adapter.tensor_api.squeeze(mask)
                sum_map=x2ms_adapter.x2ms_sum(x2ms_adapter.tensor_api.detach(den_map[i])*mask)/self.opt.gt_factor
                if scene=='104207':
                    self.mae1_mask.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='200608':
                    self.mae2_mask.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='200702':
                    self.mae3_mask.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='202201':
                    self.mae4_mask.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='500717':
                    self.mae5_mask.update(abs(sum_map-data[3][i]['gt_cnt']))

                mask=x2ms_adapter.tensor_api.numpy(mask)
                k=np.ones((30,30),np.uint8)
                mask=cv2.dilate(mask,k,iterations=1)
                mask=x2ms_adapter.tensor_api.x2ms_float(x2ms_adapter.from_numpy(mask))
                sum_map=x2ms_adapter.x2ms_sum(x2ms_adapter.tensor_api.detach(den_map[i])*mask)/self.opt.gt_factor

                if scene=='104207':
                    self.mae1_mask_dilate.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='200608':
                    self.mae2_mask_dilate.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='200702':
                    self.mae3_mask_dilate.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='202201':
                    self.mae4_mask_dilate.update(abs(sum_map-data[3][i]['gt_cnt']))
                elif scene=='500717':
                    self.mae5_mask_dilate.update(abs(sum_map-data[3][i]['gt_cnt']))


            if epoch<=self.opt.save_start_epochs or epoch%self.opt.vision_frequency==0:
                if step in vision_list and i==0:
                    video_name=os.path.basename(data[3][i]['dataset'])
                    img_name=data[3][i]['img_file']

                    save_root=os.path.join(self.opt.log_root_path,'test',str(epoch),data[3][i]['dataset'])

                    save_filename=os.path.join(save_root,video_name+'gt_point_',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'_cnt_'+str(data[3][i]['gt_cnt'])+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_keypoints_img(data[0][i],data[2][i], self.opt, save_filename,img_alpha=0.0)
                    save_filename=os.path.join(save_root,video_name+'out_point_',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'_cnt_'+str(sum_map)+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_keypoints_img(data[0][i],x2ms_adapter.tensor_api.detach(den_map[0,0]), self.opt, save_filename,img_alpha=0.0)

                    save_filename=os.path.join(save_root,video_name+'input_img_',x2ms_adapter.tensor_api.split(img_name, '.')[0]+'.'+x2ms_adapter.tensor_api.split(img_name, '.')[1])
                    save_img_tensor(data[0][i],save_filename,self.opt)

        res=self.get_test_res()
        return res


class den_mse_loss(nn.Cell):
    def __init__(self,opt):
        super(den_mse_loss, self).__init__()
        self.opt=opt
        # k_size=3
        self.mseloss=loss_wrapper.MSELoss()
    def construct(self,src_pred_den_map,src_gt_map):
        batch=len(src_gt_map)
        if batch>1:
            src_gt_map=x2ms_adapter.stack(src_gt_map)
        else:
            src_gt_map=src_gt_map[0]

        den_loss=self.mseloss(x2ms_adapter.tensor_api.squeeze(src_pred_den_map),x2ms_adapter.tensor_api.squeeze(src_gt_map))

        return den_loss


class loss1(nn.Cell):
    def __init__(self,opt):
        super(loss1, self).__init__()
        self.opt=opt
        # k_size=3
        self.mseloss=nn.MSELoss()
        self.get_BF_label=get_den_level_label_1(opt.BF_grid,opt.BF_thd,opt)
        self.den_loss_trade_off_list=self.opt.den_loss_trade_off
        self.domain_loss_trade_off_list=self.opt.domain_loss_trade_off
        self.p_label_loss_trade_off_list=self.opt.p_label_loss_trade_off
        self.p_label_gass_loss_trade_off_list=self.opt.p_label_gass_loss_trade_off
        self.p_label_mix_loss_trade_off_list=self.opt.p_label_mix_loss_trade_off

        self.src_local_density_loss_trade_off_list=self.opt.src_local_density_loss_trade_off
        self.tar_local_density_loss_trade_off_list=self.opt.tar_local_density_loss_trade_off

        self.src_BF_cls_weight_list=self.opt.src_BF_cls_weight
        self.tar_BF_cls_weight_list=self.opt.tar_BF_cls_weight

        self.mix_loss_trade_off_list=self.opt.mix_loss_trade_off
        self.mask_den_loss_trade_off_list=self.opt.mask_den_loss_trade_off

        self.epoch_stage=self.opt.epoch_stage

        assert len(self.domain_loss_trade_off_list)==len(self.p_label_loss_trade_off_list)==len(self.p_label_gass_loss_trade_off_list)==len(self.p_label_mix_loss_trade_off_list)
        

    def construct(self,epoch,src_den_map,tar_den_map,src_D_pred,tar_D_pred,src_gt_map,
            den_plabel,gass_plabel,mix_plabel,src_BF,src_BF_gt,tar_BF,tar_BF_plabel,src_ld,src_ld_gt,tar_ld,tar_ld_plabel,mix_pred,mix_gt):

        batch=len(src_gt_map)
        if batch>1:
            src_gt_map=mindspore.ops.stack(src_gt_map)
        else:
            src_gt_map=src_gt_map[0]

        for i in range(len(self.epoch_stage)-1):
            if epoch>self.epoch_stage[i] and epoch<=self.epoch_stage[i+1]:
                index=i

        self.den_loss_trade_off=self.den_loss_trade_off_list[index]
        self.domain_loss_trade_off=self.domain_loss_trade_off_list[index]
        self.p_label_loss_trade_off=self.p_label_loss_trade_off_list[index]
        self.p_label_gass_loss_trade_off=self.p_label_gass_loss_trade_off_list[index]
        self.p_label_mix_loss_trade_off=self.p_label_mix_loss_trade_off_list[index]
        self.src_local_density_loss_trade_off=self.src_local_density_loss_trade_off_list[index]
        self.src_BF_cls_weight=self.src_BF_cls_weight_list[index]

        self.tar_local_density_loss_trade_off=self.tar_local_density_loss_trade_off_list[index]
        self.tar_BF_cls_weight=self.tar_BF_cls_weight_list[index]

        self.mix_loss_trade_off=self.mix_loss_trade_off_list[index]
        self.mask_den_loss_trade_off=self.mask_den_loss_trade_off_list[index]

        # src den loss
        if self.den_loss_trade_off==0.:
            den_loss=mindspore.ops.zeros(1)
        else:
            for i in range(batch):
                if i==0:
                    den_loss=self.mseloss(x2ms_adapter.tensor_api.squeeze(src_den_map[i]),x2ms_adapter.tensor_api.squeeze(src_gt_map[i]))
                else:
                    den_loss+=self.mseloss(x2ms_adapter.tensor_api.squeeze(src_den_map[i]),x2ms_adapter.tensor_api.squeeze(src_gt_map[i]))
            den_loss/=batch
            den_loss*=self.den_loss_trade_off

        ## BF loss
        if self.src_BF_cls_weight==0.:
            src_BF_loss=mindspore.ops.zeros(1)
        else:
            if self.opt.BF_loss_choose.lower()=='bce':
                src_BF_loss=mindspore.ops.binary_cross_entropy_with_logits(src_BF, src_BF_gt)
            elif self.opt.BF_loss_choose.lower()=='softmax':
                src_BF_loss=mindspore.ops.cross_entropy(src_BF, x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.long(src_BF_gt), 1))
            src_BF_loss*=self.src_BF_cls_weight

        if self.tar_BF_cls_weight==0.:
            tar_BF_loss=mindspore.ops.zeros(1)
        else:
            tar_BF_plabel=mindspore.ops.cat(tar_BF_plabel,0)
            if self.opt.BF_loss_choose.lower()=='bce':
                tar_BF_loss=mindspore.ops.binary_cross_entropy_with_logits(tar_BF, tar_BF_plabel)
            elif self.opt.BF_loss_choose.lower()=='softmax':
                tar_BF_loss=mindspore.ops.cross_entropy(tar_BF, x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.long(tar_BF_plabel), 1))
            tar_BF_loss*=self.tar_BF_cls_weight

        # domain loss
        if self.domain_loss_trade_off==0.:
            domain_loss=mindspore.ops.zeros(1)
            src_loss=mindspore.ops.zeros(1)
            tar_loss=mindspore.ops.zeros(1)
        else:
            for i in range(batch):
                slabels = mindspore.ops.ones(src_D_pred.shape()).to(mindspore.float32)
                tlabels = mindspore.ops.zeros(tar_D_pred[i].shape()).to(mindspore.float32)
                if i==0:
                    tar_loss=mindspore.ops.binary_cross_entropy_with_logits(tar_D_pred[i],tlabels)
                    src_loss=mindspore.ops.binary_cross_entropy_with_logits(src_D_pred[i],slabels[i])
                else:
                    tar_loss+=mindspore.ops.binary_cross_entropy_with_logits(tar_D_pred[i],tlabels)
                    src_loss+=mindspore.ops.binary_cross_entropy_with_logits(src_D_pred[i],slabels[i])
            tar_loss/=batch
            src_loss/=batch
            domain_loss=self.domain_loss_trade_off*(src_loss+tar_loss)

        if self.p_label_loss_trade_off==0:
            p_label_den_loss=mindspore.ops.zeros(1)
        else:
            for i in range(batch):
                if i==0:
                    p_label_den_loss=self.mseloss(tar_den_map[i].squeeze(),den_plabel[i].squeeze())
                else:
                    p_label_den_loss+=self.mseloss(tar_den_map[i].squeeze(),den_plabel[i].squeeze())
            p_label_den_loss/=batch
            p_label_den_loss*=self.p_label_loss_trade_off


        if self.p_label_gass_loss_trade_off==0:
            p_label_gass_den_loss=mindspore.ops.zeros(1)
        else:
            for i in range(batch):
                if i==0:
                    p_label_gass_den_loss=self.mseloss(tar_den_map[i].squeeze(),gass_plabel[i].squeeze())
                else:
                    p_label_gass_den_loss+=self.mseloss(tar_den_map[i].squeeze(),gass_plabel[i].squeeze())
            p_label_gass_den_loss/=batch
            p_label_gass_den_loss*=self.p_label_gass_loss_trade_off


        if self.p_label_mix_loss_trade_off==0:
            p_label_mix_den_loss=mindspore.ops.zeros(1)
        else:
            for i in range(batch):
                if i==0:
                    p_label_mix_den_loss=self.mseloss(x2ms_adapter.tensor_api.squeeze(tar_den_map[i]),x2ms_adapter.tensor_api.squeeze(mix_plabel[i]))
                else:
                    p_label_mix_den_loss+=self.mseloss(x2ms_adapter.tensor_api.squeeze(tar_den_map[i]),x2ms_adapter.tensor_api.squeeze(mix_plabel[i]))
            p_label_mix_den_loss/=batch
            p_label_mix_den_loss*=self.p_label_mix_loss_trade_off


        # local density
        if self.src_local_density_loss_trade_off==0.:
            src_ldl=mindspore.ops.zeros(1)
        else:
            src_ldl=self.mseloss(src_ld,src_ld_gt)
            src_ldl*=self.src_local_density_loss_trade_off

        if self.tar_local_density_loss_trade_off==0.:
            tar_ldl=mindspore.ops.zeros(1)
        else:
            tar_ld_plabel=mindspore.ops.cat(tar_ld_plabel,0)
            tar_ldl=self.mseloss(tar_ld,tar_ld_plabel)
            tar_ldl*=self.tar_local_density_loss_trade_off

        if self.mix_loss_trade_off==0.:
            mix_loss=mindspore.ops.zeros(1)
        else:
            if self.opt.ada_mix_loss_choose.lower()=='bce':
                mix_loss=mindspore.ops.binary_cross_entropy_with_logits(mix_pred, mix_gt)
            elif self.opt.ada_mix_loss_choose.lower()=='softmax':
                mix_loss=mindspore.ops.cross_entropy(mix_pred, x2ms_adapter.tensor_api.squeeze(x2ms_adapter.tensor_api.long(mix_gt), 1))
            mix_loss*=self.mix_loss_trade_off

        if self.mask_den_loss_trade_off==0.:
            mask_den_loss=mindspore.ops.zeros(1)
        elif self.opt.mask_den_choose=='p':
            mask_plabel=self.get_BF_label(den_plabel)
            mask_plabel=mindspore.ops.stack(mask_plabel,0)
            mask_plabel = mindspore.ops.interpolate(mask_plabel,size=den_plabel[0].shape[-2:],mode='nearest')
            if self.opt.BF_loss_choose.lower()=='bce':
                tar_mask_pred=x2ms_adapter.tensor_api.x2ms_float((mindspore.ops.sigmoid(tar_BF)>0.5))
            elif self.opt.BF_loss_choose.lower()=='softmax':
                tar_BF_softmax=mindspore.ops.softmax(tar_BF,axis=1)
                tar_mask_pred=mindspore.ops.argmax(tar_BF_softmax,dim=1).unsqueeze(1).float()
            tar_mask_pred = mindspore.ops.interpolate(tar_mask_pred,size=den_plabel[0].shape[-2:],mode='nearest')

            mask=mindspore.ops.zeros(tar_mask_pred.shape())
            mask[tar_mask_pred==mask_plabel]=1.0

            mask_den_loss=self.mseloss(mask*tar_den_map,mask*mindspore.ops.cat(den_plabel,0))
            mask_den_loss*=self.mask_den_loss_trade_off
        elif self.opt.mask_den_choose=='g':
            mask_plabel=self.get_BF_label(gass_plabel)
            mask_plabel=mindspore.ops.stack(mask_plabel,0)
            mask_plabel = mindspore.ops.interpolate(mask_plabel,size=gass_plabel[0].shape[-2:],mode='nearest')
            if self.opt.BF_loss_choose.lower()=='bce':
                tar_mask_pred=x2ms_adapter.tensor_api.x2ms_float((mindspore.ops.sigmoid(tar_BF)>0.5))
            elif self.opt.BF_loss_choose.lower()=='softmax':
                tar_BF_softmax=mindspore.ops.softmax(tar_BF,axis=1)
                tar_mask_pred=mindspore.ops.argmax(tar_BF_softmax,dim=1).unsqueeze(1).float()
            tar_mask_pred = mindspore.ops.interpolate(tar_mask_pred,size=gass_plabel[0].shape[-2:],mode='nearest')

            mask=mindspore.ops.zeros(tar_mask_pred.shape())
            mask[tar_mask_pred==mask_plabel]=1.0

            mask_den_loss=self.mseloss(mask*tar_den_map,mask*mindspore.ops.cat(gass_plabel,0))
            mask_den_loss*=self.mask_den_loss_trade_off

        elif self.opt.mask_den_choose=='m':
            mask_plabel=self.get_BF_label(mix_plabel)
            mask_plabel=mindspore.ops.stack(mask_plabel,0)
            mask_plabel = mindspore.ops.interpolate(mask_plabel,size=mix_plabel[0].shape[-2:],mode='nearest')
            if self.opt.BF_loss_choose.lower()=='bce':
                tar_mask_pred=x2ms_adapter.tensor_api.x2ms_float((mindspore.ops.sigmoid(tar_BF)>0.5))
            elif self.opt.BF_loss_choose.lower()=='softmax':
                tar_BF_softmax=mindspore.ops.softmax(tar_BF,axis=1)
                tar_mask_pred=mindspore.ops.argmax(tar_BF_softmax,dim=1).unsqueeze(1).float()
            tar_mask_pred = mindspore.ops.interpolate(tar_mask_pred,size=mix_plabel[0].shape[-2:],mode='nearest')

            mask=mindspore.ops.zeros(x2ms_adapter.tensor_api.x2ms_size(tar_mask_pred))
            mask[tar_mask_pred==mask_plabel]=1.0

            mask_den_loss=self.mseloss(mask*tar_den_map,mask*mindspore.ops.cat(mix_plabel,0))
            mask_den_loss*=self.mask_den_loss_trade_off
            
            
        loss=domain_loss+den_loss+p_label_den_loss+p_label_gass_den_loss+p_label_mix_den_loss+src_BF_loss+tar_BF_loss+src_ldl+tar_ldl+mix_loss+mask_den_loss
        return loss,src_loss,tar_loss,den_loss,p_label_den_loss,p_label_gass_den_loss,p_label_mix_den_loss,src_BF_loss,tar_BF_loss,src_ldl,tar_ldl,mix_loss,mask_den_loss

def init_weights(init_modules):
    for m in x2ms_adapter.nn_cell.modules(init_modules):
        # nn.init.kquerying_normal_(m, mode='fan_in', nonlinearity='relu')
        if isinstance(m, x2ms_nn.Conv2d):
            x2ms_adapter.nn_init.normal_(m.weight, std=0.001)
            for name, _ in x2ms_adapter.named_parameters(m):
                if name in ['bias']:
                    x2ms_adapter.nn_init.constant_(m.bias, 0)
        elif isinstance(m, x2ms_nn.BatchNorm2d):
            x2ms_adapter.nn_init.constant_(m.weight, 1)
            x2ms_adapter.nn_init.constant_(m.bias, 0)
        elif isinstance(m, x2ms_nn.ConvTranspose2d):
            x2ms_adapter.nn_init.normal_(m.weight, std=0.001)
            for name, _ in x2ms_adapter.named_parameters(m):
                if name in ['bias']:
                    x2ms_adapter.nn_init.constant_(m.bias, 0)


