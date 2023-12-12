import math
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from op import FusedLeakyReLU

from veri3d_deepfashion import VoxelHuman as VeRi3D_DEEPFASHION_MODEL

class VolumeRenderDiscConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, activate=False):
        super(VolumeRenderDiscConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, bias=bias and not activate)

        self.activate = activate
        if self.activate:
            self.activation = FusedLeakyReLU(out_channels, bias=bias, scale=1)
            bias_init_coef = np.sqrt(1 / (in_channels * kernel_size * kernel_size))
            nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)


    def forward(self, input):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: (N,C_out,H_out,W_out)
        :return: Conv2d + activation Result
        """
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class AddCoords(nn.Module):
    def __init__(self):
        super(AddCoords, self).__init__()

    def forward(self, input_tensor):
        """
        :param input_tensor: shape (N, C_in, H, W)
        :return:
        """
        batch_size_shape, channel_in_shape, dim_y, dim_x = input_tensor.shape
        xx_channel = torch.arange(dim_x, dtype=torch.float32, device=input_tensor.device).repeat(1,1,dim_y,1)
        yy_channel = torch.arange(dim_y, dtype=torch.float32, device=input_tensor.device).repeat(1,1,dim_x,1).transpose(2,3)

        xx_channel = xx_channel / (dim_x - 1)
        yy_channel = yy_channel / (dim_y - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size_shape, 1, 1, 1)
        yy_channel = yy_channel.repeat(batch_size_shape, 1, 1, 1)
        out = torch.cat([input_tensor, yy_channel, xx_channel], dim=1)

        return out


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True):
        super(CoordConv2d, self).__init__()

        self.addcoords = AddCoords()
        # self.conv = nn.Conv2d(in_channels, out_channels,
        self.conv = nn.Conv2d(in_channels + 2, out_channels,
                              kernel_size, stride=stride, padding=padding, bias=bias)

    def forward(self, input_tensor):
        """
        input_tensor_shape: (N, C_in,H,W)
        output_tensor_shape: N,C_out,H_out,W_out）
        :return: CoordConv2d Result
        """
        out = self.addcoords(input_tensor)
        # out = input_tensor
        out = self.conv(out)

        return out


class CoordConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, bias=True, activate=True):
        super(CoordConvLayer, self).__init__()
        layers = []
        stride = 1
        self.activate = activate
        self.padding = kernel_size // 2 if kernel_size > 2 else 0

        self.conv = CoordConv2d(in_channel, out_channel, kernel_size,
                                padding=self.padding, stride=stride,
                                bias=bias and not activate)

        if activate:
            self.activation = FusedLeakyReLU(out_channel, bias=bias, scale=1)

        bias_init_coef = np.sqrt(1 / (in_channel * kernel_size * kernel_size))
        nn.init.uniform_(self.activation.bias, a=-bias_init_coef, b=bias_init_coef)

    def forward(self, input):
        out = self.conv(input)
        if self.activate:
            out = self.activation(out)

        return out


class VolumeRenderResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = CoordConvLayer(in_channel, out_channel, 3)
        self.conv2 = CoordConvLayer(out_channel, out_channel, 3)
        self.pooling = nn.AvgPool2d(2)
        self.downsample = nn.AvgPool2d(2)
        if out_channel != in_channel:
            self.skip = VolumeRenderDiscConv2d(in_channel, out_channel, 1)
        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)
        out = self.pooling(out)

        downsample_in = self.downsample(input)
        if self.skip != None:
            skip_in = self.skip(downsample_in)
        else:
            skip_in = downsample_in

        out = (out + skip_in) / math.sqrt(2)

        return out


class VolumeRenderDiscriminator(nn.Module):
    def __init__(self, opt):
        super().__init__()
        init_size = opt.renderer_spatial_output_dim
        if isinstance(init_size, list):
            if init_size[0] == init_size[1]:
                self.multiplier = 4
            else:
                self.multiplier = 8
            init_size = init_size[1]
        final_out_channel = 1
        channels = {
            2: 512,
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 512//2,
            128: 256//2,
            256: 128//2,
            512: 64//2
        }
        self.input_dim = 3
        self.init_size = init_size

        convs = [VolumeRenderDiscConv2d(self.input_dim, channels[self.init_size], 1, activate=True)]

        log_size = int(math.log(self.init_size, 2))

        in_channel = channels[self.init_size]

        for i in range(log_size-1, 0, -1):
            out_channel = channels[2 ** i]

            convs.append(VolumeRenderResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        # self.final_conv = VolumeRenderDiscConv2d(in_channel, final_out_channel, 2)
        self.final_conv = torch.nn.Linear(in_channel * self.multiplier, final_out_channel)
        self.in_channel = in_channel


    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out.reshape(-1, self.in_channel * self.multiplier))
        # out = out.mean(-2)
        gan_preds = out[:,0:1]
        gan_preds = gan_preds.view(-1, 1)
        pose_pred = None

        return gan_preds, pose_pred

class VoxelHumanGenerator(nn.Module):
    def __init__(self, model_opt, renderer_opt, ema=False, full_pipeline=True):
        super().__init__()
        self.style_dim = model_opt.style_dim
        self.full_pipeline = full_pipeline
        self.model_opt = model_opt

        if ema or 'is_test' in model_opt.keys():
            self.is_train = False
        else:
            self.is_train = True

        # volume renderer
        if "AIST++" in renderer_opt.dataset:
            model_opt.smpl_gender = 'male'
            print(model_opt.smpl_gender)
        smpl_cfgs = {
            'model_folder': model_opt.smpl_model_folder,
            'model_type': 'smpl',
            'gender': model_opt.smpl_gender,
            'num_betas': 10
        }
        VoxHuman_Class = VeRi3D_DEEPFASHION_MODEL
        self.renderer = VoxHuman_Class(renderer_opt, smpl_cfgs, out_im_res=tuple(model_opt.renderer_spatial_output_dim), style_dim=self.style_dim)

        if self.full_pipeline:
            raise NotImplementedError

    def get_vertices_feature(self,styles):
        vertices_feature = self.renderer.get_vertices_feature(styles=styles[0])
        return vertices_feature

    def forward(self, styles, cam_poses, focals, beta, theta, trans,
                truncation=1, return_mask=False, inv_Ks=None, input_vertices_feature=None,
                vertices=None, obv2cnl_tfm=None, tpose_vertices=None):
        latent = styles
 

        if inv_Ks is None:
            thumb_rgb, mask, xyz, smpl_verts, rgba_map = self.renderer(cam_poses, focals, beta, theta, trans, styles=latent[0], \
                                                                            vertices_feature = input_vertices_feature,
                                                                            truncation = truncation, vertices=vertices, obv2cnl_tfm=obv2cnl_tfm, tpose_vertices=tpose_vertices)
        else:
            thumb_rgb, mask, xyz, smpl_verts, rgba_map = self.renderer(cam_poses, focals, beta, theta, trans, styles=latent[0], \
                                                                            inv_Ks=inv_Ks, vertices_feature = input_vertices_feature,
                                                                            truncation = truncation, vertices=vertices, obv2cnl_tfm=obv2cnl_tfm, tpose_vertices=tpose_vertices)


        if self.full_pipeline:
            raise NotImplementedError
        else:
            rgb = None

        out = (rgb, thumb_rgb)
        if return_mask:
            out += (mask,)
        out += (rgba_map, )
        out += (smpl_verts, )

        return out 
