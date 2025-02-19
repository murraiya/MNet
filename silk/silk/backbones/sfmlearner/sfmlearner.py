# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from silk.backbones.silk.coords import (
    CoordinateMappingProvider,
    mapping_from_torch_module,
)
from silk.backbones.superpoint.magicpoint import MagicPoint, vgg_block
from silk.flow import AutoForward, Flow
from torchvision.transforms.functional import InterpolationMode, resize



def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )
    
    
def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )


class DispHead(torch.nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels: int = 128,
        lat_channels: int = 256,
        out_channels: int = 1,
        use_batchnorm: bool = True,
        padding: int = 1,
        detach: bool = False,
        alpha = 10, 
        beta = 0.01,
    ) -> None:
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}
        self.alpha = alpha
        self.beta = beta
        
        self._detach = detach

        self._detH1 = vgg_block(
            in_channels,
            lat_channels,
            3,
            use_batchnorm=use_batchnorm,
            padding=padding,
        )

        if use_batchnorm:
            # no relu (bc last layer) - option to have batchnorm or not
            self._detH2 = nn.Sequential(
                nn.Conv2d(lat_channels, out_channels, 1, padding=0),
                nn.BatchNorm2d(out_channels),
            )
        else:
            # if no batch norm
            self._detH2 = nn.Sequential(
                nn.Conv2d(lat_channels, out_channels, 1, padding=0),
            )

        # self.predict_disp = predict_disp(out_channels)

    def mappings(self):
        mapping = mapping_from_torch_module(self._detH1)
        mapping = mapping + mapping_from_torch_module(self._detH2)

        return mapping

    def forward(self, x: torch.Tensor):
        if self._detach:
            x = x.detach()

        x = self._detH1(x)
        x = self.alpha * self._detH2(x) + self.beta
        print("in disp head ", x.shape)
        return x


    @staticmethod
    def make_depth_maps(disp_head_output):
        depth = [1/disp for disp in disp_head_output]
        # print("make6d shape", pose.shape)
        # print("final pose shape", pose.shape)
        return depth




class PoseHead(torch.nn.Module, CoordinateMappingProvider):
    def __init__(
        self,
        in_channels: int = 128,
        out_channels: int = 256,
        use_batchnorm: bool = True,
        padding: int = 1,
    ) -> None:
        torch.nn.Module.__init__(self)
        CoordinateMappingProvider.__init__(self)

        assert padding in {0, 1}

        # self._poseH1 = conv(in_channels, in_channels, kernel_size=5)
        # self._poseH2 = conv(in_channels, out_channels)
        # self.pose_pred = nn.Conv2d(out_channels, 6, kernel_size=1, padding=0)
        # pose net (decoder)
        self._poseH1 = vgg_block(
            in_channels,
            in_channels, 
            5,
            use_batchnorm=use_batchnorm,
            padding=padding,
        )

        self._poseH2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, padding=0),
        )
        self._pose_pred = nn.Conv2d(out_channels, 6, kernel_size=1, padding=0)


    def mappings(self):
        mapping = mapping_from_torch_module(self._poseH1)
        mapping = mapping + mapping_from_torch_module(self._poseH2)
        mapping = mapping + mapping_from_torch_module(self._pose_pred)

        return mapping

    def forward(self, x: torch.Tensor):
        # print("pose input shape",x.shape)
        x = self._poseH1(x)
        x = self._poseH2(x)
        # print("pose mid shape", x.shape)
        x = self._pose_pred(x)
        # print("pose output shape",x.shape)

        return x

    
    @staticmethod
    def make_6d_pose(pose_head_ouput, scale_factor=1.0):
        pose = pose_head_ouput.mean(3).mean(2)
        # print("make6d shape", pose.shape)
        pose = 0.01 * pose.view(pose.size(0), 1, 6)
        # print("final pose shape", pose.shape)
        return pose

