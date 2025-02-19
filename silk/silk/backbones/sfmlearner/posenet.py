# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_

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

class PoseNet(torch.nn.Module, CoordinateMappingProvider):

    def __init__(
        self,
        in_channels,
        pose_net = None,
        pose_net_output_names:str = None,
        use_batchnorm: bool = True,
        padding: int = 1,
        detach: bool = False,
        nb_ref_imgs=1, 
    ) -> None:
        CoordinateMappingProvider.__init__(self)
        torch.nn.Module.__init__(self)

        self._detach = detach
        self.nb_ref_imgs = nb_ref_imgs

        self.conv1 = conv(in_channels*(1+self.nb_ref_imgs), 16, kernel_size=7)
        self.conv2 = conv(16, 32, kernel_size=5)
        self.conv3 = conv(32, 64)
        self.conv4 = conv(64, 128)
        self.conv5 = conv(128, 256)
        self.conv6 = conv(256, 256)
        self.conv7 = conv(256, 256)
        self.pose_pred = nn.Conv2d(256, 6*self.nb_ref_imgs, kernel_size=1, padding=0)

    def mappings(self):
        mapping = mapping_from_torch_module(self._desH1)
        mapping = mapping + mapping_from_torch_module(self._desH2)
        return mapping

    def mappings(self):
        mapping = mapping_from_torch_module(self.conv1)
        mapping = mapping + mapping_from_torch_module(self.conv2)
        mapping = mapping + mapping_from_torch_module(self.conv3)
        mapping = mapping + mapping_from_torch_module(self.conv4)
        mapping = mapping + mapping_from_torch_module(self.conv5)
        mapping = mapping + mapping_from_torch_module(self.conv6)
        
        return mapping
   
    def forward(self, x: torch.Tensor):

        input = torch.cat(input, 1)
        o1 = self.conv1(x)
        o2 = self.conv2(o1)
        o3 = self.conv3(o2)
        o4 = self.conv4(o3)
        o5 = self.conv5(o4)
        o6 = self.conv6(o5)
        o7 = self.conv7(o6)

        pose = self.pose_pred(o7)
        pose = pose.mean(3).mean(2)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6)

        return pose







class PoseNet(AutoForward, torch.nn.Module):
  
    def __init__(
        self,
        *,
        in_channels,
        use_batchnorm: bool = True,
        input_name: str = "two_backbone_features",
        pose_net=None,
        pose_net_output_name="pose6d",
        default_outputs=("relative_pose"),
        **sfmlearner_kwargs,
    ):
        """
        Parameters
        ----------
        use_batchnorm : bool, optional
            Specify if the model uses batch normalization, by default True
        """
        torch.nn.Module.__init__(self)
        self.posenet = PoseNet(
            self,
            in_channels = in_channels,
            nb_ref_imgs=1,
            **sfmlearner_kwargs,
        )

        AutoForward.__init__(self, self.SfMLearner.flow, default_outputs)

        self.posenet.add_pose_net(
            pose_net_output_name,
            (
                PoseNet(
                    in_channels=128, 
                )
                if pose_net is None
                else pose_net
            ),
        )

        PoseNet.add_pose_net_post_processing(
            self.flow,
            input_name=input_name,
            pose_net_output_name="pose6d",
            prefix="",
        )

    @staticmethod
    def add_pose_net_post_processing(
        flow: Flow,
        input_name: str = "two_backbone_features",
        pose_net_output_name: str = "pose6d",
        prefix: str = "posenet.",
    ):
        flow.define_transition(
            f"{prefix}pose6d",
            logits_to_prob,
            pose_net_output_name,
        )




class ParametricVGG(torch.nn.Module, CoordinateMappingProvider):
    DEFAULT_NORMALIZATION_FN = torch.nn.Identity()

    def mappings(self):
        return mapping

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        
        return x
