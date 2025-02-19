# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Iterable, Tuple, Union

import torch
import torch.nn as nn
from silk.backbones.silk.coords import (
    CoordinateMappingComposer,
    CoordinateMappingProvider,
)

# from silk.backbones.abstract.shared_backbone_multiple_heads import (
#     SharedBackboneMultipleHeads_Pose,
# )
from silk.backbones.loftr.resnet_fpn import ResNetFPN_8_2
from silk.backbones.superpoint.magicpoint import (
    # Backbone as VGGBackbone,
    DetectorHead as VGGDetectorHead,
    MagicPoint,
)
from silk.backbones.superpoint.vgg import ParametricVGG as VGGBackbone
from silk.backbones.superpoint.superpoint import (
    DescriptorHead as VGGDescriptorHead,
    SuperPoint,
)
from silk.backbones.sfmlearner.sfmlearner import(
    PoseHead as VGGPoseHead,
    DispHead as VGGDispHead,
)
from silk.flow import AutoForward, Flow
from silk.models.superpoint_utils import get_dense_positions


def from_feature_coords_to_image_coords(model, desc_positions):
    if isinstance(desc_positions, tuple):
        return tuple(
            from_feature_coords_to_image_coords(
                model,
                dp,
            )
            for dp in desc_positions
        )
    coord_mapping = model.coordinate_mapping_composer.get("images", "raw_descriptors")
    desc_positions = torch.cat(
        [
            coord_mapping.reverse(desc_positions[..., :2]),
            desc_positions[..., 2:],
        ],
        dim=-1,
    )

    return desc_positions


def from_logit_coords_to_image_coords(model, logits):
    if isinstance(logits, tuple):
        return tuple(
            from_logit_coords_to_image_coords(
                model,
                dp,
            )
            for dp in logits
        )
    coord_mapping = model.coordinate_mapping_composer.get("images", "logits")
    logits = torch.cat(
        [
            coord_mapping.reverse(logits[..., :2]),
            logits[..., 2:],
        ],
        dim=-1,
    )

    return logits


class SiLKBase(AutoForward, torch.nn.Module):
    def __init__(
        self,
        backbone,
        input_name: Union[str, Tuple[str]]= "images",
        backbone_output_name: Union[str, Tuple[str]] = "features",
        default_outputs: Union[str, Iterable[str]]= ("normalized_descriptors", "logits"), #"pose6d"),
    ):
        torch.nn.Module.__init__(self)
        self.backbone = backbone
        self.detector_heads = set()
        self.descriptor_heads = set()
        self.pose_head = set()

        
        AutoForward.__init__(self, Flow(input_name), default_outputs=default_outputs)
        self.flow.define_transition(
            backbone_output_name,
            self.backbone,
            input_name,
        )
        # backbone input, output shapes
        # in vgg.py forward  torch.Size([2, 1, 164, 164])
        # in vgg.py forward  torch.Size([2, 128, 148, 148])


        self.coordinate_mappings_composer = CoordinateMappingComposer()
        assert isinstance(self.backbone, CoordinateMappingProvider)
        self.coordinate_mappings_composer.set(
            input_name,
            backbone_output_name,
            self.backbone.mappings(),
        )

    @property
    def coordinate_mapping_composer(self):
        return self.coordinate_mappings_composer

    def add_head(self, head_name, head, backbone_output_name=None):
        self.flow.define_transition(
            head_name,
            head,
            backbone_output_name
        )
        self.detector_heads.add(head_name)
        self.coordinate_mappings_composer.set(
            backbone_output_name,
            head_name,
            head.mappings(),
        )

    def add_pose_head(self, head_name, head, consec_backbone_output_name=None):
        # self.flow.define_transition(
        #     "cat_backbone_features", 
        #     lambda x: print(x.shape, x[0,:,:,:].shape, x[1,:,:,:].shape, torch.cat([x[0,:,:,:], x[1,:,:,:]], dim = 2).unsqueeze(0).shape),
        #     consec_backbone_output_name,
        # )# torch.Size([2, 128, 148, 148]) torch.Size([128, 148, 148]) torch.Size([128, 148, 148]) torch.Size([1, 128, 148, 296])


        self.flow.define_transition(
            "cat_backbone_features", 
            lambda x: torch.cat([x[0,:,:,:], x[1,:,:,:]], dim = 2).unsqueeze(0), 
            consec_backbone_output_name,
        )
        self.flow.define_transition(
            head_name, 
            head, 
            "cat_backbone_features"
        )
        self.pose_head.add(head_name) #"pose_head_features"
        self.coordinate_mappings_composer.set(
            consec_backbone_output_name,
            head_name,
            head.mappings(),
        )

class SiLKVGG(SiLKBase):
    def __init__(
        self,
        in_channels,
        *,
        feat_channels: int = 128,
        lat_channels: int = 128,
        desc_channels: int = 128,
        pose_channels:int = 256,
        use_batchnorm: bool = True,
        backbone=None,
        detector_head=None,
        descriptor_head=None,
        pose_head=None,
        disp_head=None,
        detection_threshold: float = 0.8,
        detection_top_k: int = 100,
        nms_dist=4,
        border_dist=4,
        descriptor_scale_factor: float = 1.0,
        learnable_descriptor_scale_factor: bool = False,
        normalize_descriptors: bool = True,
        padding: int = 1,
        **base_kwargs,
    ) -> None:
        detector_head = (
            VGGDetectorHead(
                in_channels=feat_channels,
                lat_channels=lat_channels,
                out_channels=1,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ).to("cuda:1")
            if detector_head is None
            else detector_head
        )

        descriptor_head = (
            VGGDescriptorHead(
                in_channels=feat_channels,
                out_channels=desc_channels,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ).to("cuda:1")
            if descriptor_head is None
            else descriptor_head
        )

        pose_head = (
            VGGPoseHead(
                in_channels=feat_channels, # backbone_output
                out_channels=pose_channels,
                padding=padding,
            ).to("cuda:1")
            if pose_head is None
            else pose_head
        )
        
        disp_head = (
            VGGDispHead(
                in_channels=feat_channels,
                lat_channels=lat_channels,
                out_channels=1,
                use_batchnorm=use_batchnorm,
                padding=padding,
            ).to("cuda:1")
            if disp_head is None
            else disp_head
        )
        
        SiLKBase.__init__(
            self, 
            backbone=backbone,
            **base_kwargs,
        )

        self.add_head("logits", detector_head, "features")

        self.add_head("raw_descriptors", descriptor_head, "features")
        
        # self.add_pose_head("pose_head_feature", pose_head, "features")

        # self.add_head("disp_maps", disp_head, "features")


        self.descriptor_scale_factor = nn.parameter.Parameter(
            torch.tensor(descriptor_scale_factor),
            requires_grad=learnable_descriptor_scale_factor,
        )
        self.normalize_descriptors = normalize_descriptors

        MagicPoint.add_detector_head_post_processing(
            self.flow,
            "logits",
            prefix="",
            cell_size=1,
            detection_threshold=detection_threshold,
            detection_top_k=detection_top_k,
            nms_dist=nms_dist,
            border_dist=border_dist,
        )
        SiLKVGG.add_descriptor_head_post_processing(
            self.flow,
            input_name="images",
            descriptor_head_output_name="raw_descriptors",
            prefix="",
            scale_factor=self.descriptor_scale_factor,
            normalize_descriptors=normalize_descriptors,
        )
        # SiLKVGG.add_pose_head_post_processing(
        #     self.flow,
        #     pose_head_output_name="pose_head_feature",
        #     pose_name="pose6d",
        #     prefix="",
        # )
        # SiLKVGG.add_disp_head_post_processing(
        #     self.flow,
        #     disp_head_output_name="disp_maps",
        #     name = "depth_maps",
        # )



    @staticmethod
    def add_pose_head_post_processing(
        flow: Flow,
        pose_head_output_name: str = "pose_head_feature",
        pose_name:str = "pose6d",
        prefix: str = "",
        scale_factor: float = 1.4,
    ):
        flow.define_transition(
            pose_name,
            VGGPoseHead.make_6d_pose,
            pose_head_output_name,
        )
    
    
    @staticmethod
    def add_disp_head_post_processing(
        flow: Flow,
        disp_head_output_name: str = "disp_maps",
        name:str = "depth_maps",
    ):
        flow.define_transition(
            name,
            VGGDispHead.make_depth_maps,
            disp_head_output_name,
        )

    @staticmethod
    def add_descriptor_head_post_processing(
        flow: Flow,
        input_name: str, # = "images",
        descriptor_head_output_name: str = "raw_descriptors",
        positions_name: str = "positions",
        prefix: str = "superpoint.",
        scale_factor: float = 1.0,
        normalize_descriptors: bool = True,
    ):
        flow.define_transition(
            f"{prefix}normalized_descriptors",
            partial(
                SuperPoint.normalize_descriptors,
                scale_factor=scale_factor,
                normalize=normalize_descriptors,
            ),
            descriptor_head_output_name,
        )
        flow.define_transition(
            f"{prefix}dense_descriptors",
            SiLKVGG.get_dense_descriptors,
            f"{prefix}normalized_descriptors",
        )
        flow.define_transition(f"{prefix}image_size", SuperPoint.image_size, input_name)
        flow.define_transition(
            f"{prefix}sparse_descriptors",
            partial(
                SiLKVGG.sparsify_descriptors,
                scale_factor=scale_factor,
                normalize_descriptors=normalize_descriptors,
            ),
            descriptor_head_output_name,
            f"positions",
        )
        flow.define_transition(
            f"{prefix}sparse_positions",
            lambda x: x,
            f"positions",
        )
        flow.define_transition(
            f"{prefix}dense_positions",
            SiLKVGG.get_dense_positions,
            f"probability",
        )

    @staticmethod
    def get_dense_positions(probability):
        batch_size = probability.shape[0]
        device = probability.device
        dense_positions = get_dense_positions(
            probability.shape[2],
            probability.shape[3],
            device,
            batch_size=batch_size,
        )

        dense_probability = probability.reshape(probability.shape[0], -1, 1)
        dense_positions = torch.cat((dense_positions, dense_probability), dim=2)

        return dense_positions

    @staticmethod
    def get_dense_descriptors(normalized_descriptors):
        dense_descriptors = normalized_descriptors.reshape(
            normalized_descriptors.shape[0],
            normalized_descriptors.shape[1],
            -1,
        )
        dense_descriptors = dense_descriptors.permute(0, 2, 1)
        return dense_descriptors

    @staticmethod
    def sparsify_descriptors(
        raw_descriptors,
        positions,
        scale_factor: float = 1.0,
        normalize_descriptors: bool = True,
    ):
        sparse_descriptors = []
        for i, pos in enumerate(positions):
            pos = pos[:, :2]
            pos = pos.floor().long()

            descriptors = raw_descriptors[i, :, pos[:, 0], pos[:, 1]].T

            # L2 normalize the descriptors
            descriptors = SuperPoint.normalize_descriptors(
                descriptors,
                scale_factor,
                normalize_descriptors,
            )

            sparse_descriptors.append(descriptors)
        return tuple(sparse_descriptors)

