# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Dict, Optional, Union
from PIL import Image
from torchvision.utils import save_image



import pytorch_lightning as pl
import torch
import numpy as np
import skimage.io as io
from silk.backbones.loftr.positional_encoding import PositionEncodingSine
from silk.backbones.silk.silk import SiLKBase as BackboneBase
from silk.config.core import ensure_is_instance
from silk.config.optimizer import Spec
from silk.cv.homography import HomographicSampler
from silk.flow import AutoForward, Flow
from silk.losses.info_nce import (
    keep_mutual_correspondences_only,
    positions_to_unidirectional_correspondence,
)
from silk.losses.sfmlearner.sfm_loss import pose_loss
from silk.matching.mnn import (
    compute_dist,
    double_softmax_distance,
    match_descriptors,
    mutual_nearest_neighbor,
)
from silk.models.abstract import OptimizersHandler, StateDictRedirect
from silk.transforms.abstract import MixedModuleDict, NamedContext, Transform
from silk.cv.homography import HomographicSampler
from silk.transforms.tensor import NormalizeRange
from torchvision.transforms import Grayscale

_DEBUG_MODE_ENABLED = True

# @torch.no_grad()
def matcher(
    postprocessing="none",
    threshold=1.0,
    temperature=0.1,
    return_distances=False,
):
    if postprocessing == "none" or postprocessing == "mnn":
        return partial(mutual_nearest_neighbor, return_distances=return_distances)
    elif postprocessing == "ratio-test":
        return partial(
            mutual_nearest_neighbor,
            match_fn=partial(match_descriptors, max_ratio=threshold),
            distance_fn=partial(compute_dist, dist_type="cosine"),
            return_distances=return_distances,
        )
    elif postprocessing == "double-softmax":
        return partial(
            mutual_nearest_neighbor,
            match_fn=partial(match_descriptors, max_distance=threshold),
            distance_fn=partial(double_softmax_distance, temperature=temperature),
            return_distances=return_distances,
        )

    raise RuntimeError(f"postprocessing {postprocessing} is invalid")


class SiLKBase(
    OptimizersHandler,
    AutoForward,
    StateDictRedirect,
    pl.LightningModule,
):
    def __init__(
        self,
        model,
        loss,
        optimizer_spec: Optional[Spec] = None,
        image_aug_transform: Optional[Transform] = None,
        contextualizer: Optional[torch.nn.Module] = None,
        ghost_similarity: Optional[float] = None,
        learn_ghost_similarity: bool = False,
        feature_downsampling_mode: str = "scale",
        greyscale_input: bool = True,
        **kwargs,
    ):
        pl.LightningModule.__init__(self, **kwargs)
        OptimizersHandler.__init__(self, optimizer_spec)  # see below

        assert isinstance(model, BackboneBase)

        self._feature_downsampling_mode = feature_downsampling_mode
        self._greyscale_input = greyscale_input

        if ghost_similarity is not None:
            self._ghost_sim = torch.nn.parameter.Parameter(
                torch.tensor(ghost_similarity),
                requires_grad=learn_ghost_similarity,
            )
        else:
            self._ghost_sim = None

        ghost_sim_module = torch.nn.Module()
        ghost_sim_module.ghost_sim = self._ghost_sim

        state = MixedModuleDict(
            {
                "model": model,
                "contextualizer": contextualizer,
                "ghost_similarity": ghost_sim_module,
            }
        )

        StateDictRedirect.__init__(self, state)
        AutoForward.__init__(self, Flow("batch", "use_image_aug"), "loss")

        self._loss = loss
        self._model = model 
        #silk.backbones.silk.silk.SiLKVGG (backbone = silk.backbones.superpoint.vgg.ParametricVGG)
        self._contextualizer = contextualizer
        if contextualizer:
            self._pe = PositionEncodingSine(256, max_shape=(512, 512))
        self._image_aug_transform = image_aug_transform

    @property
    def coordinate_mapping_composer(self):
        return self._model.coordinate_mapping_composer

    def _grayify(self, images):

        if self._greyscale_input:
            images = Grayscale(num_output_channels=1)(images)
            return NormalizeRange(0.0, 255.0, 0.0, 1.0)(images)
        return images


    
    def _init_loss_flow(
        self,
        images_input_name,
        corr_fn,
        depth_map_1,
        depth_map_2,
        pose_gt_forward,
        pose_gt_backward, 
        intrinsics,
        *corr_args,
        **corr_kwargs,
    ):
        self.flow.define_transition(
            "augmented_images",
            self._aug_images,
            images_input_name,
            "use_image_aug",
        )
        self.flow.define_transition(
            f"gray_images",
            self._grayify,
            f"augmented_images",
        )
        self.flow.define_transition(
            ("descriptors", "logits", "pose6d"),
            self._model.forward,
            # images="gray_images",
            # outputs=Flow.Constant(("normalized_descriptors", "logits")),
            "gray_images",
        )
        self.flow.define_transition(
            "descriptors_shape",
            lambda x: x.shape,
            "descriptors",
        )
        self.flow.define_transition(
            ("corr_forward", "corr_backward"),
            corr_fn,
            "descriptors",
            "image_shape",
            pose_gt_forward,
            pose_gt_backward,
            intrinsics,
            depth_map_1,
            depth_map_2,
        )
        self.flow.define_transition(
            ("logits_0", "logits_1"),
            self._split_logits,
            "logits",
        )
        self.flow.define_transition(
            ("descriptors_0", "descriptors_1"),
            self._split_descriptors,
            "descriptors",
        )
        self.flow.define_transition(
            ("acontextual_descriptor_loss", "keypoint_loss", "precision", "recall"),
            self._loss,
            "descriptors_0",
            "descriptors_1",
            "corr_forward",
            "corr_backward",
            "logits_0",
            "logits_1",
            Flow.Constant(self._ghost_sim),
        )
        self.flow.define_transition(
            "pose_loss",
            pose_loss,
            "intrinsics",
            "pose6d",
            "logits_0",
            "logits_1",
        )
        self.flow.define_transition(
            ("contextual_descriptor_0", "contextual_descriptor_1"),
            self._contextualize,
            "descriptors_0",
            "descriptors_1",
            "descriptors_shape",
        )
        self.flow.define_transition(
            "contextual_descriptor_loss",
            self._contextual_loss,
            "contextual_descriptor_0",
            "contextual_descriptor_1",
            "corr_forward",
            "corr_backward",
            "logits_0",
            "logits_1",
        )
        self._loss_fn = self.flow.with_outputs(
            (
                # "pose_loss",
                "contextual_descriptor_loss",
                "acontextual_descriptor_loss",
                "keypoint_loss",
                "precision",
                "recall",
            )
        )

    @property
    def model(self):
        return self._model

    def model_forward_flow(self, *args, **kwargs):
        return self._model.forward_flow(*args, **kwargs)

    def _apply_pe(self, descriptors_0, descriptors_1, descriptors_shape):
        if not self._pe:
            return descriptors_0, descriptors_1
        _0 = torch.zeros((1,) + descriptors_shape[1:], device=descriptors_0.device)
        pe = self._pe(_0)
        pe = self._img_to_flat(pe)
        pe = pe * self.model.descriptor_scale_factor

        return descriptors_0 + pe, descriptors_1 + pe

    def _contextualize(self, descriptors_0, descriptors_1, descriptors_shape=None):
        if self._contextualizer is None:
            return descriptors_0, descriptors_1

        spatial_shape = False
        if not descriptors_shape:
            spatial_shape = True
            assert descriptors_0.ndim == 4
            assert descriptors_1.ndim == 4

            descriptors_shape = descriptors_0.shape
            descriptors_0 = self._img_to_flat(descriptors_0)
            descriptors_1 = self._img_to_flat(descriptors_1)

        assert descriptors_0.ndim == 3
        assert descriptors_1.ndim == 3

        descriptors_0 = descriptors_0.detach()
        descriptors_1 = descriptors_1.detach()

        descriptors_0, descriptors_1 = self._apply_pe(
            descriptors_0, descriptors_1, descriptors_shape
        )

        descriptors_0, descriptors_1 = self._contextualizer(
            descriptors_0, descriptors_1
        )

        if spatial_shape:
            descriptors_0 = self._flat_to_img(descriptors_0, descriptors_shape)
            descriptors_1 = self._flat_to_img(descriptors_1, descriptors_shape)

        return descriptors_0, descriptors_1

    def _contextual_loss(
        self,
        descriptors_0,
        descriptors_1,
        corr_forward,
        corr_backward,
        logits_0,
        logits_1,
    ):
        if self._contextualizer is None:
            return 0.0

        logits_0 = logits_0.detach()
        logits_1 = logits_1.detach()

        desc_loss, _, _, _ = self._loss(
            descriptors_0,
            descriptors_1,
            corr_forward,
            corr_backward,
            logits_0,
            logits_1,
        )

        return desc_loss

    def _aug_images(self, images, use_image_aug):
        if use_image_aug:
            images = self._image_aug_transform(images)
        return images

    def _split_descriptors(self, descriptors):
        desc_0 = SiLKBase._img_to_flat(descriptors[0::2])
        desc_1 = SiLKBase._img_to_flat(descriptors[1::2])
        return desc_0, desc_1

    def _split_logits(self, logits):
        logits_0 = SiLKBase._img_to_flat(logits[0::2]).squeeze(-1)
        logits_1 = SiLKBase._img_to_flat(logits[1::2]).squeeze(-1)
        return logits_0, logits_1

    @staticmethod
    def _img_to_flat(x):
        # x : BxCxHxW
        batch_size = x.shape[0]
        channels = x.shape[1]
        x = x.reshape(batch_size, channels, -1)
        x = x.permute(0, 2, 1)
        return x

    @staticmethod
    def _flat_to_img(x, shape):
        # x : BxNxC
        assert len(shape) == 4
        assert shape[0] == x.shape[0]
        assert shape[1] == x.shape[2]

        x = x.permute(0, 2, 1)
        x = x.reshape(shape)
        return x

    def _total_loss(self, mode, batch, use_image_aug: bool):
        ctx_desc_loss, actx_desc_loss, keypt_loss, precision, recall  = \
        self._loss_fn(
            batch, use_image_aug
        )

        f1 = (2 * precision * recall) / (precision + recall)

        loss = ctx_desc_loss + actx_desc_loss + keypt_loss

        self.log(f"{mode}.total.loss", loss)
        # self.log(f"{mode}.pose.loss", pose_loss)
        self.log(f"{mode}.contextual.descriptors.loss", ctx_desc_loss)
        self.log(f"{mode}.acontextual.descriptors.loss", actx_desc_loss)
        self.log(f"{mode}.keypoints.loss", keypt_loss)
        self.log(f"{mode}.precision", precision)
        self.log(f"{mode}.recall", recall)
        self.log(f"{mode}.f1", f1)
        if (self._ghost_sim is not None) and (mode == "train"):
            self.log("ghost.sim", self._ghost_sim)

        return loss

    def training_step(self, batch, batch_idx):
        return self._total_loss(
            "train",
            batch,
            use_image_aug=True,
        )

    def validation_step(self, batch, batch_idx):
        return self._total_loss(
            "val",
            batch,
            use_image_aug=False,
        )


class SiLKRandomHomographies(SiLKBase):
    def __init__(
        self,
        model,
        loss,
        optimizer_spec: Union[Spec, None] = None,
        image_aug_transform: Union[Transform, None] = None,
        training_random_homography_kwargs: Union[Dict[str, Any], None] = None,
        **kwargs,
    ):
        SiLKBase.__init__(
            self,
            model,
            loss,
            optimizer_spec,
            image_aug_transform,
            **kwargs,
        )

        # homographic sampler arguments
        self._training_random_homography_kwargs = (
            {}
            if training_random_homography_kwargs is None
            else training_random_homography_kwargs
        )
        self.flow.define_transition(
            ("images_1", "images_2", "image_shape", "pose_gt_forward", "pose_gt_backward", "intrinsics", "depth_map_1", "depth_map_2"),
            self._check_batch, 
            "batch"
        )

        self.flow.define_transition(
            "warped_images",
            self._warp_images,
            "images_1",
            "images_2",
            "intrinsics",
            "pose_gt_forward",
            "pose_gt_backward",
            "depth_map_1",
            "depth_map_2",
        )
        self._init_loss_flow(
            "warped_images",
            self._get_corr,
            "depth_map_1",
            "depth_map_2",
            "pose_gt_forward",
            "pose_gt_backward",
            "intrinsics",
            "descriptors",
            "image_shape",
        )

    def _check_batch(self, batch):
        ensure_is_instance(batch, NamedContext)
        
        batch.ensure_exists("image_1")
        batch.ensure_exists("image_2")
        batch.ensure_exists("rel_pose")
        batch.ensure_exists("intrinsics")
        batch.ensure_exists("depth_map_1")
        batch.ensure_exists("depth_map_2")
        
        # check data shape
        assert len(batch["image_1"].shape) == 4
        assert len(batch["image_2"].shape) == 4

        image_1 = batch["image_1"].to(self.device)
        image_2 = batch["image_2"].to(self.device)
        shape = image_1.shape
        pose_forward = torch.from_numpy(batch["rel_pose"][0][0][0]).to(self.device)
        pose_backward = torch.from_numpy(batch["rel_pose"][0][0][1]).to(self.device)

        intrinsics = torch.from_numpy(batch["intrinsics"][0][0]).to(self.device)
        depth_map_1 = torch.from_numpy(batch["depth_map_1"][0]).to(self.device)
        depth_map_2 = torch.from_numpy(batch["depth_map_2"][0]).to(self.device)
        
        return image_1, image_2, shape, pose_forward, pose_backward, intrinsics, depth_map_1, depth_map_2

    def _warp_images(self, images_1, images_2 , intrinsics, pose_forward, pose_backward, depth_map_1, depth_map_2):
        shape = images_1.shape
        images_1 = images_1.to(torch.float32)
        images_2 = images_2.to(torch.float32)

        # apply two homographic transforms to each input images
        sampler = HomographicSampler(
            shape[0],     # batch_size 1 
            # shape[-2:],   # 3, 164, 164 ??
            device=images_1.device,
        )
        # print("in_warp_images",intrinsics.shape)

        # print("warp depth map shape", depth_map.shape)
        # warp depth map shape torch.Size([128, 416])
        # print("================")
        # print(pose_backward.shape)
        # print(pose_forward.shape)
        # print(pose_backward)
        # print(pose_forward)
        # print("================")
        
        warped_images_1 = sampler.extract_crop(depth_map_2, intrinsics, pose_backward, images_2, shape[-2:])
        warped_images_2 = sampler.extract_crop(depth_map_1, intrinsics, pose_forward, images_1, shape[-2:])
      

        io.imsave("/root/silk/folder_for_viz/images_1.png", images_1[0][0].squeeze().cpu())
        io.imsave("/root/silk/folder_for_viz/images_2.png", images_2[0][0].squeeze().cpu())

        io.imsave("/root/silk/folder_for_viz/forward_warped_1.png", warped_images_1[0][0].cpu())
        io.imsave("/root/silk/folder_for_viz/forward_warped_2.png", warped_images_2[0][0].cpu())
        
        
        # in _extract_crop  tensor([[[ 7.2797e-01,  4.0495e-02,  0.0000e+00],
        # [-2.6342e-02,  5.7839e-01,  4.3608e-18],
        # [ 3.2213e-02, -4.4215e-01,  1.0000e+00]]], device='cuda:1',
        # dtype=torch.float64)
        # in _transform_points  torch.Size([1, 1, 1, 3, 3])
        # in _transform_points  tensor([[[[[ 7.2797e-01,  4.0495e-02,  0.0000e+00],
        # [-2.6342e-02,  5.7839e-01,  4.3608e-18],
        # [ 3.2213e-02, -4.4215e-01,  1.0000e+00]]]]], device='cuda:1',
        # dtype=torch.float64)
        # in _transform_points  tensor([[[ 7.2797e-01,  4.0495e-02,  0.0000e+00],
        # [-2.6342e-02,  5.7839e-01,  4.3608e-18],
        # [ 3.2213e-02, -4.4215e-01,  1.0000e+00]]], device='cuda:1',
        # dtype=torch.float64)
        # in _transform_points  164
        # in _transform_points  torch.Size([164, 3, 3])
        #after extract crop returns sampled pixels? oo
        # print("in _warp_images ", images.shape, warped_images.shape) 
        # in _warp_images  torch.Size([1, 3, 164, 164]) torch.Size([1, 3, 164, 164])

        
        images = torch.stack((images_1, images_2), dim=1) #original and warped
        # print("in _warp_images ", images.shape)
        # in _warp_images  torch.Size([1, 2, 3, 164, 164])

        images = images.view((-1,) + shape[1:])        
        # print("in _warp_images ", images.shape)
        # in _warp_images  torch.Size([2, 3, 164, 164])
    
        return images

    def get_kpt_position(self):
        linear_mapping = self._model.coordinate_mapping_composer.get("images", "logits")
        # print(linear_mapping)
        return linear_mapping 
    
        # self.flow.define_transition(
        #     ("corr_forward", "corr_backward"),
        #     corr_fn,
        #     "descriptors",
        #     "image_shape",
        #     pose_gt_forward,
        #     pose_gt_backward,
        #     intrinsics,
        #     crop_loc,
        # )

    def _get_corr(self, descriptors, image_shape, pose_gt_forward, pose_gt_backward, intrinsics, depth_map_1, depth_map_2):
        sampler = HomographicSampler(
            image_shape[0],     # batch_size 1 
            # shape[-2:],   # 3, 164, 164 ??
            device=descriptors.device,
            # **self._training_random_homography_kwargs,
        )

        # print("in _get_corr ", type(sampler))
        #in _get_corr  <class 'silk.transforms.cv.homography.RandomHomographicSampler'>

        # print("in _get_corr ", descriptors.shape, type(descriptors))
        #in _get_corr  torch.Size([2, 128, 146, 146]) <class 'torch.Tensor'>

        # print("in _get_corr ", descriptors.shape, type(descriptors))

        # print("in _get_corr ", image_shape)
        #in _get_corr  torch.Size([1, 3, 164, 164])

        batch_size = image_shape[0]
        descriptors_height = descriptors.shape[2]
        descriptors_width = descriptors.shape[3]
        cell_size = 1.0

        # remove confidence value
        positions = HomographicSampler._create_meshgrid(
            descriptors_height,
            descriptors_width,
            device=descriptors.device,
            normalized=False,
        )
        positions = positions.expand(batch_size, -1, -1, -1)  # add batch dim
        positions = positions.reshape(batch_size, -1, 2)

        
        coord_mapping = self._model.coordinate_mapping_composer.get(
            "images",
            "raw_descriptors",
        )

        # send to image coordinates
        positions = coord_mapping.reverse(positions)

        # transform label positions to transformed image space
        warped_positions_forward = sampler.transform_points(
            depth_map_2,
            pose_gt_backward, 
            intrinsics,
            positions,
            image_shape=image_shape[-2:],
            direction="forward",
            ordering="xy",
        )
        # print("warped points forward ", type(warped_positions_forward), warped_positions_forward.shape)
        # print("warped points forward ", warped_positions_forward)

        # warped points forward  <class 'torch.Tensor'> torch.Size([1, 21316, 2])
        # warped points forward  tensor([[[  10.7225,   28.2833],
        #          [  11.2574,   28.0490],
        #          [  11.7974,   27.8124],
        #          ...,
        #          [2044.7441, 1595.0794],
        #          [2378.3369, 1831.1356],
        #          [2834.7800, 2154.1229]]], device='cuda:1', dtype=torch.float64)


        warped_positions_backward = sampler.transform_points(
            depth_map_1,
            pose_gt_forward, 
            intrinsics,
            positions,
            image_shape=image_shape[-2:],
            direction="backward",
            ordering="xy",
        )
        # print("warped points backward ", type(warped_positions_backward), warped_positions_backward.shape)
        # print("warped points backward ", warped_positions_backward)
        # warped points backward  <class 'torch.Tensor'> torch.Size([1, 21316, 2])
        # warped points backward  tensor([[[  5.2779, -25.3738],
        #          [  7.3847, -24.1958],
        #          [  9.4441, -23.0443],
        #          ...,
        #          [104.9484, 115.3639],
        #          [105.2543, 115.2318],
        #          [105.5579, 115.1007]]], device='cuda:1', dtype=torch.float64)


        # send back to descriptor coordinates
        warped_positions_forward = coord_mapping.apply(warped_positions_forward)
        warped_positions_backward = coord_mapping.apply(warped_positions_backward)
        # print("in _get_corr ", type(warped_positions_forward), warped_positions_forward.shape)
        # in _get_corr  <class 'torch.Tensor'> torch.Size([1, 21316, 2])

        # print("in _get_corr ", type(warped_positions_backward), warped_positions_backward.shape)
        # in _get_corr  <class 'torch.Tensor'> torch.Size([1, 21316, 2])
        
        corr_forward = positions_to_unidirectional_correspondence(
            warped_positions_forward,
            descriptors_width,
            descriptors_height,
            cell_size,
            ordering="xy",
        )

        corr_backward = positions_to_unidirectional_correspondence(
            warped_positions_backward,
            descriptors_width,
            descriptors_height,
            cell_size,
            ordering="xy",
        )

        # print("in _get_corr ", type(corr_forward), corr_forward.shape)
        # in _get_corr  <class 'torch.Tensor'> torch.Size([1, 21316])

        # print("in _get_corr ", type(corr_backward), corr_backward.shape)
        # in _get_corr  <class 'torch.Tensor'> torch.Size([1, 21316])


        corr_forward, corr_backward = keep_mutual_correspondences_only(
            corr_forward, corr_backward
        )

        # print("in _get_corr ", type(corr_forward), corr_forward.shape)
        # in _get_corr  <class 'torch.Tensor'> torch.Size([1, 21316])

        # print("in _get_corr ", type(corr_backward), corr_backward.shape)
        # in _get_corr  <class 'torch.Tensor'> torch.Size([1, 21316])

        return corr_forward, corr_backward

   