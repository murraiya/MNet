# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
from typing import Any, Dict, Optional, Union
from PIL import Image

import math
import cv2
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np
import skimage.io as io
from silk.backbones.loftr.positional_encoding import PositionEncodingSine
from silk.backbones.silk.silk import SiLKBase as BackboneBase
from silk.backbones.silk.silk import SiLKVGG
# from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.config.core import ensure_is_instance
from silk.config.optimizer import Spec
from silk.cv.homography import HomographicSampler
from silk.flow import AutoForward, Flow
from silk.losses.info_nce.loss import (
    keep_mutual_correspondences_only,
    positions_to_unidirectional_correspondence,
    sparse_positions_to_corr,
)
from silk.losses.sfmlearner.sfm_loss import epiploar_loss, photometric_reconstruction_loss
from silk.matching.mnn import (
    compute_dist,
    double_softmax_distance,
    match_descriptors,
    mutual_nearest_neighbor,
)
from silk.models.abstract import OptimizersHandler, StateDictRedirect
from silk.transforms.abstract import MixedModuleDict, NamedContext, Transform
from silk.transforms.tensor import NormalizeRange
from torchvision.transforms import Grayscale
from silk.models.sift import SIFT


_DEBUG_MODE_ENABLED = True

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
        self.predicted_pose = None
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

        self._model = model 
        self._sift = SIFT("cuda:1")
        self._loss = loss
        #silk.backbones.silk.silk.SiLKVGG (backbone = silk.backbones.superpoint.vgg.ParametricVGG)
        self._contextualizer = contextualizer
        if contextualizer:
            self._pe = PositionEncodingSine(256, max_shape=(512, 512))
        self._image_aug_transform = image_aug_transform

    # def processFrame(self, img, rel_gt, abs_gt):
    #     positions_sift, descriptors_sift = self.sift(img)
    #     print("sift number", len(positions_sift), len(positions_sift[0]))
    #     matches_sift, dist_sift = SILK_MATCHER(descriptors_sift[0], descriptors_sift[1])

    #     E, mask = cv2.findEssentialMat(
	# 		positions_sift[1][matches_sift[:, 1]].detach().cpu().numpy()[:, [1,0]],
	# 		positions_sift[0][matches_sift[:, 0]].detach().cpu().numpy()[:, [1,0]],
    #         focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    #     _, R_sift, t_sift, mask = cv2.recoverPose(E, 
   	# 		positions_sift[1][matches_sift[:, 1]].detach().cpu().numpy()[:, [1,0]],
	# 		positions_sift[0][matches_sift[:, 0]].detach().cpu().numpy()[:, [1,0]],
	# 		focal=self.focal, pp = self.pp)

	# 	absolute_scale = self.getAbsoluteScale(abs_gt)
		

	# 	R_=R.copy()
	# 	t_=(absolute_scale*R@t).copy()

	# 	R_sift_ = R_sift.copy()
	# 	t_sift_ = absolute_scale*R_sift@t_sift
		

	# 	return R_, t_, R_sift_, t_sift_
    
    @property
    def coordinate_mapping_composer(self):
        return self._model.coordinate_mapping_composer

    def from_feature_coords_to_image_coords(self, desc_positions):
        if isinstance(desc_positions, tuple):
            return tuple(
                self.from_feature_coords_to_image_coords(
                    dp,
                )
                for dp in desc_positions
            )
        coord_mapping = self.coordinate_mapping_composer.get("images", "raw_descriptors")
        # print(coord_mapping, "from image to desc")
        desc_positions = torch.cat(
            [
                coord_mapping.reverse(desc_positions[..., :2]),
                desc_positions[..., 2:],
            ],
            dim=-1,
        )
        return desc_positions


    def from_disp_coords_to_image_coords(self, desc_positions):
        if isinstance(desc_positions, tuple):
            return tuple(
                self.from_feature_coords_to_image_coords(
                    dp,
                )
                for dp in desc_positions
            )
        coord_mapping = self.coordinate_mapping_composer.get("images", "disp_maps")
        # print(coord_mapping, "from image to disp")
        desc_positions = torch.cat(
            [
                coord_mapping.reverse(desc_positions[..., :2]),
                desc_positions[..., 2:],
            ],
            dim=-1,
        )
        return desc_positions


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
        image_shape,
    ):
        self.flow.define_transition(
            "augmented_images",
            self._aug_images,
            images_input_name,
        )
        self.flow.define_transition(
            "gray_images",
            self._grayify,
            "augmented_images",
        )
        # self.flow.define_transition(
        #     ("descriptors", "logits", "sparse_positions", "sparse_descriptors", "probability"),
        #     self._model.forward_flow,
        #     outputs=Flow.Constant(("normalized_descriptors", "logits", "sparse_positions", "sparse_descriptors", "probability")),
        #     images = "gray_images",
        # )
        self.flow.define_transition(
            ("descriptors", "sparse_positions", "sparse_descriptors", "probability"),
            self._model.forward_flow,
            outputs=Flow.Constant(("normalized_descriptors", "sparse_positions", "sparse_descriptors", "probability")),
            images = "gray_images",
        )

        self.flow.define_transition(
            "sparse_positions_",
            self.from_feature_coords_to_image_coords,
            "sparse_positions",
        )
        self.flow.define_transition(
            "sparse_positions_1",
            lambda x: x[0],
            "sparse_positions_",            
        )
        self.flow.define_transition(
            "sparse_positions_2",
            lambda x: x[1],
            "sparse_positions_",            
        )
        self.flow.define_transition(
            "descriptors_shape",
            lambda x: x.shape,
            "descriptors",
        )
        # self.flow.define_transition(
        #     ("logits_0", "logits_1"),
        #     self._split_logits,
        #     "logits",
        # )
        # self.flow.define_transition(
        #     ("descriptors_0", "descriptors_1"),
        #     self._split_descriptors,
        #     "descriptors",
        # )
        # self.flow.define_transition(
        #     ("pose_loss", "predicted_pose", "predicted_pose_inv", "matched_sparse_positions_1", "matched_sparse_positions_2", "matches"),
        #     epiploar_loss,
        #     intrinsics,
        #     "pose6d",
        #     "sparse_positions_1",
        #     "sparse_positions_2",
        #     "sparse_descriptors",
        # )
        # self.flow.define_transition(
        #     "pose_loss",
        #     epiploar_loss,
        #     intrinsics,
        #     pose_gt_forward,
        #     "sparse_positions_1",
        #     "sparse_positions_2",
        #     "sparse_descriptors",
        # )
        
        self.flow.define_transition(
            ("corr_forward", "corr_backward"),
            corr_fn,
            "descriptors",
            image_shape,
            "sparse_positions",
            pose_gt_forward,
            pose_gt_backward,
            intrinsics,
            depth_map_1,
            depth_map_2,
            images_input_name,
        )
        self.flow.define_transition(
            # ("acontextual_descriptor_loss", "keypoint_loss", "precision", "recall"),
            ("acontextual_descriptor_loss", "keypoint_loss"),
            self._loss,
            "sparse_positions", 
            "sparse_descriptors", 
            "corr_forward", 
            "corr_backward",
            Flow.Constant(self._ghost_sim),
        )
        self.flow.define_transition(
            "recon_loss",
            photometric_reconstruction_loss,
            "gray_images",
            depth_map_1,
            depth_map_2,
            intrinsics,
            pose_gt_forward, 
            pose_gt_backward,
            "probability", 
        )
        # self.flow.define_transition(
        #     ("contextual_descriptor_0", "contextual_descriptor_1"),
        #     self._contextualize,
        #     "descriptors_0",
        #     "descriptors_1",
        #     "descriptors_shape",
        # )
        # self.flow.define_transition(
        #     "contextual_descriptor_loss",
        #     self._contextual_loss,
        #     "contextual_descriptor_0",
        #     "contextual_descriptor_1",
        #     "corr_forward",
        #     "corr_backward",
        #     "logits_0",
        #     "logits_1",
        # )
        self._loss_fn = self.flow.with_outputs(
            (
                "recon_loss",
                "acontextual_descriptor_loss", 
                "keypoint_loss"               
                # "precision",
                # "recall",
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

    def _aug_images(self, images):
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
        # pose_loss, ctx_desc_loss, actx_desc_loss, keypt_loss, precision, recall = \
        # self._loss_fn(
        #     batch, use_image_aug
        # )
        recon_loss, actx_desc_loss, keypt_loss = \
        self._loss_fn(
            batch, use_image_aug
        )
        
        loss_for_log = recon_loss + actx_desc_loss
        
        self.log(f"{mode}.total.loss", loss_for_log)
        self.log(f"{mode}.acontextual.descriptors.loss", actx_desc_loss)
        # self.log(f"{mode}.keypoints.loss", keypt_loss)
        self.log(f"{mode}.recon.loss", recon_loss)
        # self.log(f"{mode}.precision", precision)
        # self.log(f"{mode}.recall", recall)
        if (self._ghost_sim is not None) and (mode == "train"):
            self.log("ghost.sim", self._ghost_sim)
        print(recon_loss, actx_desc_loss)


        # if math.isnan(actx_desc_loss):
        #     # print("actx_desc_loss is nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     loss = None
            
        # else:
        #     loss = actx_desc_loss

        # # if math.isnan(actx_desc_loss):
        # #     # print("actx_desc_loss is nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # #     loss = keypt_loss
            
        # #     if math.isnan(keypt_loss):
        # #         # print("keypt loss is nan!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # #         loss = None
        # # elif math.isnan(actx_desc_loss) == 0:
        # #     if math.isnan(keypt_loss):
        # #         loss = actx_desc_loss
        # #     else:
        # #         loss = actx_desc_loss + keypt_loss

        # if loss == None: loss = recon_loss
        # else: loss += recon_loss
        # self.log(f"{mode}.monitor.loss", loss)

        return loss_for_log

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
        )
        self._init_loss_flow(
            "warped_images",
            self._get_corr,
            "depth_map_1",
            "depth_map_2",
            "pose_gt_forward",
            "pose_gt_backward",
            "intrinsics",
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
        
        assert len(batch["image_1"].shape) == 4
        assert len(batch["image_2"].shape) == 4
        
        image_1 = batch["image_1"].to(self.device)
        image_2 = batch["image_2"].to(self.device)
        shape = image_1.shape
        pose_forward = torch.from_numpy(batch["rel_pose"][0][0][0]).to(self.device)
        pose_backward = torch.from_numpy(batch["rel_pose"][0][0][1]).to(self.device)
        # print(pose_forward, pose_backward)

        intrinsics = torch.from_numpy(batch["intrinsics"][0][0]).to(self.device)
        depth_map_1 = torch.from_numpy(batch["depth_map_1"][0]).to(self.device)
        depth_map_2 = torch.from_numpy(batch["depth_map_2"][0]).to(self.device)

        return image_1, image_2, shape, pose_forward, pose_backward, intrinsics, depth_map_1, depth_map_2

    def _warp_images(self, images_1, images_2):
        shape = images_1.shape
        images_1 = images_1.to(torch.float32)
        images_2 = images_2.to(torch.float32)
        
        images = torch.stack((images_1, images_2), dim=1) #original and warped
        images = images.view((-1,) + shape[1:])        
        
        return images

    def get_kpt_position(self):
        linear_mapping = self._model.coordinate_mapping_composer.get("images", "logits")
        # print(linear_mapping)
        return linear_mapping 


    # the 241015 version
    def _get_corr(self, descriptors, image_shape, sparse_positions, pose_gt_forward, pose_gt_backward, intrinsics, depth_map_1, depth_map_2, img):
        sampler = HomographicSampler(
            image_shape[0],     # batch_size 1 
            # shape[-2:],   # 3, 164, 164 ??
            device=descriptors.device,
            # **self._training_random_homography_kwargs,
        )

        batch_size = image_shape[0]
        descriptors_height = descriptors.shape[2]
        descriptors_width = descriptors.shape[3]
        cell_size = 1.0

        # remove confidence value
        # positions = HomographicSampler._create_meshgrid(
        #     descriptors_height,
        #     descriptors_width,
        #     device=descriptors.device,
        #     normalized=False,
        # )
        # # print(positions.shape)
        # # torch.Size([1, 352, 1208, 2])

        # positions = positions.expand(batch_size, -1, -1, -1)  # add batch dim
        # positions = positions.reshape(batch_size, -1, 2)
        # print(positions[0][:10]) 0.5, 0.5 ~
        
        coord_mapping = self._model.coordinate_mapping_composer.get(
            "images",
            "raw_descriptors",
        )
        # print(coord_mapping)
        # x <- tensor([1., 1.]) x + tensor([-9., -9.])
        
        
        # print(sparse_positions[0].shape, sparse_positions[0].dtype)
        # print(positions.shape, positions.dtype)
        # torch.Size([7001, 3]) torch.float32
        # torch.Size([1, 425216, 2]) torch.float32

        # send to image coordinates
        # positions = coord_mapping.reverse(positions)

        
        # make x,y order
        sparse_positions_0 = sparse_positions[0][ :, :2][:, [1,0]]
        sparse_positions_1 = sparse_positions[1][ :, :2][:, [1,0]]
        
        if sparse_positions_0.shape[0] != sparse_positions_1.shape[0]:
            print(sparse_positions_0.shape, sparse_positions_1.shape)
            # torch.Size([10001, 2]) torch.Size([10000, 2])
            # torch.Size([10000, 2]) torch.Size([10001, 2])
            exit(0)

        sparse_corr_0 = positions_to_unidirectional_correspondence(
            sparse_positions_0.clone().unsqueeze(0),
            descriptors_width,
            descriptors_height,
            cell_size,
            ordering="xy",
        )

        sparse_corr_1 = positions_to_unidirectional_correspondence(
            sparse_positions_1.clone().unsqueeze(0),
            descriptors_width,
            descriptors_height,
            cell_size,
            ordering="xy",
        )


        sparse_positions_0 = coord_mapping.reverse(sparse_positions_0).unsqueeze(0)
        sparse_positions_1 = coord_mapping.reverse(sparse_positions_1).unsqueeze(0)
        

        
        # print(sparse_positions_0.shape) #torch.Size([1, 7000, 2])
        # print(sparse_positions_0[0][:10])
        # torch.Size([1, 7001, 2])
        # tensor([[ 55.5000,   9.5000],
        #         [ 56.5000,   9.5000],
        #         [ 57.5000,   9.5000],
        #         [144.5000,   9.5000],
        #         [197.5000,   9.5000],
        #         [198.5000,   9.5000],
        #         [208.5000,   9.5000],
        #         [267.5000,   9.5000],
        #         [268.5000,   9.5000],
        #         [331.5000,   9.5000]], device='cuda:1')


        # this is right version of making corr 241113
        # transform label positions to transformed image space
        warped_positions_backward = sampler.transform_points(
            depth_map_2,
            pose_gt_forward, 
            intrinsics.clone(),
            sparse_positions_1.clone(),
            # image_shape=image_shape[-2:],
            ordering="xy",
            # shape=(descriptors_height,descriptors_width),    
            imshape = image_shape,
        )
        # warped mask is in image2 coordinate, value 0 where no point should be extracte
        # print(warped_positions_backward.shape) torch.Size([1, 425216, 2])
        # print(max(warped_positions_backward[0,:,0]), max(warped_positions_backward[0,:,1]))
        # tensor(601.2357, device='cuda:1', dtype=torch.float64) tensor(178.6385, device='cuda:1', dtype=torch.float64)

        
        warped_positions_forward = sampler.transform_points(
            depth_map_1,
            pose_gt_backward, 
            intrinsics,
            sparse_positions_0.clone(),
            # image_shape=image_shape[-2:],
            ordering="xy",
            # shape=(descriptors_height,descriptors_width),
            imshape = image_shape,
        )
        # img_pair_visual(
        #     image1=img[0],
        #     image2=img[1],
        #     matched_keypoints=positions,
        #     matched_warped_keypoints=warped_positions_forward,
        # )
        # img_pair_visual(
        #     image1=img[0],
        #     image2=img[1],
        #     matched_keypoints=warped_positions_backward,
        #     matched_warped_keypoints=positions,
        # )
        

        # send back to descriptor coordinates
        warped_positions_forward = coord_mapping.apply(warped_positions_forward)
        warped_positions_backward = coord_mapping.apply(warped_positions_backward)
        
        # print(warped_positions_backward.shape, warped_positions_forward.shape)
        # torch.Size([1, 7001, 2]) torch.Size([1, 7001, 2])


        # print("-----------------------")
        # print(min(warped_positions_backward[0,:,0]), max(warped_positions_backward[0,:,0]))
        # print(min(warped_positions_backward[0,:,1]), max(warped_positions_backward[0,:,1]))

        # print(min(warped_positions_forward[0,:,0]), max(warped_positions_forward[0,:,0]))
        # print(min(warped_positions_forward[0,:,1]), max(warped_positions_forward[0,:,1]))


        # tensor(10.5201, device='cuda:1', dtype=torch.float64) tensor(1198.5054, device='cuda:1', dtype=torch.float64)
        # tensor(0.9843, device='cuda:1', dtype=torch.float64) tensor(335.3899, device='cuda:1', dtype=torch.float64)
        # tensor(-10.4018, device='cuda:1', dtype=torch.float64) tensor(1212.9620, device='cuda:1', dtype=torch.float64)
        # tensor(-1.7463, device='cuda:1', dtype=torch.float64) tensor(359.1085, device='cuda:1', dtype=torch.float64)

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

        corr_backward = sparse_positions_to_corr(
            sparse_corr_0, corr_backward
        )
        corr_forward = sparse_positions_to_corr(
            sparse_corr_1, corr_forward
        )
        # print(torch.count_nonzero(corr_forward>0), torch.count_nonzero(corr_backward>0))
        # # tensor(6959, device='cuda:1') tensor(7001, device='cuda:1')
        # print(corr_forward.shape, corr_backward.shape)
        # torch.Size([1, 7001]) torch.Size([1, 7001])
        # print(floored_positions_0.shape, floored_positions_1.shape)
        # torch.Size([1, 7001, 2]) torch.Size([1, 7001, 2])


        corr_forward, corr_backward = keep_mutual_correspondences_only(
            corr_forward, corr_backward
        )

        # print(torch.count_nonzero(corr_forward>0), torch.count_nonzero(corr_backward>0))
        # # tensor(360487, device='cuda:1') tensor(360487, device='cuda:1')
        
        return corr_forward, corr_backward




def img_pair_visual( # need cv.imread, uint8 img
    image1,
    image2,
    matched_keypoints,
    matched_warped_keypoints,
):
    image1 = image1.permute(1,2,0).squeeze().cpu().numpy().astype(np.uint8).copy()
    image2 = image2.permute(1,2,0).squeeze().cpu().numpy().astype(np.uint8).copy()

    io.imsave("folder_for_viz/im1.png", image1)
    io.imsave("folder_for_viz/im2.png", image2)

    matched_keypoints = matched_keypoints.squeeze().cpu().numpy().astype(np.int32)
    matched_warped_keypoints = matched_warped_keypoints.squeeze().cpu().numpy().astype(np.int32)

    # draw matched keypoint points and lines associating matched keypoints (point correspondences)
    for i in range(len(matched_keypoints)):
        if i%5!=0: continue
        img1_coords = matched_keypoints[i]
        img2_coords = matched_warped_keypoints[i]
        # add the width so the coordinates show up correctly on the second image

        radius = 1
        thickness = 0
        color = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255),
            )
        # points will be red (BGR color)
        image1 = cv2.circle(image1, img1_coords, radius, color, thickness)
        image2 = cv2.circle(image2, img2_coords, radius, color, thickness)
        
    
    io.imsave("./folder_for_viz/im1_.png", image1)
    io.imsave("./folder_for_viz/im2_.png", image2)
   
    return 