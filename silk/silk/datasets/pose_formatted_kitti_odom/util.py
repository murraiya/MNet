# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import sys
sys.path.append(r'/root/silk')

import os
from copy import deepcopy

import numpy as np
import skimage.io as io
import torch
import cv2
from skimage.transform import resize

import torchvision
from silk.backbones.silk.silk import SiLKVGG as SiLK
from silk.backbones.superpoint.vgg import ParametricVGG

from silk.config.model import load_model_from_checkpoint
from silk.models.silk import matcher



    
# CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "../../assets/models/silk/analysis/alpha/pvgg-4.ckpt")
CHECKPOINT_PATH = os.path.join(
    os.path.dirname(__file__), "../../lightning_logs/sfm_kitti_raw_deafault_setting/checkpoints/epoch=85-step=85999.ckpt"
)
DEVICE = "cuda:0"

SILK_NMS = 0  # NMS radius, 0 = disabled
SILK_BORDER = 0  # remove detection on border, 0 = disabled
SILK_THRESHOLD = 1.0  # keypoint score thresholding, if # of keypoints is less than provided top-k, then will add keypoints to reach top-k value, 1.0 = disabled
SILK_TOP_K = 10000  # minimum number of best keypoints to output, could be higher if threshold specified above has low value
SILK_DEFAULT_OUTPUT = (  # outputs required when running the model
    "dense_positions",
    "normalized_descriptors",
    "probability",
)
SILK_SCALE_FACTOR = 1.41  # scaling of descriptor output, do not change
SILK_BACKBONE = ParametricVGG(
    use_max_pooling=False,
    padding=0,
    normalization_fn=[torch.nn.BatchNorm2d(i) for i in (64, 64, 128, 128)],
)
SILK_MATCHER = matcher(postprocessing="ratio-test", threshold=0.6)
# SILK_MATCHER = matcher(postprocessing="double-softmax", threshold=0.6, temperature=0.1)
# SILK_MATCHER = matcher(postprocessing="none")


def convert_points_from_homogeneous(
        points: torch.Tensor, eps: float = 1e-8
    ) -> torch.Tensor:
        # we check for points at max_val
        z_vec: torch.Tensor = points[..., -1:]

        # set the results of division by zeror/near-zero to 1.0
        # follow the convention of opencv:
        # https://github.com/opencv/opencv/pull/14411/files
        mask: torch.Tensor = torch.abs(z_vec) > eps
        scale = torch.where(mask, 1.0 / (z_vec + eps), torch.ones_like(z_vec))

        return scale * points[..., :-1]


def load_images(*paths, as_gray=True):
    images = np.stack([io.imread(path, as_gray=as_gray) for path in paths])
    images = torch.tensor(images, device=DEVICE, dtype=torch.float32)
    if not as_gray:
        images = images.permute(0, 3, 1, 2)
        images = images / 255.0
    else:
        images = images.unsqueeze(1)  # add channel dimension
    return images

def load_image(*paths, img_height = 128, img_width = 416, as_gray=True):
    def load_im(path, height, width, as_gray):
        tmp = io.imread(path, as_gray=as_gray)
        tmp = resize(tmp, (height, width)) 
        # tmp = tmp.contiguous()
        return tmp
         
    image = np.stack([load_im(path, img_height, img_width, as_gray=as_gray) for path in paths])
    # image = torch.tensor(image, device=DEVICE, dtype=torch.float32)
    image = torch.tensor(image, dtype=torch.float32)
    
    for i in range(image.shape[0]):
        image[i] = image[i].contiguous()
    
    if not as_gray:
        image = image.permute(0, 3, 1, 2)
        image = image / 255.0
    else:
        image = image.unsqueeze(1)  # add channel dimension
        image = image / 255.0

    # cv2.imwrite(image[0], "image.jpg")
    # print(image[0].shape, type(image[0]))
    # cv2.imwrite('img.jpg', image[0])


    # print(image.shape, image[0].is_contiguous())
    return image


def get_model(
    checkpoint=CHECKPOINT_PATH,
    nms=SILK_NMS,
    device=DEVICE,
    default_outputs=SILK_DEFAULT_OUTPUT,
):
    # load model
    model = SiLK(
        in_channels=1,
        backbone=deepcopy(SILK_BACKBONE),
        detection_threshold=SILK_THRESHOLD,
        detection_top_k=SILK_TOP_K,
        nms_dist=nms,
        border_dist=SILK_BORDER,
        default_outputs=default_outputs,
        descriptor_scale_factor=SILK_SCALE_FACTOR,
        padding=0,
    )
    model = load_model_from_checkpoint(
        model,
        checkpoint_path=checkpoint,
        state_dict_fn=lambda x: {k[len("_mods.model.") :]: v for k, v in x.items()},
        device=device,
        freeze=True,
        eval=True,
    )
    return model
