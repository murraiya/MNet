# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append(r'/root/silk')

import os
import numpy as np
import cv2
from tqdm import tqdm
from SiT_dataset import test_framework_SiT
from util import convert_points_from_homogeneous, get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image

# DATASET_PATH = "/data/SiT/Outdoor_Alley_3-003"
# DATASET_PATH = "/data/SiT/Corridor_1-002"
DATASET_PATH = "/data/SiT/Three_way_Intersection_4-005"


def main():
    framework = test_framework_SiT(DATASET_PATH)
    last_img, last_positions, last_desc = None, None, None
    curr_img, curr_positions, curr_desc = None, None, None
    model = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))

    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    out = cv2.VideoWriter('/data/script_pair_viz.avi', fourcc, 25.0, (3840, 1200))

    for sample in tqdm(framework):
        curr_img = sample["img"]
        curr_path = sample["path"]
        print(type(curr_img), curr_img.shape) # torch.Size([1, 1, 1200, 1920])
        curr_positions , curr_desc = model(curr_img)
        print(len(curr_desc)) # 1
        print(type(curr_desc)) # Tuple
        print(type(curr_desc[0])) #torch.Tensor
        print(curr_desc[0].shape) #torch.Size([21, 128])

        # logits? print(curr_positions, curr_positions[0].shape) 10001, 3
        curr_positions = from_feature_coords_to_image_coords(model, curr_positions)
        # get pixel positions from logits?


        if last_img != None:
            matches = SILK_MATCHER(last_desc[0], curr_desc[0])
            image_pair = create_img_pair_visual(last_path, curr_path, None, None,
                            last_positions[0][matches[:,0]].detach().cpu().numpy(),
                            curr_positions[0][matches[:,1]].detach().cpu().numpy())
            out.write(image_pair)
            
            
        last_img, last_path, last_positions, last_desc = curr_img, curr_path, curr_positions, curr_desc

    out.release()
    print("done")


if __name__ == "__main__":
    main()
