# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append(r'/root/silk')

import os
import numpy as np
import cv2
# from path import Path
from tqdm import tqdm
from pose_formatted_kitti_odom_dataset import test_framework_KITTI_fomatted_odom
from util import convert_points_from_homogeneous, get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords_1, from_feature_coords_to_image_coords_2
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image


# DATASET_PATH = "/data/kitti_odom_color_2012_dataset"
formatted_odom_DATASET_PATH = "/data/formatted_kitti_odom"
# sfm_DATASET_PATH = "/data/sfm_formatted_kitti_odom"

def main():
    # framework = test_framework_sfm_KITTI(sfm_DATASET_PATH)
    # framework = test_framework_KITTI(DATASET_PATH, SEQUENCE_NUM, 128, 416)
    framework = test_framework_KITTI_fomatted_odom(formatted_odom_DATASET_PATH)
    # print('{} files to test'.format(len(framework))) # 1591 files to test
    last_img, last_positions, last_desc = None, None, None
    curr_img, curr_positions, curr_desc = None, None, None
    model = get_model(default_outputs=("sparse_positions_1", "sparse_descriptors_1", "sparse_positions_2", "sparse_descriptors_2"))

    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    out = cv2.VideoWriter('/data/script_pair_viz.avi', fourcc, 10.0, (2482, 376))

    for sample in tqdm(framework):
        # last_img = sample["img_1"]
        # curr_img = sample["img_2"]
        path = sample["path"]
        # print(type(curr_img), curr_img.shape) # torch.Size([1, 1, 376, 1241])
        # io.imsave("./read_img.png", curr_img[0].cpu())
        last_positions , last_desc, curr_positions, curr_desc = model(sample["images_1"].to("cuda:1"), sample["images_2"].to("cuda:1"))
        
        print(len(curr_desc)) # 1
        print(type(curr_desc)) # Tuple
        print(type(curr_desc[0])) #torch.Tensor
        print(curr_desc[0].shape) #torch.Size([21, 128])

        # logits? print(curr_positions, curr_positions[0].shape) 10001, 3
        last_positions = from_feature_coords_to_image_coords_1(model, last_positions)
        curr_positions = from_feature_coords_to_image_coords_2(model, curr_positions)
        # get pixel positions from logits?

        matches = SILK_MATCHER(last_desc[0], curr_desc[0])
        image_pair = create_img_pair_visual(path[0], path[1], None, None,
                        last_positions[0][matches[:,0]].detach().cpu().numpy(),
                        curr_positions[0][matches[:,1]].detach().cpu().numpy())
        out.write(image_pair)
            
    out.release()
    print("done")


if __name__ == "__main__":
    main()
