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
from kitti_odom_dataset import test_framework_KITTI
from formatted_kitti_odom_dataset import test_framework_KITTI_fomatted_odom
from util import convert_points_from_homogeneous, get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image


DATASET_PATH = "/data/kitti_odom_color_2012_dataset"
formatted_odom_DATASET_PATH = "/data/formatted_kitti_odom"
sfm_DATASET_PATH = "/data/sfm_formatted_kitti_odom"

SEQUENCE_NUM = "09" # 09 or 10 .. sfm did this
OUTPUT_IMAGE_PATH = "./img.png"


def main():
    # framework = test_framework_sfm_KITTI(sfm_DATASET_PATH)
    # framework = test_framework_KITTI(DATASET_PATH, SEQUENCE_NUM, 128, 416)
    framework = test_framework_KITTI_fomatted_odom(formatted_odom_DATASET_PATH)
    # print('{} files to test'.format(len(framework))) # 1591 files to test
    last_img, last_positions, last_desc = None, None, None
    curr_img, curr_positions, curr_desc = None, None, None
    model = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))

    # fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter('/data/script_pair_viz.avi', fourcc, 25.0, (2482, 376))

    for sample in tqdm(framework):
        curr_img = sample["img"]
        curr_path = sample["path"]
        print(type(curr_img), curr_img.shape) # torch.Size([1, 1, 376, 1241])
        # io.imsave("./read_img.png", curr_img[0].cpu())
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
    print(f"result saved in {OUTPUT_IMAGE_PATH}")
    print("done")


if __name__ == "__main__":
    main()
