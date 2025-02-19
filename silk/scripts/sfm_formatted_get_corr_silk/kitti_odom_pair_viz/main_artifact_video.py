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
from formatted_kitti_odom_dataset_artifact import test_framework_KITTI_artifact_formatted_odom
from util import convert_points_from_homogeneous, get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
from silk.cli.image_pair_visualization import img_pair_visual, save_image


DATASET_PATH = "/data/kitti_odom_color_2012_dataset"
formatted_odom_DATASET_PATH = "/data/formatted_kitti_odom"
sfm_DATASET_PATH = "/data/sfm_formatted_kitti_odom"

SEQUENCE_NUM = "09" # 09 or 10 .. sfm did this
OUTPUT_IMAGE_PATH = "./img.png"

def count_bads(box, positions):
    cnt = 0
    for pos in positions:
        if pos[1]>=box[0] and pos[1]<=box[1] and pos[0]>=box[2] and pos[0]<=box[3]:
            cnt+=1
    return cnt 

def main():
    framework = test_framework_KITTI_artifact_formatted_odom(formatted_odom_DATASET_PATH)
    last_cv, last_img, last_positions, last_desc = None, None, None, None
    curr_cv, curr_img, curr_positions, curr_desc = None, None, None, None
    model = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))

    fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
    out = cv2.VideoWriter('/data/script_artifact_pair_viz.avi', fourcc, 25.0, (2482, 376))

    for sample in tqdm(framework):
        curr_img = sample["img"]
        curr_cv = sample["cv_img"]
        box = sample["box"]
        # print(curr_img.shape) # torch.Size([1, 1, 376, 1241])
        # print(box)
        curr_positions , curr_desc = model(curr_img)
        # logits? print(curr_positions, curr_positions[0].shape) 10001, 3
        curr_positions = from_feature_coords_to_image_coords(model, curr_positions)
        cnt = count_bads(box[0], curr_positions[0])
        if cnt>0:
            print("detect artifact point: ", cnt)
        # get pixel positions from logits?
        if last_img != None:
            matches = SILK_MATCHER(last_desc[0], curr_desc[0])
            cnt_match = count_bads(box[0], curr_positions[0][matches[:,1]])
            
            image_pair = img_pair_visual(last_cv[0], curr_cv[0],
                            last_positions[0][matches[:,0]].detach().cpu().numpy(),
                            curr_positions[0][matches[:,1]].detach().cpu().numpy())
            out.write(image_pair)
            if cnt_match > 0:
                print("matched artifact point: ", cnt)
                cv2.imsave('/data/script_artifact_pair_viz.png', image_pair)
            
        last_img, last_cv, last_positions, last_desc = curr_img, curr_cv, curr_positions, curr_desc

    out.release()
    print("done")


if __name__ == "__main__":
    main()
