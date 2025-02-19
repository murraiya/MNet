# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import sys
sys.path.append(r'/root/silk')

import os
import torch
import numpy as np
import cv2
# from path import Path
from tqdm import tqdm
from formatted_kitti_odom_dataset_artifact import test_framework_KITTI_artifact_formatted_odom
# from sfm_kitti_odom_dataset import test_framework_sfm_KITTI
from silk.cli.image_pair_visualization import save_image
# from vo_tracking import PinholeCamera, VisualOdometry
from vo import PinholeCamera, VisualOdometry

formatted_odom_DATASET_PATH = "/data/formatted_kitti_odom"
DATASET_PATH = "/data/kitti_odom_color_2012_dataset"
sfm_DATASET_PATH = "/data/sfm_formatted_kitti_odom"

SEQUENCE_NUM = "09" # 09 or 10 .. sfm did this
OUTPUT_IMAGE_PATH = "./img.png"


def main():
    traj = np.zeros((1500,1500,3), dtype=np.uint8)

    # framework = test_framework_sfm_KITTI(sfm_DATASET_PATH)
    framework = test_framework_KITTI_artifact_formatted_odom(formatted_odom_DATASET_PATH)
    # framework = test_framework_KITTI(DATASET_PATH, SEQUENCE_NUM, 128, 416)
    print('{} files to test'.format(len(framework))) # 1591 files to test
    

    cam = PinholeCamera(715.7424, 718.5575, 609.2514, 186.07974)
    vo = VisualOdometry(cam)

    for sample in tqdm(framework):
        # intrinsic = sample['cam']
        img = sample['img']
        gt = sample['pose']
        print(gt)
        # print("img loaded with size ", img.shape) # torch.Size([1, 1, 370, 1226])
        # print(type(gt[0]), gt[0])
        vo.update(img, gt[0])

        curr_t = vo.curr_t
        x, y, z = curr_t[0], curr_t[1], curr_t[2]
        # print(x, y, z)
        draw_x, draw_y = int(x)+290, int(z)+900
        true_x, true_y = int(vo.gt[0][3])+290, int(vo.gt[2][3])+900
        print(x, y, z)
        print(vo.gt[0][3], vo.gt[1][3], vo.gt[2][3])

        cv2.circle(traj, (draw_x,draw_y), 1, (255, 255, 0), 2)
        cv2.circle(traj, (true_x,true_y), 1, (0, 0, 255), 2) # GT in red
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        cv2.imwrite('map.png', traj)


    # save_image(
    #     image_pair,
    #     os.path.dirname(OUTPUT_IMAGE_PATH),
    #     os.path.basename(OUTPUT_IMAGE_PATH),
    # )

    print(f"result saved in {OUTPUT_IMAGE_PATH}")
    print("done")


if __name__ == "__main__":
    main()
