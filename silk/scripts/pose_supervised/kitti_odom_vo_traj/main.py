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
from pose_formatted_kitti_odom_dataset import test_framework_KITTI_fomatted_odom
# from sfm_kitti_odom_dataset import test_framework_sfm_KITTI
from silk.cli.image_pair_visualization import save_image
from pinhole_camera import PinholeCamera
from visual_odometry import VisualOdometry
# from vo_pose import VisualOdometry as VOpose
# from vo_silk_opencv import VisualOdometry as VOcv

formatted_odom_DATASET_PATH = "/data/formatted_kitti_odom"
# DATASET_PATH = "/data/kitti_odom_color_2012_dataset"
# sfm_DATASET_PATH = "/data/sfm_formatted_kitti_odom"

OUTPUT_IMAGE_PATH = "./img.png"


# silk.silk.models.silk line 39 uncomment plz
def main():
    traj = np.zeros((1500,1500,3), dtype=np.uint8)

    # framework = test_framework_sfm_KITTI(sfm_DATASET_PATH)
    framework = test_framework_KITTI_fomatted_odom(formatted_odom_DATASET_PATH)
    # framework = test_framework_KITTI(DATASET_PATH, SEQUENCE_NUM, 128, 416)
    print('{} files to test'.format(len(framework))) # 1591 files to test
    

    cam = PinholeCamera(715.7424, 718.5575, 609.2514, 186.07974)
    # vopose = VOpose(cam)
    # vocv = VOcv(cam)
    vo = VisualOdometry(cam)

    for sample in tqdm(framework): #2790 files to test
        # intrinsic = sample['cam']
        image_1 = sample['images_1'].to("cuda:1")
        image_2 = sample['images_2'].to("cuda:1")
        intrinsic = torch.from_numpy(sample['intrinsic'])
        path = sample['path']
        rel_gt = sample['rel_pose'][0] # np.ndarray 4x4
        abs_gt = sample['abs_pose'] # [0], [1] np.ndarray 4x4 transformation mat
        # print(path[0]) # confirmed that it is sequential
        # print(path[1])
        vo.update(image_1, image_2, abs_gt, rel_gt, intrinsic)

        abs_gt_x, abs_gt_z = int(vo.abs_gt[0][0][3])+290, int(vo.abs_gt[0][2][3])+900 # red
        rel_gt_x, rel_gt_z = int(vo.rel_gt[0][3])+290, int(vo.rel_gt[2][3])+900 # green
        silk_vo_x, silk_vo_z = int(vo.silk_t[0])+290, int(vo.silk_t[2])+900 # blue
        pred_x, pred_z = int(1000*vo.pred[0][3])+290, int(1000*vo.pred[2][3])+900 # mint
        print(pred_x, pred_z)
        # print("if silk_t 3-vec ", vo.silk_t.shape == (3,1))

        cv2.circle(traj, (abs_gt_x, abs_gt_z), 1, (0, 0, 255), 4) # red BGR order
        cv2.circle(traj, (rel_gt_x, rel_gt_z), 1, (0, 255, 0), 3) # green
        cv2.circle(traj, (silk_vo_x, silk_vo_z), 1, (255, 0, 0), 2) # blue
        cv2.circle(traj, (pred_x, pred_z), 1, (255, 255, 0), 4) # mint

        cv2.imwrite('map.png', traj)

    print("avg matches len of silk+opencv vo: ", vo.silk_macthes_len/vo.frames_num)
    # print("rel gt & rel predicted l2 error of 4x4 matrix: ", vopose.error/vopose.frames_num)
    
    print("done")


if __name__ == "__main__":
    main()
