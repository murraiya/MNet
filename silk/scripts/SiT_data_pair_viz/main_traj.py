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
from tqdm import tqdm
from SiT_dataset import test_framework_SiT
from silk.cli.image_pair_visualization import save_image
from vo import PinholeCamera, VisualOdometry

# DATASET_PATH = "/data/SiT/Outdoor_Alley_3-003"
DATASET_PATH = "/data/SiT/Corridor_1-002"
# DATASET_PATH = "/data/SiT/Three_way_Intersection_4-005"


def main():
    traj = np.zeros((1500,1500,3), dtype=np.uint8)

    framework = test_framework_SiT(DATASET_PATH)
    print('{} files to test'.format(len(framework))) # 1591 files to test
    

    cam = PinholeCamera(1014.8045, 1119.96277, 973.26314, 592.73372)
    vo = VisualOdometry(cam)

    for sample in tqdm(framework):
        img = sample['img']
        gt = sample['relative_Trs']
        print(gt)
        # print(type(img), len(img), img[0].shape) <class 'torch.Tensor'> 1 torch.Size([1, 1200, 1920])
        vo.update(img, gt[0])

        curr_t = vo.curr_t
        x, y, z = curr_t[0], curr_t[1], curr_t[2]
        # print(x, y, z)
        draw_x, draw_y = int(10*x)+290, int(10*z)+900
        true_x, true_y = int(10*vo.gt[0][3])+290, int(10*vo.gt[2][3])+900

        cv2.circle(traj, (draw_x,draw_y), 1, (255, 255, 0), 2)
        cv2.circle(traj, (true_x,true_y), 1, (0, 0, 255), 2) # GT in red
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        cv2.imwrite('map.png', traj)
    print("done")


if __name__ == "__main__":
    main()
