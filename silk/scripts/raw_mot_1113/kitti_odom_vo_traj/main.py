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
import imageio as io
# from path import Path
from tqdm import tqdm
# from sfm_kitti_odom_dataset import test_framework_sfm_KITTI
from sfm_kitti_mot_dataset import kittiMOTdataset
from formatted_kitti_odom_dataset import test_framework_KITTI_fomatted_odom
from silk.cli.image_pair_visualization import save_image
from visual_odometry import VisualOdometry

# formatted_odom_DATASET_PATH = "/data/plzdontremove/formatted_kitti_odom/"
# DATASET_PATH = "/data/kitti_odom_color_2012_dataset"
sfm_DATASET_PATH = "/data/MOTkitti/testing/"

OUTPUT_IMAGE_PATH = "./img.png"

# silk.silk.models.silk line 39 uncomment plz
def main(seq):
    # seq = 15
    
    traj = np.zeros((600,600,3), dtype=np.uint8)
    framework = kittiMOTdataset(sfm_DATASET_PATH, seq)
    # framework = test_framework_KITTI_fomatted_odom(formatted_odom_DATASET_PATH)
    print('{} files to test'.format(len(framework))) # 1591 files to test
    
    vo = VisualOdometry()

    pred_array = np.zeros((len(framework), 12))
    gt_array = np.zeros((len(framework), 12))

    gt_last_t = np.array([0,0,0])
    gt_last_R = np.array([[1,0,0],[0,1,0],[0,0,1]])
    
    pred_last_t = np.array([0,0,0])
    pred_last_R = np.array([[1,0,0],[0,1,0],[0,0,1]])    

    sift_last_t = np.array([0,0,0])
    sift_last_R = np.array([[1,0,0],[0,1,0],[0,0,1]])

    j=-1

    for sample in tqdm(framework): #2790 files to test
        j+=1
        
        gt_pose = sample["rel_pose"][0]
        image = sample['images'].to("cuda:1")
        # image_2 = sample['images_2'].to("cuda:1")
        intrinsic = torch.from_numpy(sample['intrinsic'])
        # print(Rt_cam2_gt) #4,4
        # gt_pose = sample['rel_pose'][1] # np.ndarray 4x4
        abs_gt = sample['abs_pose'] # [0], [1] np.ndarray 4x4 transformation mat
        R_, t_, R_sift, t_sift = vo.update(image, abs_gt, gt_pose, intrinsic)
        if j==0:
            silk_sjsj = np.concatenate([R_, t_], axis=1)
            sift_T = np.concatenate([R_sift, t_sift], axis=1)
        else:
            silk_sjsj = np.concatenate([R_, t_], axis=1)
            sift_T = np.concatenate([R_sift, t_sift], axis=1)
            
        
        # pred_last_t = pred_last_t + pred_last_R @ silk_sjsj[:3,3]
        # pred_last_R = pred_last_R @ (silk_sjsj[:3,:3])
        # # print(pred_last_t)
        # pred_x = pred_last_t[0]
        # pred_y = pred_last_t[1]
        # pred_z = pred_last_t[2]
        # pred_mat = np.concatenate([pred_last_R, pred_last_t.reshape(3,1)], axis=1)
        # # print(pred_mat.shape)
        # # pred_mat = np.vstack([pred_mat, np.array([[0,0,0,1]])]) # make it 4x4
        # pred_array[j] = pred_mat.reshape(-1) 
        
        # gt_last_t = gt_last_t + gt_last_R @ gt_pose[:3,3]
        # gt_last_R = gt_last_R @ (gt_pose[:3,:3])
        # gt_x = gt_last_t[0]
        # gt_y = gt_last_t[1]
        # gt_z = gt_last_t[2]
        # gt_mat = np.concatenate([gt_last_R, gt_last_t.reshape(3,1)], axis=1)
        # # print(gt_mat.shape)
        # gt_array[j] = gt_mat.reshape(-1)

        
        
        # # cv2.circle(traj, (int(posehead_y)+290, int(posehead_z)+200), 1, (0, 255, 0), 1) # green
        # cv2.circle(traj, (int(pred_x)+290, int(pred_z)+200), 1, (255, 0, 0), 1) # blue
        # cv2.circle(traj, (int(gt_x)+290, int(gt_z)+200), 1, (0, 0, 255), 1) # red BGR order
        # cv2.imwrite('map.png'.format(seq), traj)

        pred_last_t = pred_last_t + pred_last_R @ silk_sjsj[:3,3]
        pred_last_R = pred_last_R @ (silk_sjsj[:3,:3])
        pred_x = pred_last_t[0]
        pred_y = pred_last_t[1]
        pred_z = pred_last_t[2]
        pred_mat = np.concatenate([pred_last_R, pred_last_t.reshape(3,1)], axis=1)
        pred_array[j] = pred_mat.reshape(-1) 
        
        sift_last_t = sift_last_t + sift_last_R @ sift_T[:3,3]
        sift_last_R = sift_last_R @ (sift_T[:3,:3])
        sift_x = sift_last_t[0]
        sift_y = sift_last_t[1]
        sift_z = sift_last_t[2]
        sift_mat = np.concatenate([sift_last_R, sift_last_t.reshape(3,1)], axis=1)
        # sift_array[j] = pred_mat.reshape(-1) 
        
        
        
        gt_last_t = gt_last_t + gt_last_R @ gt_pose[:3,3]
        gt_last_R = gt_last_R @ (gt_pose[:3,:3])
        gt_x = gt_last_t[0]
        gt_y = gt_last_t[1]
        gt_z = gt_last_t[2]
        gt_mat = np.concatenate([gt_last_R, gt_last_t.reshape(3,1)], axis=1)
        gt_array[j] = gt_mat.reshape(-1)

        cv2.circle(traj, (int(sift_x)+290, int(sift_z)+200), 1, (0, 255, 0), 1) # green
        cv2.circle(traj, (int(pred_x)+290, int(pred_z)+200), 1, (255, 0, 0), 1) # blue
        cv2.circle(traj, (int(gt_x)+290, int(gt_z)+200), 1, (0, 0, 255), 1) # red BGR order
        cv2.imwrite('map.png'.format(seq), traj)


    cv2.imwrite('folder_for_viz/sparse_prev_loss_9/mot_{:02}.png'.format(seq), traj)

    # np.savetxt('/data/silkimpl/SiLKimpl_pose_2/ISAP_mot_{:02}.txt'.format(seq), pred_array[:-1])
    # np.savetxt('/data/silkimpl/SiLKimpl_pose_2/posehead_mot_{:02}.txt'.format(seq), posehead_array[:-1])
    # np.savetxt('/data/silkimpl/SiLKimpl_pose_2/gt_mot_{:02}.txt'.format(seq), gt_array[:-1])


test_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]       

if __name__ == '__main__':
    fps=20

    for seq in test_seqs:
        # fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        # out = cv2.VideoWriter('/data/silkimpl/SiLKimpl_pose/mot_{:02}.avi'.format(seq), fourcc, fps, (416, 128))
        main(seq)
        # out.release()
        
    print("done")
