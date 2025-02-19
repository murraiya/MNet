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
# from sfm_kitti_odom_dataset import test_framework_sfm_KITTI
from sfm_kitti_mot_dataset import kittiMOTdataset
from formatted_kitti_odom_dataset import test_framework_KITTI_fomatted_odom
from silk.cli.image_pair_visualization import save_image
from visual_odometry import PinholeCamera, VisualOdometry

formatted_odom_DATASET_PATH = "/data/plzdontremove/formatted_kitti_odom/"
# DATASET_PATH = "/data/kitti_odom_color_2012_dataset"
sfm_DATASET_PATH = "/data/MOTkitti/formatted/test"

OUTPUT_IMAGE_PATH = "./img.png"


# silk.silk.models.silk line 39 uncomment plz
def main():
    traj = np.zeros((1500,1500,3), dtype=np.uint8)
    framework = kittiMOTdataset(sfm_DATASET_PATH)
    # framework = test_framework_KITTI_fomatted_odom(formatted_odom_DATASET_PATH)
    print('{} files to test'.format(len(framework))) # 1591 files to test
    
    
    # cam = PinholeCamera(715.7424, 718.5575, 609.2514, 186.07974) 
    cam = PinholeCamera(241.6745, 246.2849, 204.1680,  59.0008)
    
    # vopose = VOpose(cam)
    # vocv = VOcv(cam)
    vo = VisualOdometry(cam)
    errors = np.zeros((len(framework), 2), np.float32)
    pred_array = np.zeros((len(framework), 3))
    gt_array = np.zeros((len(framework), 3))
    
    j=-1
    silk_last_R = np.array([[1,0,0],
                            [0,1,0],
                            [0,0,1]])
    silk_last_t = np.array([[0],[0],[0]])
    for sample in tqdm(framework): #2790 files to test
        j+=1
        # intrinsic = sample['cam']
        image_1 = sample['images_1'].to("cuda:1")
        image_2 = sample['images_2'].to("cuda:1")
        intrinsic = torch.from_numpy(sample['intrinsic'])
        print(intrinsic)
        path = sample['path']
        rel_gt = sample['rel_pose'][0] # np.ndarray 4x4
        abs_gt = sample['abs_pose'] # [0], [1] np.ndarray 4x4 transformation mat
        # print(path[0]) # confirmed that it is sequential
        # print(path[1])
        vo.update(image_1, image_2, abs_gt, rel_gt)
        # print(vo.silk_t)
        pred_array[j] = vo.silk_t[0][0], vo.silk_t[1][0], vo.silk_t[2][0]
        gt_array[j] = vo.abs_gt[0][0][3], vo.abs_gt[0][1][3], vo.abs_gt[0][2][3]
        
        abs_gt_x, abs_gt_z = int(vo.abs_gt[0][0][3])+290, int(vo.abs_gt[0][2][3])+900 # red
        silk_vo_x, silk_vo_z = int(vo.silk_t[0])+290, int(vo.silk_t[2])+900 # blue
                
        cv2.circle(traj, (abs_gt_x, abs_gt_z), 1, (0, 0, 255), 4) # red BGR order
        cv2.circle(traj, (silk_vo_x, silk_vo_z), 1, (255, 0, 0), 2) # blue
        cv2.imwrite('map.png', traj)
        
        abs_gt_0 = abs_gt[0].copy()
        abs_gt_1 = abs_gt[1].copy()
        abs_gt_1[:,-1] -= abs_gt_0[:,-1]
        abs_gt_0[:,-1] -= abs_gt_0[:,-1]
        compensated_gt_1 = np.linalg.inv(abs_gt_0[:3,:3]) @ abs_gt_1[:3, :4]
        compensated_gt_0 = np.linalg.inv(abs_gt_0[:3,:3]) @ abs_gt_0[:3, :4]
        compensated_gt = np.stack([compensated_gt_0, compensated_gt_1])
        
        
        R=(vo.silk_R).copy()
        t=(vo.silk_t).copy()
        silk_last_T = np.concatenate([silk_last_R.copy(), silk_last_t.copy()], axis=-1)
        silk_curr_T = np.concatenate([R, t], axis=-1)
        silk_curr_T[:,-1] -= silk_last_T[:,-1]
        silk_last_T[:,-1] -= silk_last_T[:,-1]
        compensated_silk_1 = np.linalg.inv(silk_last_T[:3,:3]) @ silk_curr_T
        compensated_silk_0 = np.linalg.inv(silk_last_T[:3,:3]) @ silk_last_T
        compensated_silk = np.stack([np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]]), compensated_silk_1])
        print(compensated_gt.shape, compensated_silk.shape) # 2, 3, 4
        
        silk_last_R = (vo.silk_R).copy()
        silk_last_t = (vo.silk_t).copy()
        print(compensated_gt)
        print(compensated_silk)
        ATE, RE = compute_pose_error(compensated_gt, compensated_silk)
        errors[j] = ATE, RE
        


    # print("avg matches len of silk+opencv vo: ", vo.silk_macthes_len/vo.frames_num)
    # print("rel gt & rel predicted l2 error of 4x4 matrix: ", vopose.error/vopose.frames_num)
    
    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))
    
    np.save('/data/ISAP_data_fig_table/ours_pred_11_mot.npy', pred_array)
    np.save('/data/ISAP_data_fig_table/ours_gt_11_mot.npy', gt_array)

    print("done")



def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = gt.shape[0]
    print(snippet_length)
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    for gt_pose, pred_pose in zip(gt, pred):
        # Residual matrix to which we compute angle's sin and cos
        R = gt_pose[:,:3] @ np.linalg.inv(pred_pose[:,:3])
        s = np.linalg.norm([R[0,1]-R[1,0],
                            R[1,2]-R[2,1],
                            R[0,2]-R[2,0]])
        c = np.trace(R) - 1
        # Note: we actually compute double of cos and sin, but arctan2 is invariant to scale
        RE += np.arctan2(s,c)

    return ATE/snippet_length, RE/snippet_length



if __name__ == "__main__":
    main()
