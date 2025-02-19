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
from formatted_kitti_odom_dataset import test_framework_KITTI_fomatted_odom
from silk.cli.image_pair_visualization import save_image
from vo import VisualOdometry

formatted_odom_DATASET_PATH = "/data/plzdontremove/formatted_kitti_odom/"
# DATASET_PATH = "/data/kitti_odom_color_2012_dataset"
# sfm_DATASET_PATH = "/data/MOTkitti/formatted/test"

OUTPUT_IMAGE_PATH = "./img.png"


# silk.silk.models.silk line 39 uncomment plz
def main():
    traj = np.zeros((1500,1500,3), dtype=np.uint8)
    # framework = kittiMOTdataset(sfm_DATASET_PATH)
    framework = test_framework_KITTI_fomatted_odom(formatted_odom_DATASET_PATH)
    print('{} files to test'.format(len(framework))) # 1591 files to test
    

    # cam = PinholeCamera(715.7424, 718.5575, 609.2514, 186.07974)
    # (7.070912000000e+02, 7.070912000000e+02, 6.018873000000e+02, 1.831104000000e+02)
    
    vo = VisualOdometry()
    errors = np.zeros((len(framework), 2), np.float32)
    pred_array = np.zeros((len(framework), 3))
    gt_array = np.zeros((len(framework), 3))
    last_silk_Rt_ = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # last_silk_t_ = np.array([[0],[0],[0]])
    j=-1
    for sample in tqdm(framework): #2790 files to test
        j+=1
        Rt_cam2_gt = sample['Rt_cam2_gt']
        image_1 = sample['images_1'].to("cuda:1")
        image_2 = sample['images_2'].to("cuda:1")
        intrinsic = torch.from_numpy(sample['intrinsic'])
        # print(Rt_cam2_gt) #4,4
        rel_gt = sample['rel_pose'][0] # np.ndarray 4x4
        abs_gt = sample['abs_pose'] # [0], [1] np.ndarray 4x4 transformation mat
        R_, t_ = vo.update(image_1, image_2, abs_gt, rel_gt, intrinsic, Rt_cam2_gt)
        # print(vo.silk_t)
        pred_array[j] = vo.silk_t[0][0], vo.silk_t[1][0], vo.silk_t[2][0]
        gt_array[j] = vo.abs_gt[0][0][3], vo.abs_gt[0][1][3], vo.abs_gt[0][2][3]
        
        abs_gt_x, abs_gt_z = int(vo.abs_gt[0][0][3])+290, int(vo.abs_gt[0][2][3])+900 # red
        silk_vo_x, silk_vo_z = int(vo.silk_t[0])+290, int(vo.silk_t[2])+900 # blue
                
        cv2.circle(traj, (abs_gt_x, abs_gt_z), 1, (0, 0, 255), 4) # red BGR order
        cv2.circle(traj, (silk_vo_x, silk_vo_z), 1, (255, 0, 0), 2) # blue
        cv2.imwrite('map.png', traj)
        
        # compensated_gt = np.stack([abs_gt[0][:3,:4], abs_gt[1][:3,:4]])
        # def relative_pose_cam_to_body(
        #     relative_scene_pose, Rt_cam2_gt
        # ):
        #     """ transform the camera pose from camera coordinate to body coordinate
        #     """
        #     # return np.linalg.inv(Rt_cam2_gt) @ relative_scene_pose 0.0567         
        #     # return relative_scene_pose @ Rt_cam2_gt    0.0561,
        #     # return np.linalg.inv(Rt_cam2_gt) @ relative_scene_pose @ Rt_cam2_gt 0.0267,
    
        
        silk_sjsj = np.concatenate([R_, t_], axis=1)
        pred_rel = np.vstack([silk_sjsj, np.array([[0,0,0,1]])]) # make it 4x4
        # pred_rel = relative_pose_cam_to_body(pred_rel, Rt_cam2_gt)[0]
        # silk_R_ = (vo.silk_R).copy()
        # silk_t_ = (vo.silk_t).copy()
        # silk_sjsj = np.concatenate([silk_R_, silk_t_], axis=1)
        # silk_sjsj = np.vstack([silk_sjsj, np.array([[0,0,0,1]])]) # make it 4x4
        # gt_rel = np.linalg.inv(abs_gt[0]) @ abs_gt[1]
        # pred_rel = np.linalg.inv(last_silk_Rt_)@silk_sjsj
        
        # pred_rel = relative_pose_cam_to_body(pred_rel, Rt_cam2_gt)[0]
        # rel_err = np.linalg.inv(gt_rel)@pred_rel
        # ATE = translation_error(rel_err)
        # RE = rotation_error(rel_err)
        
        ATE, RE = compute_pose_error(np.expand_dims(rel_gt[:3,:4], axis=0), np.expand_dims(pred_rel[:3,:4], axis=0))
        errors[j] = ATE, RE
        last_silk_Rt_ = (silk_sjsj).copy()
        
    mean_errors = errors.mean(0)
    std_errors = errors.std(0)
    error_names = ['ATE','RE']
    print('')
    print("Results")
    print("\t {:>10}, {:>10}".format(*error_names))
    print("mean \t {:10.4f}, {:10.4f}".format(*mean_errors))
    print("std \t {:10.4f}, {:10.4f}".format(*std_errors))
    
    np.save('/data/ISAP_data_fig_table/ours_pred_odom_10.npy', pred_array)
    np.save('/data/ISAP_data_fig_table/ours_gt_odom_10.npy', gt_array)

    print("done")

# def rotation_error(pose_error):
#     a = pose_error[0, 0]
#     b = pose_error[1, 1]
#     c = pose_error[2, 2]
#     d = 0.5*(a+b+c-1.0)
#     rot_error = np.arccos(max(min(d, 1.0), -1.0))
#     return rot_error

# def translation_error(pose_error):
#     dx = pose_error[0, 3]
#     dy = pose_error[1, 3]
#     dz = pose_error[2, 3]
#     trans_error = np.sqrt(dx**2+dy**2+dz**2)
#     return trans_error

# def compute_rpe(gt, pred):
#     trans_errors = []
#     rot_errors = []
#     for i in range(len(gt)-1):
#         gt1 = gt[i]
#         gt2 = gt[i+1]
#         gt_rel = np.linalg.inv(gt1) @ gt2

#         pred1 = pred[i]
#         pred2 = pred[i+1]
#         pred_rel = np.linalg.inv(pred1) @ pred2
#         rel_err = np.linalg.inv(gt_rel) @ pred_rel
        
#         trans_errors.append(translation_error(rel_err))
#         rot_errors.append(rotation_error(rel_err))
#     rpe_trans = np.mean(np.asarray(trans_errors))
#     rpe_rot = np.mean(np.asarray(rot_errors))
#     return rpe_trans, rpe_rot


def compute_pose_error(gt, pred):
    RE = 0
    snippet_length = 1
    print(snippet_length)
    scale_factor = np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2)#0.0254
    # scale_factor = np.sqrt(gt[:,:,-1] ** 2) 0.0289, 0.0023
    # scale_factor = np.sqrt(gt[:,:,-1] ** 2)/np.sqrt(pred[:,:,-1] ** 2) 0.0116, 0.0023
    
    # scale_factor = np.sqrt(np.sum(gt[:,:,-1] ** 2)/np.sum(pred[:,:,-1] ** 2)) 0.0267, 0.0023
    # scale_factor = np.sqrt(np.sum(gt[:,:,-1] * pred[:,:,-1])/np.sum(pred[:,:,-1] ** 2))# nan, 0.0023
    # scale_factor = np.sqrt(np.sum(gt[:,:,-1] ** 2)) # 0.0267
    # scale_factor = np.sum(gt[:,:,-1] ** 2)
    # scale_factor = np.sqrt((np.sum(gt[:,:,-1] * pred[:,:,-1]))/(np.sum(pred[:,:,-1] ** 2)))

    
    print(gt[:,:,-1]**2)
    print(np.sum(gt[:,:,-1]**2))
    print(np.sqrt(np.sum(gt[:,:,-1]**2)))
    

    print("scale_factor")
    print(scale_factor)
    print("ATE")
    print(gt[:,:,-1])
    print(scale_factor * pred[:,:,-1])
    print(pred[:,:,-1])
    print((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    
    ATE = np.linalg.norm((gt[:,:,-1] - scale_factor * pred[:,:,-1]).reshape(-1))
    print(ATE)
    print("--------------------")

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
    # return ATE/snippet_length, RE/snippet_length



if __name__ == "__main__":
    main()
