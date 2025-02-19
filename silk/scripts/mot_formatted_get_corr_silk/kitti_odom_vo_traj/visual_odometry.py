import sys
sys.path.append(r'/root/silk')

import numpy as np 
import cv2
import torch
from util import get_model, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords, from_logit_coords_to_image_coords
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image
from geometry_utils import pose_vec2mat
from silk.losses.sfmlearner.sfm_loss import pose_vec2mat

STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2

import sys
sys.path.append(r'/root/silk')

class VisualOdometry:
	def __init__(self):
		self.frame_stage = 1
		self.R = None
		self.t = np.array((0,0,0))
		self.model = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))
		# self.model = get_model(default_outputs=("pose6d", "sparse_positions", "sparse_descriptors"))
		self.frames_num = 0
		self.silk_macthes_len = 0
		self.silk_macthes_len = 0
  
	def relative_pose_cam_to_body(
        self, relative_scene_pose
	):
		""" transform the camera pose from camera coordinate to body coordinate
		"""
		# return relative_scene_pose @ Rt_cam2_gt                
		return np.linalg.inv(self.Rt_cam2_gt) @ relative_scene_pose @ self.Rt_cam2_gt
        
		
	def getAbsoluteScale(self, abs_gt):  #specialized for KITTI odometry dataset
		last_gtX = abs_gt[1][0][3]
		last_gtY = abs_gt[1][1][3]
		last_gtZ = abs_gt[1][2][3]
		gtX = abs_gt[0][0][3]
		gtY = abs_gt[0][1][3]
		gtZ = abs_gt[0][2][3]

		return np.sqrt((gtX - last_gtX)*(gtX - last_gtX)+(gtY - last_gtY)*(gtY - last_gtY)+(gtZ - last_gtZ)*(gtZ - last_gtZ))

	@torch.no_grad
	def processSecondFrame(self, img, rel_gt, abs_gt):
		positions, descriptors = self.model(img)
		# pose6d, positions, descriptors = self.model(img)
				
		positions = from_feature_coords_to_image_coords(self.model, positions)
# 		print("==============")
# 		print(type(pose6d))
# 		print(len(positions), len(descriptors))
# 		print(positions[0].shape, positions[1].shape, descriptors[0].shape, descriptors[1].shape)
# torch.Size([3001, 3]) torch.Size([3001, 3]) torch.Size([3001, 128]) torch.Size([3001, 128])

		positions_1, positions_2 = positions[0], positions[1]
		
		matches, dist = SILK_MATCHER(descriptors[0], descriptors[1])
  
		self.frames_num+=1
		self.silk_macthes_len+=len(matches)
		
		E, mask = cv2.findEssentialMat(
			positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, 
   			positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
			focal=self.focal, pp = self.pp)
		
		self.rel_gt = rel_gt #4x4 ndarray
		self.frame_stage = STAGE_DEFAULT_FRAME 
		absolute_scale = self.getAbsoluteScale(abs_gt)
		
		R_ = (R).copy()
		t_ = (absolute_scale*t).copy()

		# pose6d = pose6d[:1, :, :][0] # pose6d of non-warped image, should be (1, 1, 6)
		
		# pose_mat = pose_vec2mat(pose6d, rotation_mode="euler")[0] #3x4
		# R = pose_mat[:3,:3].cpu().numpy()
		# t = pose_mat[:3, 3].unsqueeze(1).cpu().numpy()
		
		# curr_scale = np.sqrt(np.sum(t**2))
		# scale_factor = np.sum(gt[:,-1] * pred[:,-1])/np.sum(pred[:,-1] ** 2)
		# R__ = R.copy()
		# t__ = ((absolute_scale/curr_scale)*t).copy()
		
		return R_, t_ #, R__, t__


	@torch.no_grad
	def processFrame(self, img, rel_gt, abs_gt):
		positions, descriptors = self.model(img)
		# pose6d, positions, descriptors = self.model(img)
				
		positions = from_feature_coords_to_image_coords(self.model, positions)
# 		print("==============")
# 		print(type(pose6d))
		print(positions[0].shape, descriptors[0].shape)
		# torch.Size([3001, 3]) torch.Size([3001, 128])
		# this aligns
  
# 		print(positions[0].shape, positions[1].shape, descriptors[0].shape, descriptors[1].shape)
# torch.Size([3001, 3]) torch.Size([3001, 3]) torch.Size([3001, 128]) torch.Size([3001, 128])

		positions_1, positions_2 = positions[0], positions[1]
		
		matches, dist = SILK_MATCHER(descriptors[0], descriptors[1])
  
  		# print(dist[:10])
		# print(dist[300:310])
		# print(max(dist))#-0.4658,
		# print(min(dist))#-0.9875,
		# print("len(matches) ", len(matches))
		self.frames_num+=1
		self.silk_macthes_len+=len(matches)

		E, mask_ = cv2.findEssentialMat(
			positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
            # focal=self.focal, pp=self.pp)
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		if(E.shape[0]>3):
			E = E[:3]
		_, R, t, mask = cv2.recoverPose(E, #return R1->2, t1->2
   			positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
			focal=self.focal, pp = self.pp)
		
		absolute_scale = self.getAbsoluteScale(abs_gt)
		

		R_=R.copy()
		t_=(absolute_scale*R@t).copy()
		
		# pose6d = pose6d[:1, :, :][0] # pose6d of non-warped image, should be (1, 1, 6)
		# pose_mat = pose_vec2mat(pose6d, rotation_mode="euler")[0] #3x4
		# R = pose_mat[:3,:3].cpu().numpy()
		# t = pose_mat[:3, 3].unsqueeze(1).cpu().numpy()
		# curr_scale = np.sqrt(np.sum(t**2))
		# R__ = R.copy()
		# t__ = ((absolute_scale/curr_scale)* R @ t).copy()

		return R_, t_ #, R__, t__

	def update(self, img, abs_gt, rel_gt, intrinsics):
		self.focal = (float(intrinsics[0,0]) + float(intrinsics[1,1])) / 2
		self.pp = (float(intrinsics[0,2]), float(intrinsics[1,2]))
		
		# print(self.abs_gt[0].shape)

		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			R_, t_ = self.processFrame(img, rel_gt, abs_gt)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			R_, t_ = self.processSecondFrame(img, rel_gt, abs_gt)
		return R_, t_, #R, t
   


