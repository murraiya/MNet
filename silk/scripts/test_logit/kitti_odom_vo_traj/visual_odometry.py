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
from silk.models.sift import SIFT


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
		self.sift = SIFT("cuda:1")

	def getAbsoluteScale(self, abs_gt):  #specialized for KITTI odometry dataset
		last_gtX = abs_gt[1][0][3]
		last_gtY = abs_gt[1][1][3]
		last_gtZ = abs_gt[1][2][3]
		gtX = abs_gt[0][0][3]
		gtY = abs_gt[0][1][3]
		gtZ = abs_gt[0][2][3]

		return np.sqrt((gtX - last_gtX)*(gtX - last_gtX)+(gtY - last_gtY)*(gtY - last_gtY)+(gtZ - last_gtZ)*(gtZ - last_gtZ))

	# @torch.no_grad
	# def processSecondFrame(self, img, rel_gt, abs_gt):
	# 	positions_sift, descriptors_sift = self.sift(img)
	# 	positions, descriptors = self.model(img)
				
	# 	positions = from_feature_coords_to_image_coords(self.model, positions)

	# 	positions_1, positions_2 = positions[0], positions[1]
		
	# 	matches, dist = SILK_MATCHER(descriptors[0], descriptors[1])
	# 	matches_sift, dist_sift = SILK_MATCHER(descriptors_sift[0], descriptors_sift[1])
  
	# 	self.frames_num+=1
	# 	self.silk_macthes_len+=len(matches)
		
	# 	E, mask = cv2.findEssentialMat(
	# 		positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
	# 		positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
    #         focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
	# 	_, R, t, mask = cv2.recoverPose(E, 
   	# 		positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
	# 		positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
	# 		focal=self.focal, pp = self.pp)
		
	# 	E, mask = cv2.findEssentialMat(
	# 		positions_sift[1][matches_sift[:, 1]].detach().cpu().numpy()[:, [1,0]],
	# 		positions_sift[0][matches_sift[:, 0]].detach().cpu().numpy()[:, [1,0]],
    #         focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
	# 	_, R_sift, t_sift, mask = cv2.recoverPose(E, 
   	# 		positions_sift[1][matches_sift[:, 1]].detach().cpu().numpy()[:, [1,0]],
	# 		positions_sift[0][matches_sift[:, 0]].detach().cpu().numpy()[:, [1,0]],
	# 		focal=self.focal, pp = self.pp)
		
		

	# 	self.frame_stage = STAGE_DEFAULT_FRAME 
	# 	absolute_scale = self.getAbsoluteScale(abs_gt)
		
	# 	return R, absolute_scale*t, R_sift, absolute_scale*t_sift


	@torch.no_grad
	def processFrame(self, img, rel_gt, abs_gt):
		positions_sift, descriptors_sift = self.sift(img)
		positions, descriptors = self.model(img)
		print("sift number", len(positions_sift), len(positions_sift[0]))
		positions = from_feature_coords_to_image_coords(self.model, positions)
		positions_1, positions_2 = positions[0], positions[1]
		
		matches, dist = SILK_MATCHER(descriptors[0], descriptors[1])
		matches_sift, dist_sift = SILK_MATCHER(descriptors_sift[0], descriptors_sift[1])

		self.frames_num+=1
		self.silk_macthes_len+=len(matches)

		E, mask_ = cv2.findEssentialMat(
			positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		if(E.shape[0]>3):
			E = E[:3]
		_, R, t, mask = cv2.recoverPose(E, #return R1->2, t1->2
   			positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
			focal=self.focal, pp = self.pp)
		
		E, mask = cv2.findEssentialMat(
			positions_sift[1][matches_sift[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_sift[0][matches_sift[:, 0]].detach().cpu().numpy()[:, [1,0]],
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R_sift, t_sift, mask = cv2.recoverPose(E, 
   			positions_sift[1][matches_sift[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_sift[0][matches_sift[:, 0]].detach().cpu().numpy()[:, [1,0]],
			focal=self.focal, pp = self.pp)

		absolute_scale = self.getAbsoluteScale(abs_gt)
		

		R_=R.copy()
		t_=(absolute_scale*R@t).copy()

		R_sift_ = R_sift.copy()
		# t_sift_ = R_sift@t_sift
		t_sift_ = absolute_scale*R_sift@t_sift
		

		return R_, t_, R_sift_, t_sift_

	def update(self, img, abs_gt, rel_gt, intrinsics):
		self.focal = (float(intrinsics[0,0]) + float(intrinsics[1,1])) / 2
		self.pp = (float(intrinsics[0,2]), float(intrinsics[1,2]))
		
		# print(self.abs_gt[0].shape)

		R_, t_, R_sift, t_sift = self.processFrame(img, rel_gt, abs_gt)
		return R_, t_, R_sift, t_sift 
   


