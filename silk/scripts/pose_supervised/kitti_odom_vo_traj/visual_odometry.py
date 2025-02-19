import sys
sys.path.append(r'/root/silk')

import numpy as np 
import cv2
import torch
from util import get_model, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords_1, from_feature_coords_to_image_coords_2
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image
from geometry_utils import pose_vec2mat

STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2


class VisualOdometry:
	def __init__(self, cam):
		self.frame_stage = 1
		self.cam = cam
		self.R = None
		self.t = np.array((0,0,0))
		self.focal = (cam.fx + cam.fy) / 2
		self.pp = (cam.cx, cam.cy)
		self.model = get_model(default_outputs=("sparse_positions_1", "sparse_descriptors_1", "sparse_positions_2", "sparse_descriptors_2", "pose6d"))
		self.frames_num = 0
		self.silk_macthes_len = 0
		self.silk_macthes_len = 0
		
	def getAbsoluteScale(self):  #specialized for KITTI odometry dataset
		last_gtX = self.abs_gt[1][0][3]
		last_gtY = self.abs_gt[1][1][3]
		last_gtZ = self.abs_gt[1][2][3]
		gtX = self.abs_gt[0][0][3]
		gtY = self.abs_gt[0][1][3]
		gtZ = self.abs_gt[0][2][3]

		return np.sqrt((gtX - last_gtX)*(gtX - last_gtX)+(gtY - last_gtY)*(gtY - last_gtY)+(gtZ - last_gtZ)*(gtZ - last_gtZ))

	@torch.no_grad
	def processSecondFrame(self, img_1, img_2, rel_gt, intrinsic):
		positions_1, descriptors_1, positions_2, descriptors_2, predicted = self.model(img_1, img_2)
		predicted = pose_vec2mat(predicted[:1, :, :][0])[0].to("cpu")
		predicted = torch.cat([predicted, torch.tensor([[0,0,0,1]])], dim = 0) # make it 4x4
		self.pred = predicted

		positions_1 = from_feature_coords_to_image_coords_1(self.model, positions_1)
		positions_2 = from_feature_coords_to_image_coords_2(self.model, positions_2)
		
		positions_1, positions_2 = positions_1[0], positions_2[0]
	
		# print(len(self.descriptors[0]), len(self.last_descriptors[0]))
		matches = SILK_MATCHER(descriptors_1[0], descriptors_2[0])

		self.frames_num+=1
		self.silk_macthes_len+=len(matches)
		
		E, mask = cv2.findEssentialMat(
            positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
            positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.silk_R, self.silk_t, mask = cv2.recoverPose(E, 
			positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
			focal=self.focal, pp = self.pp)
		
		self.rel_gt = rel_gt #4x4 ndarray
		self.frame_stage = STAGE_DEFAULT_FRAME 

	@torch.no_grad
	def processFrame(self, img_1, img_2, rel_gt, intrinsic):
		# this is in homogenous coord. why is this works?
        # homogenous coord now
        # print("one of the feature coords ", len(curr_positions), curr_positions[0].shape, curr_positions[0][0], curr_positions[0][10000])
        # 1 torch.Size([10001, 3]) tensor([  9.5000, 139.5000,   0.1875], device='cuda:0') tensor([3.6050e+02, 1.5350e+02, 1.7798e-01], device='cuda:0')
		
		positions_1, descriptors_1, positions_2, descriptors_2, predicted = self.model(img_1, img_2)
		
		positions_1 = from_feature_coords_to_image_coords_1(self.model, positions_1)
		positions_2 = from_feature_coords_to_image_coords_2(self.model, positions_2)
		positions_1 = positions_1[0]
		positions_2 = positions_2[0] 

		matches = SILK_MATCHER(descriptors_1[0], descriptors_2[0])
		print("len(matches) ", len(matches))
		self.frames_num+=1
		self.silk_macthes_len+=len(matches)

		E, mask = cv2.findEssentialMat(
            positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
            positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		if(E.shape[0]>3):
			E = E[:3]
		_, R, t, mask = cv2.recoverPose(E, 
			positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
			focal=self.focal, pp = self.pp)
		# if (R==np.eye(3,3)).all:
		# 	print("fuck")
		# fuck my R are all eyes

		absolute_scale = self.getAbsoluteScale()
		self.silk_t = self.silk_t + absolute_scale*self.silk_R.dot(t) 
		self.silk_R = R.dot(self.silk_R)

		self.rel_gt = (self.rel_gt).dot(rel_gt) #this is so right!!!!!!!!!!!!!!!!
		print(predicted)
		print(predicted[:1, :, :][0])
		predicted = pose_vec2mat(predicted[:1, :, :][0])[0].to("cpu")
		print(predicted)
		rot_matrices = torch.linalg.inv(predicted[:3,:3])
		# print(rot_matrices)
		# print(predicted[:3,3])
		tr_vectors = -rot_matrices @ predicted[:3,3]
		# print(tr_vectors)
		# print(rot_matrices.shape)
		# print(tr_vectors.shape)
		predicted = torch.cat([rot_matrices, tr_vectors.unsqueeze(1)], axis=1)
		# print(predicted)
		predicted = torch.cat([predicted, torch.tensor([[0,0,0,1]])], dim = 0) # make it 4x4
		scale_factor = torch.sum(torch.from_numpy(rel_gt[:,-1]) * predicted[:,-1])/torch.sum(predicted[:,-1] ** 2)
		self.pred = self.pred@(scale_factor*predicted)
		# print(np.cat(self.pred))
	
	def update(self, img_1, img_2, abs_gt, rel_gt, intrinsic):
		self.abs_gt = abs_gt # list, 2, (4,4), (4,4)

		# print(intrinsic.shape)
		# print(intrinsic)
		# return 

		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(img_1, img_2, rel_gt, intrinsic)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame(img_1, img_2, rel_gt, intrinsic)


