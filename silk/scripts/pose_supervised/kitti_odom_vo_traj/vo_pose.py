import sys
sys.path.append(r'/root/silk')

import numpy as np 
import cv2
from util import get_model, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords_1, from_feature_coords_to_image_coords_2

STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
MinNumFeature = 500

class VisualOdometry:
	def __init__(self, cam):
		self.frame_stage = 1
		self.cam = cam
		self.R = None
		self.t = np.array((0,0,0))
		self.focal = (cam.fx + cam.fy) / 2
		self.pp = (cam.cx, cam.cy)
		self.model = get_model(default_outputs=("sparse_positions_1", "sparse_descriptors_1", "sparse_positions_2", "sparse_descriptors_2", "pose6d"))

	def getAbsoluteScale(self):  #specialized for KITTI odometry dataset
		last_gtX = self.last_gt[0][3]
		last_gtY = self.last_gt[1][3]
		last_gtZ = self.last_gt[2][3]
		gtX = self.gt[0][3]
		gtY = self.gt[1][3]
		gtZ = self.gt[2][3]
		return np.sqrt((gtX - last_gtX)*(gtX - last_gtX)+(gtY - last_gtY)*(gtY - last_gtY)+(gtZ - last_gtZ)*(gtZ - last_gtZ))

	
	def processSecondFrame(self, img_1, img_2):
		positions_1, descriptors_1, positions_2, descriptors_2, pose6d = self.model(img_1, img_2)
		positions = from_feature_coords_to_image_coords_1(self.model, positions_1)
		self.curr_positions = positions[0]
	
	
		# print(len(self.curr_descriptors[0]), len(self.last_descriptors[0]))
		matches = SILK_MATCHER(descriptors_0[0], descriptors_1[0])
		# print("match length: ", len(matches))
		E, mask = cv2.findEssentialMat(
            self.curr_positions[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
            self.last_positions[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.curr_R, self.curr_t, mask = cv2.recoverPose(E, 
												self.curr_positions[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
            									self.last_positions[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
												focal=self.focal, pp = self.pp)
		self.last_positions, self.last_descriptors = self.curr_positions, self.curr_descriptors
		self.frame_stage = STAGE_DEFAULT_FRAME 

	def processFrame(self):
       
		positions, self.curr_descriptors = self.model(self.img)
		positions = from_feature_coords_to_image_coords(self.model, positions)
		self.curr_positions = positions[0]
		# this is in homogenous coord. why is this works?
        # homogenous coord now
        # print("one of the feature coords ", len(curr_positions), curr_positions[0].shape, curr_positions[0][0], curr_positions[0][10000])
        # 1 torch.Size([10001, 3]) tensor([  9.5000, 139.5000,   0.1875], device='cuda:0') tensor([3.6050e+02, 1.5350e+02, 1.7798e-01], device='cuda:0')
               
		matches = SILK_MATCHER(self.last_descriptors[0], self.curr_descriptors[0])
		E, mask = cv2.findEssentialMat(
			self.curr_positions[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
            self.last_positions[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		print("matches len: ", len(matches))
		if(E.shape[0]>3):
			E = E[:3]

		_, R, t, mask = cv2.recoverPose(E, 
							self.curr_positions[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
							self.last_positions[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
							focal=self.focal, pp = self.pp, mask = mask)
		absolute_scale = self.getAbsoluteScale()
		if(absolute_scale > 0.1):
			self.curr_t = self.curr_t + absolute_scale*self.curr_R.dot(t) 
			# self.curr_t = self.curr_t + self.curr_R.dot(t) 
			self.curr_R = R.dot(self.curr_R)
		else: 
			print("!!!!!!!!!!!!!!!!!!!!!under absolute scale!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		self.last_positions, self.last_descriptors = self.curr_positions, self.curr_descriptors

	def update(self, img, gt):
		self.img = img
		self.gt = gt
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame()
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_gt = self.gt


