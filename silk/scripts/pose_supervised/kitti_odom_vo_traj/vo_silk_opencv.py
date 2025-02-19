import sys
sys.path.append(r'/root/silk')

import numpy as np 
import cv2
from util import get_model, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords_1, from_feature_coords_to_image_coords_2
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image

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
		self.model = get_model(default_outputs=("sparse_positions_1", "sparse_descriptors_1", "sparse_positions_2", "sparse_descriptors_2"))
		self.frames_num = 0
		self.macthes_len = 0
		fourcc = cv2.VideoWriter_fourcc('M','P','E','G')
		self.out = cv2.VideoWriter('/data/script_pair_viz.avi', fourcc, 10.0, (2482, 376))
	
	def __del__(self):
		self.out.release()

	def getAbsoluteScale(self):  #specialized for KITTI odometry dataset
		last_gtX = self.gt[1][0][3]
		last_gtY = self.gt[1][1][3]
		last_gtZ = self.gt[1][2][3]
		gtX = self.gt[0][0][3]
		gtY = self.gt[0][1][3]
		gtZ = self.gt[0][2][3]
		# print("gt R norm: ", np.linalg.norm(self.gt[0][:3,:3]))

		return np.sqrt((gtX - last_gtX)*(gtX - last_gtX)+(gtY - last_gtY)*(gtY - last_gtY)+(gtZ - last_gtZ)*(gtZ - last_gtZ))

	def processSecondFrame(self, img_1, img_2):
		positions_1, descriptors_1, positions_2, descriptors_2 = self.model(img_1, img_2)
		positions_1 = from_feature_coords_to_image_coords_1(self.model, positions_1)
		positions_2 = from_feature_coords_to_image_coords_2(self.model, positions_2)
		
		positions_1, positions_2 = positions_1[0], positions_2[0]
	
		# print(len(self.descriptors[0]), len(self.last_descriptors[0]))
		matches = SILK_MATCHER(descriptors_1[0], descriptors_2[0])
		image_pair = create_img_pair_visual(self.path[0], self.path[1], None, None,
			positions_1[matches[:,0]].detach().cpu().numpy(),
			positions_2[matches[:,1]].detach().cpu().numpy())
		
		self.out.write(image_pair)

		self.frames_num+=1
		self.macthes_len+=len(matches)
		
		E, mask = cv2.findEssentialMat(
            positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
            positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.R, self.t, mask = cv2.recoverPose(E, 
			positions_2[matches[:, 1]].detach().cpu().numpy()[:, [1,0]],
			positions_1[matches[:, 0]].detach().cpu().numpy()[:, [1,0]],
			focal=self.focal, pp = self.pp)
		
		self.frame_stage = STAGE_DEFAULT_FRAME 

	def processFrame(self, img_1, img_2):
		# this is in homogenous coord. why is this works?
        # homogenous coord now
        # print("one of the feature coords ", len(curr_positions), curr_positions[0].shape, curr_positions[0][0], curr_positions[0][10000])
        # 1 torch.Size([10001, 3]) tensor([  9.5000, 139.5000,   0.1875], device='cuda:0') tensor([3.6050e+02, 1.5350e+02, 1.7798e-01], device='cuda:0')
		positions_1, descriptors_1, positions_2, descriptors_2 = self.model(img_1, img_2)
		positions_1 = from_feature_coords_to_image_coords_1(self.model, positions_1)
		positions_2 = from_feature_coords_to_image_coords_2(self.model, positions_2)
		
		positions_1 = positions_1[0]
		positions_2 = positions_2[0] 

		matches = SILK_MATCHER(descriptors_1[0], descriptors_2[0])

		image_pair = create_img_pair_visual(self.path[0], self.path[1], None, None,
		positions_1[matches[:,0]].detach().cpu().numpy(),
		positions_2[matches[:,1]].detach().cpu().numpy())
		
		self.out.write(image_pair)

		self.frames_num+=1
		self.macthes_len+=len(matches)

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
		if (R==np.eye(3,3)).all:
			print("fuck")
		# fuck my R are all eyes

		# no abs scale check. bc almost every scene moved so small distance 
		absolute_scale = self.getAbsoluteScale()
		self.t = self.t + absolute_scale*self.R.dot(t) 
		self.R = R.dot(self.R)

		# absolute_scale = self.getAbsoluteScale()
		# if(absolute_scale > 0.1):
		# 	self.t = self.t + absolute_scale*self.R.dot(t) 
		# 	# self.t = self.t + self.R.dot(t) 
		# 	self.R = R.dot(self.R)
		# else: 
		# 	print("!!!!!!!!!!!!!!!!!!!!!under absolute scale!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


	def update(self, img_1, img_2, gt, path):
		self.gt = gt # list, 2, (4,4), (4,4)
		# print(type(gt), len(gt), gt[0].shape, gt[1].shape)
		# print(gt[0], gt[1])
		self.path = path
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame(img_1, img_2)
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame(img_1, img_2)


