import sys
sys.path.append(r'/root/silk')

import numpy as np 
import cv2
from util import convert_points_from_homogeneous, get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords
# coordinate mapping is from: model.coordinate_mapping_composer.get
from silk.cli.image_pair_visualization import create_img_pair_visual, save_image



STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
MinNumFeature = 500
kMinNumFeature = 1500

lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

class PinholeCamera:
	def __init__(self, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]

class VisualOdometry:
	def __init__(self, cam):
		self.frame_stage = 0
		self.cam = cam
		self.curr_R = None
		self.curr_t = np.array((0,0,0))
		self.focal = (cam.fx + cam.fy) / 2
		self.pp = (cam.cx, cam.cy)
		self.model = get_model(default_outputs=("sparse_positions", "sparse_descriptors"))
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

	def getAbsoluteScale(self):  #specialized for KITTI odometry dataset
		last_gtX = self.last_gt[0][3]
		last_gtY = self.last_gt[1][3]
		last_gtZ = self.last_gt[2][3]
		gtX = self.gt[0][3]
		gtY = self.gt[1][3]
		gtZ = self.gt[2][3]
		return np.sqrt((gtX - last_gtX)*(gtX - last_gtX)+(gtY - last_gtY)*(gtY - last_gtY)+(gtZ - last_gtZ)*(gtZ - last_gtZ))

	def processFirstFrame(self):
		positions, self.last_descriptors = self.model(self.img)
		positions = from_feature_coords_to_image_coords(self.model, positions)
		self.last_positions = positions[0]
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		positions, self.curr_descriptors = self.model(self.img)
		positions = from_feature_coords_to_image_coords(self.model, positions)
		self.curr_positions = positions[0]
	
	
		# print(len(self.curr_descriptors[0]), len(self.last_descriptors[0]))
		matches = SILK_MATCHER(self.last_descriptors[0], self.curr_descriptors[0])
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
		
		# no abs scale check. bc almost every scene moved so small distance 
		absolute_scale = self.getAbsoluteScale()
		self.curr_t = self.curr_t + absolute_scale*self.curr_R.dot(t) 
		self.curr_R = R.dot(self.curr_R)
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


