import sys
sys.path.append(r'/root/silk')

import numpy as np 
import cv2
from util import convert_points_from_homogeneous, get_model, load_images, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords


STAGE_FIRST_FRAME = 0
STAGE_SECOND_FRAME = 1
STAGE_DEFAULT_FRAME = 2
MinNumFeature = 500
kMinNumFeature = 1500

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
		self.last_positions = convert_points_from_homogeneous(positions[0])
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		positions, self.curr_descriptors = self.model(self.img)
		positions = from_feature_coords_to_image_coords(self.model, positions)
		self.curr_positions = convert_points_from_homogeneous(positions[0])
		# print(positions)
		# (tensor([[9.5000e+00, 7.8500e+01, 2.2396e-01],
        # [9.5000e+00, 7.9500e+01, 2.2660e-01],
        # [9.5000e+00, 8.0500e+01, 2.5340e-01],
        # ...,
        # [3.6050e+02, 4.8350e+02, 2.2610e-01],
        # [3.6050e+02, 4.8450e+02, 2.4935e-01],
        # [3.6050e+02, 4.8550e+02, 2.2333e-01]], device='cuda:0'),)
		# print(self.curr_positions)
		# tensor([[  42.4191,  350.5158],
        # [  41.9233,  350.8316],
        # [  37.4901,  317.6791],
        # ...,
        # [1594.4502, 2138.4651],
        # [1445.7769, 1943.0759],
        # [1614.1733, 2173.8728]], device='cuda:0')
		# print(self.curr_descriptors)
		# (tensor([[-0.0189,  0.1544,  0.0483,  ...,  0.0959,  0.0744, -0.1022],
        # [ 0.0940,  0.2579, -0.0076,  ...,  0.0846,  0.0986, -0.1485],
        # [ 0.1486,  0.2700, -0.0858,  ..., -0.0171,  0.1595, -0.2378],
        # ...,
        # [-0.0341,  0.0972, -0.0529,  ...,  0.0778,  0.2352, -0.3302],
        # [ 0.0735,  0.2372, -0.0033,  ...,  0.0369,  0.0906, -0.3515],
        # [ 0.1860,  0.2727, -0.0309,  ...,  0.0466,  0.1169, -0.3144]], device='cuda:0'),)
		print(len(self.curr_descriptors[0]), len(self.last_descriptors[0]))
		matches = SILK_MATCHER(self.last_descriptors[0], self.curr_descriptors[0])
		print(len(matches))
		E, mask = cv2.findEssentialMat(
            self.curr_positions[matches[:, 0]].detach().cpu().numpy(),
            self.last_positions[matches[:, 0]].detach().cpu().numpy(),
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.curr_R, self.curr_t, mask = cv2.recoverPose(E, 
												self.curr_positions[matches[:, 0]].detach().cpu().numpy(),
            									self.last_positions[matches[:, 0]].detach().cpu().numpy(),
												focal=self.focal, pp = self.pp)
		self.last_positions, self.last_descriptors = self.curr_positions[matches[:, 0]], self.curr_descriptors[matches[:, 0]]
		#TODO here desc cant have matches? or!!!!!!!! detach when assign.
		
		self.frame_stage = STAGE_DEFAULT_FRAME 

	def stoptracking(self, rematch: bool):
		positions, self.last_descriptors = self.model(self.img)
		positions = from_feature_coords_to_image_coords(self.model, positions)
		self.last_positions = convert_points_from_homogeneous(positions[0])
		if rematch==True:
			self.matches = SILK_MATCHER(self.last_descriptors[0], self.curr_descriptors[0])

	def processFrame(self):
		if(self.last_positions.shape[0] < 100):
			self.stoptracking()

		positions, self.curr_descriptors = self.model(self.img)
		positions = from_feature_coords_to_image_coords(self.model, positions)
        # homogenous coord now
        # print("one of the feature coords ", len(curr_positions), curr_positions[0].shape, curr_positions[0][0], curr_positions[0][10000])
        # 1 torch.Size([10001, 3]) tensor([  9.5000, 139.5000,   0.1875], device='cuda:0') tensor([3.6050e+02, 1.5350e+02, 1.7798e-01], device='cuda:0')
        # print("desc shape ", len(curr_descriptors), curr_descriptors[0].shape)
        # 1 torch.Size([10001, 128])
		self.curr_positions = convert_points_from_homogeneous(positions[0])
        # print(curr_positions_[0])        
        # tensor([ 50.6605, 743.9100]
		# print(self.last_descriptors)
		# print(self.curr_descriptors)
		# print(self.last_descriptors[0])
		# print(self.curr_descriptors[0])
		
		self.matches = SILK_MATCHER(self.last_descriptors[0], self.curr_descriptors[0])
		if len(self.matches) < 10:
			self.stoptracking(rematch=True)

		E, mask = cv2.findEssentialMat(
            self.curr_positions[self.matches[:, 0]].detach().cpu().numpy(),
            self.last_positions[self.matches[:, 0]].detach().cpu().numpy(),
            focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		print("matches len: ", len(self.matches))
		# print(E)
		if(E.shape[0]>3):
			E = E[:3]

		# print(self.curr_positions)
		# tensor([[  44.7296, 1160.6145],
        # [  45.1738, 1176.8973],
        # [  45.4433, 1341.7725],
        # ...,
        # [1616.2306, 2302.1758],
        # [1595.2845, 2276.7654],
        # [1752.8738, 2506.5366]], device='cuda:0')

		# print(self.curr_positions[0])
		# tensor([  44.7296, 1160.6145], device='cuda:0')
		# print(self.curr_positions[matches[:, 0]].detach().cpu().numpy())		
		# print(matches[:, 0])


		_, R, t, mask = cv2.recoverPose(E, 
							self.curr_positions[self.matches[:, 0]].detach().cpu().numpy(),
							self.last_positions[self.matches[:, 0]].detach().cpu().numpy(),
							focal=self.focal, pp = self.pp, mask = mask)
		absolute_scale = self.getAbsoluteScale()
		if(absolute_scale > 0.1):
			self.curr_t = self.curr_t + absolute_scale*self.curr_R.dot(t) 
			self.curr_R = R.dot(self.curr_R)
			
		self.last_positions, self.last_descriptors = self.curr_positions[self.matches[:, 0]], self.curr_descriptors[matches[:, 0]]

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


