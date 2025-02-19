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
		self.detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)

	def getAbsoluteScale(self):  #specialized for KITTI odometry dataset
		last_gtX = self.last_gt[0][3]
		last_gtY = self.last_gt[1][3]
		last_gtZ = self.last_gt[2][3]
		gtX = self.gt[0][3]
		gtY = self.gt[1][3]
		gtZ = self.gt[2][3]
		return np.sqrt((gtX - last_gtX)*(gtX - last_gtX)+(gtY - last_gtY)*(gtY - last_gtY)+(gtZ - last_gtZ)*(gtZ - last_gtZ))

lk_params = dict(winSize  = (21, 21), 
				#maxLevel = 3,
             	criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

def featureTracking(image_ref, image_cur, px_ref):
	kp2, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, px_ref, None, **lk_params)  #shape: [k,2] [k,1] [k,1]

	st = st.reshape(st.shape[0])
	kp1 = px_ref[st == 1]
	kp2 = kp2[st == 1]

	return kp1, kp2


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
		self.new_frame = None
		self.last_frame = None
		self.curr_R = None
		self.curr_t = np.array((0,0,0))
		self.px_ref = None
		self.px_cur = None
		self.focal = cam.fx
		self.pp = (cam.cx, cam.cy)
		self.trueX, self.trueY, self.trueZ = 0, 0, 0
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
		self.px_ref = self.detector.detect(self.new_frame)
		print(len(self.px_ref))

		self.px_ref = np.array([x.pt for x in self.px_ref], dtype=np.float32)
		print(len(self.px_ref))
		self.frame_stage = STAGE_SECOND_FRAME

	def processSecondFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, self.curr_R, self.curr_t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		self.frame_stage = STAGE_DEFAULT_FRAME 
		self.px_ref = self.px_cur

	def processFrame(self):
		self.px_ref, self.px_cur = featureTracking(self.last_frame, self.new_frame, self.px_ref)
		E, mask = cv2.findEssentialMat(self.px_cur, self.px_ref, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
		_, R, t, mask = cv2.recoverPose(E, self.px_cur, self.px_ref, focal=self.focal, pp = self.pp)
		absolute_scale = self.getAbsoluteScale()
		if(absolute_scale > 0.1):
			self.curr_t = self.curr_t + absolute_scale*self.curr_R.dot(t) 
			self.curr_R = R.dot(self.curr_R)
		if(self.px_ref.shape[0] < kMinNumFeature):
			self.px_cur = self.detector.detect(self.new_frame)
			self.px_cur = np.array([x.pt for x in self.px_cur], dtype=np.float32)
		self.px_ref = self.px_cur

	def update(self, img, gt):
		# print(type(img), type(img[0]))
		# self.new_frame = cv2.UMat(img[0].detach().cpu().numpy())
		self.new_frame = img
		self.gt = gt
		if(self.frame_stage == STAGE_DEFAULT_FRAME):
			self.processFrame()
		elif(self.frame_stage == STAGE_SECOND_FRAME):
			self.processSecondFrame()
		elif(self.frame_stage == STAGE_FIRST_FRAME):
			self.processFirstFrame()
		self.last_frame = self.new_frame
		self.last_gt = self.gt