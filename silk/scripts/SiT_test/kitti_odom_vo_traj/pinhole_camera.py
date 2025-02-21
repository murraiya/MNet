import sys
sys.path.append(r'/root/silk')

import numpy as np 
import cv2
from util import get_model, SILK_MATCHER
from silk.backbones.silk.silk import from_feature_coords_to_image_coords_1, from_feature_coords_to_image_coords_2
# coordinate mapping is from: model.coordinate_mapping_composer.get

class PinholeCamera:
	def __init__(self, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]
