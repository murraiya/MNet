# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch
import os
import cv2
import skimage.io as io
from tqdm import tqdm
from path import Path
from util import load_images_artifact
from silk.cli.image_pair_visualization import save_image

def load_as_array(path, dtype=None):
    array = np.load(path)
    if dtype is not None:
        return array.astype(dtype)
    else:
        return array

class test_framework_KITTI_artifact_formatted_odom(object):
    def __init__(self, root, mode = 'test'):
        self.root = root
        self.samples = None
        frame_list_path = self.root + "/test.txt"
        self.frames = [
            [self.root + "/" + frame[:-8], frame[-7:-1]] for frame in open(frame_list_path)
        ]
        self.crawl_folders()


    def generator(self):
        for sample in self.samples:
            img, cv_img, box = load_images_artifact(sample["img_path"][0])
            
            yield {
                'img': img,
                'cv_img': cv_img,
                'box': box,
                'path': sample["img_path"][0],
                'pose': sample["pose"]
            }
            
    def __iter__(self):
        return self.generator()
    
    def __len__(self):
        return len(self.samples)


    def crawl_folders(self, sequence_length = 0):
        sequence_set = []
        Ks = {}
        poses = {}
        Rt_cam2_gts = {}
        scenes = list(set([frame[0] for frame in self.frames]))
        # test scenes is 2. 1201, 1591 frames each.
        for scene in tqdm(scenes):
            Ks[scene] = (
                load_as_array(scene + "/cam.npy").astype(np.float32).reshape((3, 3))
            )
            poses[scene] = (
                load_as_array(scene + "/poses.npy")
                .astype(np.float32)
                .reshape(-1, 3, 4)
            )
            # print("poses len ", len(poses[scene]))
        for frame in tqdm(self.frames):
            # print("frame: ", frame) # ['/data/formatted_kitti_odom/09_02', '000000']
            scene = frame[0] # '/data/formatted_kitti_odom/09_02'
            frame_id = frame[1] # '000000'
            frame_num = int(frame_id)   
            
            sample = { # dict type
                "scene": scene,
                "img_path": [],
                "pose": [],
                "relative_scene_poses": [],
                "frame_ids": [],
                "ids": [],
            }
            img_file = scene + "/%s.jpg" % ("%06d" % frame_num)
            sample["img_path"].append(img_file)
            sample["pose"].append(poses[scene][frame_num])
            sequence_set.append(sample)
        self.samples = sequence_set
        

if __name__ == "__main__":
    print("test dataset loader for inference")
    DATASET_PATH = "/data/formatted_kitti_odom"
    SEQUENCE_NUM = "09" # 09 or 10 .. sfm did this
    OUTPUT_IMAGE_PATH = "./inference_data_loader_img.png"

    framework = test_framework_KITTI_artifact_formatted_odom(DATASET_PATH, SEQUENCE_NUM)
    print('{} files to test'.format(len(framework))) # 1591 files to test

    for sample in tqdm(framework):
        path = sample["path"]
        img = sample["img"]
        img = img[0].detach().cpu().numpy()
        io.imsave("./test_dataset_loader.png", img.squeeze())
        # img contains float32