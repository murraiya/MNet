# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch
import os
import cv2
import torch.utils.data as data
import skimage.io as io
from tqdm import tqdm
from path import Path
from silk.datasets.formatted_kitti_odom.util import load_images
from silk.cli.image_pair_visualization import save_image

def load_as_array(path, dtype=None):
    array = np.load(path)
    if dtype is not None:
        return array.astype(dtype)
    else:
        return array



class formattedKittiOdom(data.Dataset):
    def __init__(self, root, train=True):
        self.samples = None
        self.root = Path(root)
        frame_list_path = self.root/'train.txt' if train else self.root/'val.txt'        
        self.frames = [[self.root + "/" + frame[:-8], frame[-7:-1]] for frame in open(frame_list_path)]
        # print("self.frames: ", self.frames)
        self.crawl_folders()


    # def generator(self):
    #     for sample in self.samples:
    #         # print("generator: ", type(sample["img_path"]), len(sample["img_path"]), type(sample["img_path"][0])) # <class 'list'> 1 <class 'str'>
    #         img = load_images(sample["img_path"][0])
            
    #         yield {
    #             'img': img,
    #             'path': sample["img_path"][0],
    #             'pose': sample["pose"]
    #         }
            
    # def __iter__(self):
    #     return self.generator()
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        img = io.imread(sample["img_path"][0])               
        # print("dataloader: img ", type(img), type(img[0][0][0]), img.shape) # 128x416x3
        
        io.imsave("./from_dataloader.png", img)
        return img, None #make it tuple

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
            # print(len(sample), len(sample["img"]), len(sample["pose"]), sample["pose"][0].shape) # 6 1 1 (3, 4)
            # print(sample["img"], sample["pose"]) # ['/data/formatted_kitti_odom/10_02/001200.jpg'] [array([[-7.561071e-01, -2.709085e-02, -6.538869e-01,  5.452426e+02], ...
        self.samples = sequence_set
        # print(self.samples[0]["img"], self.samples[0]["pose"])
        # print(self.samples[1]["img"], self.samples[1]["pose"])
        # right! confirmed with original data



# if __name__ == "__main__":
#     print("test dataset loader for inference")
#     DATASET_PATH = "/data/kitti_odom_color_2012_dataset"
#     SEQUENCE_NUM = "09" # 09 or 10 .. sfm did this
#     OUTPUT_IMAGE_PATH = "./inference_data_loader_img.png"

#     framework = test_framework_KITTI(DATASET_PATH, SEQUENCE_NUM)
#     print('{} files to test'.format(len(framework))) # 1591 files to test

#     for sample in tqdm(framework):
#         path = sample['path']
#         print(path)
#         # img = cv2.imread(path)
#         # io.imsave("./test_dataset_loader.png", img)