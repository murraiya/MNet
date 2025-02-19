# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch
import os
import cv2
import skimage.io as io
from tqdm import tqdm
from path import Path
from util import load_image
from silk.cli.image_pair_visualization import save_image


class test_framework_KITTI(object):
    def __init__(self, root, seq, width, height):
        self.root = root
        self.img_files, self.poses = read_scene_data(self.root, seq)
        self.width, self.height = width, height

    def generator(self):
        for img_list, pose in zip(self.img_files, self.poses):
            imgs = load_image(img_list, img_height = self.height, img_width = self.width)

            yield {
                'img': imgs,
                'path': img_list,
                'pose': pose
            }
            
    def __iter__(self):
        return self.generator()

    def __len__(self):
        return sum(len(imgs) for imgs in self.img_files)



def read_scene_data(data_root, sequence):
    im_sequences = []
    poses_sequences = []
    data_root = Path(data_root)
    img_dir = Path(data_root+'/sequences/'+sequence+'/image_2/')
    
    imgs = sum([list(img_dir.walkfiles('*.{}'.format("png")))], [])
    imgs.sort()
    # print('{} files to test'.format(len(test_files))) # 1591 files to test
    
    poses = np.genfromtxt(data_root/'poses'/'{}.txt'.format(sequence)).astype(np.float64).reshape(-1, 3, 4)
    # print(poses.shape) (1591, 3, 4)


    return imgs, poses



if __name__ == "__main__":
    print("test dataset loader for inference")
    DATASET_PATH = "/data/plzdontremove/kitti_odom_color_2012_dataset"
    SEQUENCE_NUM = "09" # 09 or 10 .. sfm did this
    OUTPUT_IMAGE_PATH = "./inference_data_loader_img.png"

    framework = test_framework_KITTI(DATASET_PATH, SEQUENCE_NUM)
    print('{} files to test'.format(len(framework))) # 1591 files to test

    for sample in tqdm(framework):
        path = sample['path']
