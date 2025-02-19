# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch
import os
import cv2
import skimage.io as io
from tqdm import tqdm
from path import Path
from util import load_images
from silk.cli.image_pair_visualization import save_image

def load_as_array(path, dtype=None):
    array = np.load(path)
    if dtype is not None:
        return array.astype(dtype)
    else:
        return array

class test_framework_KITTI_fomatted_odom(object):
    def __init__(self, root):
        self.root = Path(root)
        self.samples = None
        self.test_seqs = [9] #10]
        print("root: ", root)
        self.scenes = [self.root + '{:02d}_02'.format(seq) for seq in self.test_seqs]
        self.crawl_folders()

    

    def generator(self):
        for sample in self.samples:
            img1 = load_images(sample["img_path"][0][0])      
            img2 = load_images(sample["img_path"][0][1])
            yield {
                'images_1': img1,
                'images_2': img2,
                'path': sample["img_path"][0],
                'rel_pose': sample["rel_pose"][0],
                'abs_pose': sample["abs_pose"][0], 
                'intrinsic': sample["intrinsics"][0],
                'sfmformatted': sample["sfmformatted"],
            }
               
    def __iter__(self):
        return self.generator()
    
    def __len__(self):
        return len(self.samples)


    def crawl_folders(self):
        sequence_set = []

        for scene in tqdm(self.scenes):
            poses = load_as_array(scene+'/poses.npy').reshape((-1, 3, 4))
            poses_4D = np.zeros((poses.shape[0], 4, 4)).astype(np.float32)
            poses_4D[:, :3] = poses
            poses_4D[:, 3, 3] = 1
            intrinsics = load_as_array(scene+'/cam.npy').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg')) #sorted sequence of images
            assert(len(imgs) == poses.shape[0])
            
            i = -1
            last_frame = None
            # print(len(imgs), poses.shape) # 59 (59, 3, 4)
            # print(intrinsics.shape) #(3, 3)
            for image in imgs:
                i+=1 #idx for pose, intrinsics
                if last_frame is None:
                    last_frame = image
                    continue

                sample = { # dict type
                    "scene": scene, #str
                    "img_path": [], # two str
                    "abs_pose": [], # abs_pose
                    "rel_pose": [],
                    "intrinsics": [],
                    "sfmformatted": []
                }

                new_frame = image
                seq_frame = [last_frame, new_frame]
                sample["img_path"].append(seq_frame)   

                last_pose = poses_4D[i-1]
                new_pose = poses_4D[i]
                pose_forward = np.linalg.inv(last_pose) @ new_pose #1->2
                pose_backward = np.linalg.inv(new_pose) @ last_pose #2->1
                compensated_pose = (new_pose[:3,:4]).copy()
                compensated_pose[:3,3] -= last_pose[:3,3]
                compensated_pose = np.linalg.inv(last_pose[:3,:3]) @ compensated_pose
                sample["sfmformatted"].append(compensated_pose)
                sample["abs_pose"].append([last_pose, new_pose])
                sample["rel_pose"].append([pose_forward, pose_backward]) 
                sample["intrinsics"].append(intrinsics)
                sequence_set.append(sample)
                last_frame = new_frame
                # print(sample["img_path"])
                # print(sample["depth_map_path"])
                # print("-------------------")
    
        self.samples = sequence_set
       
if __name__ == "__main__":
    print("test dataset loader for inference")
    formatted_odom_DATASET_PATH = "/data/formatted_kitti_odom"

    SEQUENCE_NUM = "09" # 09 or 10 .. sfm did this
    OUTPUT_IMAGE_PATH = "./inference_data_loader_img.png"

    framework = test_framework_KITTI_fomatted_odom(formatted_odom_DATASET_PATH, SEQUENCE_NUM)
    print('{} files to test'.format(len(framework))) # 1591 files to test

    # for sample in tqdm(framework):
    #     path = sample['path']
    #     intrinsic = sample['cam']
        # print(intrinsic)
        # img = cv2.imread(path)
        # io.imsave("./test_dataset_loader.png", img)