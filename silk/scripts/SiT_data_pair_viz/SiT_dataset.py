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


def load_txt(path, delimiter = ',', reshape = None):
    data = np.loadtxt(path, delimiter=delimiter)
    if reshape is None:
        return data.astype(np.float32)
    else:
        return data.astype(np.float32).reshape(reshape)


class test_framework_SiT(object):
    def __init__(self, root, mode = 'test'):
        self.root = root
        self.samples = None
        self.frame_path_1 = self.root + "/cam_img/1/data_blur/"
        self.calib_path = self.root + "/calib/"
        self.relative_pose_path = self.root+"/ego_trajectory/"
        self.absolute_pose_path = self.root+"/rtk/"
        self.crawl_folders()

    def generator(self):
        for sample in self.samples:
            # print("generator: ", type(sample["img_path"]), len(sample["img_path"]), type(sample["img_path"][0])) # <class 'list'> 1 <class 'str'>
            
            # for silk inference
            img = load_images(sample["img_path"][0])

            # for optical flow
            # img = cv2.imread(sample["img_path"][0], 0) 
            yield {
                'img': img,
                'path': sample["img_path"][0],
                # 'pose': sample["pose"],
                'relative_Trs': sample["relative_Trs"]
            }
            
    def __iter__(self):
        return self.generator()
    
    def __len__(self):
        return len(self.samples)


    def crawl_folders(self):
        self.samples = []
        
        for i in range(200): # 200 frames they have
            sample = { # dict type
                "img_path": [],
                "pose": [],
                "relative_Trs": [],
                "frame_ids": [],
                "calib": [],
            }
            # calib = load_txt(self.calib_path + "/%s.txt" % i)
            # pose = load_txt(self.absolute_pose_path + "%s.txt" % i, delimiter=' ')
            # print(pose.shape)
            relative_Trs = load_txt(self.relative_pose_path + "%s.txt" % i, delimiter=',', reshape=(4, 4))
            print(relative_Trs.shape)
            img_file = self.frame_path_1 + "%s.png" % i
            sample["img_path"].append(img_file)
            # sample["pose"].append(pose)
            sample["relative_Trs"].append(relative_Trs)
            # sample["calib"].append(calib)
            self.samples.append(sample)
        print(len(self.samples))
        print(self.samples[0])
        print("====================fin crawl folders========================")
        


if __name__ == "__main__":
    print("test dataset loader for inference")
    # DATASET_PATH = "/data/SiT/Outdoor_Alley_3-003"
    DATASET_PATH = "/data/SiT/Three_way_Intersection_4-005"
    OUTPUT_IMAGE_PATH = "./inference_data_loader_img.png"
    OUTPUT_TRAJ_PATH = "./img.png"

    traj = np.zeros((1500,1500,3), dtype=np.uint8)

    framework = test_framework_SiT(DATASET_PATH)
    print('{} files to test'.format(len(framework))) # 1591 files to test

    for sample in tqdm(framework):
        img = sample['img']
        gt = sample['relative_Trs'][0]
        print(gt)
        # print(vo.gt[0][3], vo.gt[1][3], vo.gt[2][3])
        true_x, true_y = int(10*gt[0][3])+290, int(10*gt[2][3])+900


        cv2.circle(traj, (true_x,true_y), 1, (0, 0, 255), 2) # GT in red
        # cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        # text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
        # cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        cv2.imwrite('map.png', traj)

        # print(type(img))
        # print(img.shape)
        # io.imsave("./test_dataset_loader.png", img[0].squeeze().detach().cpu().numpy())