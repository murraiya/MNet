# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch
import os
import cv2
import skimage.io as io
from tqdm import tqdm
from path import Path
# from util import load_images
# from silk.datasets.formatted_kitti_odom.util import load_images
import torch.utils.data as data
# from silk.cli.image_pair_visualization import save_image

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


class SiT(data.Dataset):
    def __init__(self, root, train:bool):
        self.root = Path(root)
        self.scenes = []
        if train:
            folders = open("/data/SiT/ImageSets-20241017T155657Z-001/ImageSets/train.txt", 'r')
        else: 
            folders = open("/data/SiT/ImageSets-20241017T155657Z-001/ImageSets/val.txt", 'r')
        
        lines = folders.readlines()
        for line in lines:
            line = line.strip().split('*')
            # ['Three_way_Intersection', 'Three_way_Intersection_8', '199']
            self.scenes.append(self.root/line[1])

        self.crawl_folders()
         

    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            # we use P3_intrinsic, that corresponds to cam_img/4 (looking forward)
            intrinsics = open(scene/"calib/0.txt").readlines()
            # P3_intrinsic: 1187.21727 0.0 962.94357 0.0 1191.21681 588.05555 0.0 0.0 1.0
            intrinsics = intrinsics[3].strip().split(": ")[1]
            #  1187.21727 0.0 962.94357 0.0 1191.21681 588.05555 0.0 0.0 1.0
            intrinsics = np.fromstring(intrinsics, sep=' ', dtype=np.float32, count=9).reshape(3,3)
            

            pose_path = scene + "/ego_trajectory/"
            frame_path = scene + "/cam_img/4/data_rgb/"
            depth_path = scene + "/depth_pro/4/"

            imgs = sorted(frame_path.files('*.png')) #sorted sequence of images
            poses = sorted(pose_path.files('*.txt'))
            depths = sorted(depth_path.files('*.npy'))
            
            i = -1
            last_frame = None
            last_pose = None
            last_depth = None
            for image in imgs:
                i+=1 #idx for pose, intrinsics
                if last_frame is None:
                    last_frame = image
                    last_pose = np.genfromtxt(poses[i]).reshape((4, 4))
                    last_depth = depths[i]
                    continue

                sample = { # dict type
                    "img_path": [], # two str
                    "abs_pose": [], # abs_pose
                    "rel_pose": [],
                    "intrinsics": [intrinsics],
                    "depth_map_path": [],
                    "imu": [],
                }

                new_frame = image
                seq_frame = [last_frame, new_frame]
                sample["img_path"].append(seq_frame)

                # SiT has no depth pro or projected lidar
                new_depth = depths[i]
                sample["depth_map_path"].append([last_depth, new_depth])

                new_pose = np.genfromtxt(poses[i]).reshape((4, 4))
                pose_forward = np.linalg.inv(last_pose) @ new_pose #1->2
                pose_backward = np.linalg.inv(new_pose) @ last_pose #2->1
                sample["abs_pose"].append([last_pose, new_pose])
                sample["rel_pose"].append([pose_forward, pose_backward]) 
                sample["intrinsics"].append(intrinsics)
                
                sequence_set.append(sample)
                
                last_frame = new_frame
                last_pose = new_pose
                last_depth = new_depth
    
        self.samples = sequence_set
        
        
    def __getitem__(self, index: int):
        sample = self.samples[index]
        img1 = io.imread(sample["img_path"][0][0])         
        img2 = io.imread(sample["img_path"][0][1])  
        depth1 = np.load(sample["depth_map_path"][0][0]).astype(np.float64)
        depth2 = np.load(sample["depth_map_path"][0][1]).astype(np.float64)

        return img1, img2, sample["rel_pose"], sample["intrinsics"], depth1, depth2
    # return batch["image_1"], batch["image_2"], shape, batch["rel_pose"], batch["intrinsics"]


    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    DATASET_PATH = '/data/SiT'
    framework = SiT(DATASET_PATH, True)
    print('{} files to test'.format(len(framework))) # 1591 files to test

    # for sample in tqdm(framework):
    #     img = sample['img']
    #     gt = sample['relative_Trs'][0]
    #     print(gt)
    #     # print(vo.gt[0][3], vo.gt[1][3], vo.gt[2][3])
    #     true_x, true_y = int(10*gt[0][3])+290, int(10*gt[2][3])+900


    #     cv2.circle(traj, (true_x,true_y), 1, (0, 0, 255), 2) # GT in red
    #     # cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    #     # text = "Coordinates: x=%2fm y=%2fm z=%2fm"%(x,y,z)
    #     # cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

    #     cv2.imwrite('map.png', traj)

    #     # print(type(img))
    #     # print(img.shape)
    #     # io.imsave("./test_dataset_loader.png", img[0].squeeze().detach().cpu().numpy())