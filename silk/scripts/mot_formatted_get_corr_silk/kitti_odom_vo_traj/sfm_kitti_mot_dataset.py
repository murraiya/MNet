import torch.utils.data as data
import numpy as np
from path import Path
import skimage.io as io
from silk.datasets.pose_formatted_kitti_odom.util import load_images

from PIL import Image
import tqdm


class kittiMOTdataset(data.Dataset):
    """A sequence validation data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000000.npy
        root/scene_1/0000001.jpg
        root/scene_1/0000001.npy
        ..
        root/scene_1/cam.txt
        root/scene_1/pose.txt
        root/scene_2/0000000.jpg
        root/scene_2/0000000.npy
        .
    """

    def __init__(self, root, seq, train=False):
        # self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        # self.test_seqs = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.root = Path(root)
        self.scenes = [self.root/'{:04d}'.format(seq)]
        self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []
        # print(self.scenes) #just folder names
        for scene in self.scenes:
            poses = np.genfromtxt(scene/'poses.txt').reshape((-1, 3, 4))
            poses_4D = np.zeros((poses.shape[0], 4, 4)).astype(np.float32)
            poses_4D[:, :3] = poses
            poses_4D[:, 3, 3] = 1
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            imgs = sorted(scene.files('*.jpg')) #sorted sequence of images
            assert(len(imgs) == poses.shape[0])
            # print(imgs)

            i = -1
            last_frame = None
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
                    "sfmformatted": [],
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
       

    def generator(self):
        for sample in self.samples:
            img = load_images(sample["img_path"][0][0],sample["img_path"][0][1])      
            # img2 = load_images(sample["img_path"][0][1])
            # print(sample['depth_map_path'])
             
            yield {
                'images': img,
                # 'images_2': img2,
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

