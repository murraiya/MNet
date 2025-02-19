import torch.utils.data as data
import numpy as np
from path import Path
import skimage.io as io
from silk.datasets.pose_formatted_kitti_odom.util import load_images
from PIL import Image


class ValidationSetWithPose(object):
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

    def __init__(self, root, train=False, sequence_length = 2):
        self.root = Path(root)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'        
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        # self.two_frames = collect_n_scenes()
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
            # print(imgs) #ordered
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000030.jpg
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000031.jpg
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000032.jpg
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000033.jpg
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000034.jpg
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000035.jpg
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000036.jpg
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000037.jpg
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000038.jpg
            # /data/sfm_formatted_kitti_raw/2011_09_28_drive_0037_sync_02/0000000039.jpg
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
                    "depth_map_path": [],
                }

                new_frame = image
                seq_frame = [last_frame, new_frame]
                sample["img_path"].append(seq_frame)   

                d = last_frame.dirname()/(last_frame.name[:-4] + '.npy')
                assert(d.isfile()), "depth file {} not found".format(str(d))
                sample["depth_map_path"].append(d)

                last_pose = poses_4D[i-1]
                new_pose = poses_4D[i]
                pose_forward = np.linalg.inv(last_pose) @ new_pose #T1->2 = T1.inverse() @ T2
                pose_backward = np.linalg.inv(new_pose) @ last_pose #2->1
                sample["abs_pose"].append([last_pose, new_pose])
                sample["rel_pose"].append([pose_forward, pose_backward]) 
                sample["intrinsics"].append(intrinsics)
                sequence_set.append(sample)
                last_frame = new_frame
    
        self.samples = sequence_set

    
    def generator(self):
        for sample in self.samples:
            img1 = load_images(sample["img_path"][0][0])      
            img2 = load_images(sample["img_path"][0][1])
            depth = np.load(sample["depth_map_path"][0]).astype(np.float64)
             
            yield {
                'images_1': img1,
                'images_2': img2,
                'path': sample["img_path"][0],
                'rel_pose': sample["rel_pose"][0],
                'abs_pose': sample["abs_pose"][0], 
                'intrinsic': sample["intrinsics"][0],
                'depth': depth
            }
        
            
    def __iter__(self):
        return self.generator()
    
      
    # def __getitem__(self, index: int):
    #     sample = self.samples[index]
    #     # print(sample["img_path"][0][0])
    #     # print(sample["img_path"][0][1])
    #     # print(sample["depth_map_path"][0])

    #     img1 = io.imread(sample["img_path"][0][0])         
    #     img2 = io.imread(sample["img_path"][0][1])  
    #     depth = np.load(sample["depth_map_path"][0]).astype(np.float64)

    #     # print("img shape: ", img1.shape) # img shape:  (128, 416, 3)
    #     return img1, img2, sample["rel_pose"], sample["intrinsics"], depth


    def __len__(self):
        return len(self.samples)
