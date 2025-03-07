import torch.utils.data as data
import numpy as np
from path import Path
import skimage.io as io
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

    def __init__(self, root, train=True):
        self.train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]
        self.test_seqs = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

        self.root = Path(root)
        if train:        
            self.scenes = [self.root/'{:04d}'.format(seq) for seq in self.train_seqs]
        else:        
            self.scenes = [self.root/'{:04d}'.format(seq) for seq in self.test_seqs]
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

                last_d = last_frame.dirname()/(last_frame.name[:-4] + '_depth_pro.npy')
                new_d = new_frame.dirname()/(new_frame.name[:-4] + '_depth_pro.npy')
                # print(last_d, new_d)
                # /data/MOTkitti/formatted/train/0000/000000_depth_pro.npy /data/MOTkitti/formatted/train/0000/000001_depth_pro.npy
                sample["depth_map_path"].append([last_d, new_d])

                last_pose = poses_4D[i-1]
                new_pose = poses_4D[i]
                pose_forward = np.linalg.inv(last_pose) @ new_pose #1->2
                pose_backward = np.linalg.inv(new_pose) @ last_pose #2->1
                sample["abs_pose"].append([last_pose, new_pose])
                sample["rel_pose"].append([pose_forward, pose_backward]) 
                sample["intrinsics"].append(intrinsics)
                sequence_set.append(sample)
                last_frame = new_frame
                
        self.samples = sequence_set
            
    def __getitem__(self, index: int):
        sample = self.samples[index]
        # print(sample["img_path"][0][0])
        # print(sample["img_path"][0][1])
        # print(sample["depth_map_path"][0][0])
        # print(sample["depth_map_path"][0][1])

        img1 = io.imread(sample["img_path"][0][0])         
        img2 = io.imread(sample["img_path"][0][1])  
        depth1 = np.load(sample["depth_map_path"][0][0]).astype(np.float64)
        depth2 = np.load(sample["depth_map_path"][0][1]).astype(np.float64)

        # print("img shape: ", img1.shape) # img shape:  (128, 416, 3)
        return img1, img2, sample["rel_pose"], sample["intrinsics"], depth1, depth2
    # return batch["image_1"], batch["image_2"], shape, batch["rel_pose"], batch["intrinsics"]


    def __len__(self):
        return len(self.samples)



# if __name__ == "__main__":
#     print("test dataset loader for inference")
#     formatted_odom_DATASET_PATH = "/data/sfm_formatted_kitti_raw/"


#     framework = ValidationSetWithPose(formatted_odom_DATASET_PATH)
#     print('{} files to test'.format(len(framework))) # 1591 files to test

#     # for sample in tqdm(framework):
#     #     a = sample["image_1"]
#     #     print(" ")
        