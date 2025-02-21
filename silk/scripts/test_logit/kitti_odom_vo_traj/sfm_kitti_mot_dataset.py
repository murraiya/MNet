import torch.utils.data as data
from path import Path
import skimage.io as io

from silk.datasets.raw_kitti_mot.raw_kitti_mot_utils import pose_from_oxts_packet, read_calib_file_MOT, transform_from_rot_trans
# from raw_mot_utils import pose_from_oxts_packet, read_calib_file_MOT, transform_from_rot_trans
import numpy as np
from path import Path
from imageio import imread
from tqdm import tqdm
from silk.datasets.pose_formatted_kitti_odom.util import load_images



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

    def __init__(self, root, seq):
        # self.train_seqs = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        self.test_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28]       

        self.root = Path(root)
        self.crawl_folders(seq)

    def crawl_folders(self, seq):
        calib_file = self.root / 'calib'/'{:04d}.txt'.format(seq)
        oxts_file = self.root/ 'oxts'/'{:04d}.txt'.format(seq)
        calib_data = read_calib_file_MOT(calib_file)
        
        self.depth_folder = self.root/'depth_pro'/'{:04d}'.format(seq)
        self.depth = sorted(self.depth_folder.files('*.npy'))
        self.image_2_folder = self.root/'image_02'/'{:04d}'.format(seq)
        self.imgs = sorted(self.image_2_folder.files('*.png'))

        imu2velo = np.vstack([calib_data["Tr_imu_velo"].reshape((3, 4)), [0, 0, 0, 1]])
        self.velo2cam = np.vstack([calib_data["Tr_velo_cam"].reshape((3, 4)), [0, 0, 0, 1]])
        # self.rect_R = calib_data['R_rect'].reshape(3,3)
        self.rect_mat = transform_from_rot_trans(calib_data['R_rect'], np.zeros(3))
        self.P2 = np.vstack([calib_data["P2"].reshape((3, 4)), [0, 0, 0, 1]], dtype=np.float32)
        # self.P_rect = P2 @ rect_mat
        
        # sample = self.load_image(0)
        
        
        self.scaled_P_rect = self.P2
        
        scale = None
        origin = None

        # imu2cam = P2 @ rect_mat @ self.velo2cam @ imu2velo
        imu2cam = self.rect_mat @ self.velo2cam @ imu2velo
        
        sequence_set = []
        with open(oxts_file, 'r') as f:
            i = -1
            last_frame = None
            for line in f.readlines():
                i+=1

                metadata = np.fromstring(line, dtype=np.float64, sep=' ')
                lat = metadata[0]
                if scale is None:
                    scale = np.cos(lat * np.pi / 180.)
                pose_matrix = pose_from_oxts_packet(metadata[:6], scale)
                
                if last_frame is None:
                    last_frame = self.imgs[i]
                
                    if origin is None:
                        origin = pose_matrix

                    last_odo_pose = imu2cam @ np.linalg.inv(origin) @ pose_matrix @ np.linalg.inv(imu2cam)
                    last_d = self.depth[i]
                    
                    continue

                sample = { # dict type
                    "scene": seq, #str
                    "img_path": [], # two str
                    "abs_pose": [], # abs_pose
                    "rel_pose": [],
                    "intrinsics": [self.scaled_P_rect[:3, :3]],
                    "depth_map_path": [],
                }
                
                new_odo_pose = imu2cam @ np.linalg.inv(origin) @ pose_matrix @ np.linalg.inv(imu2cam)
                sample['abs_pose'].append([last_odo_pose[:3], new_odo_pose[:3]])
                pose_forward = np.linalg.inv(last_odo_pose) @ new_odo_pose #1->2
                pose_backward = np.linalg.inv(new_odo_pose) @ last_odo_pose #2->1
                sample["rel_pose"].append([pose_forward, pose_backward]) 
                
                
                new_frame = self.imgs[i]
                seq_frame = [last_frame, new_frame]
                sample["img_path"].append(seq_frame)   
                
                new_d = self.depth[i]
                # /data/MOTkitti/formatted/train/0000/000000_depth_pro.npy /data/MOTkitti/formatted/train/0000/000001_depth_pro.npy
                sample["depth_map_path"].append([last_d, new_d])
                sequence_set.append(sample)
                
                #update
                last_d = new_d
                last_frame = new_frame
                last_odo_pose = new_odo_pose
                
                
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
                # 'sfmformatted': sample["sfmformatted"],
            }

       
    def __iter__(self):
        return self.generator()

    def __len__(self):
        return len(self.samples)










# if __name__ == "__main__":
#     print("test dataset loader for inference")
#     DATASET_PATH = "/data/MOTkitti/training"


#     framework = kittiMOTdataset(DATASET_PATH)
#     print('{} files to test'.format(len(framework))) # 1591 files to test

#     for sample in tqdm(framework):
#         im1, im2, pose, intrinsic, d1, d2 = sample
        
        