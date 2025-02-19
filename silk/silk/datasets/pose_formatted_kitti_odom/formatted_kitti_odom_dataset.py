# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch.utils.data as data
import skimage.io as io
from tqdm import tqdm
from path import Path
from silk.datasets.pose_formatted_kitti_odom.util import load_images
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
        return data.astype(np.float64)
    else:
        return data.astype(np.float64).reshape(reshape)

class formattedKittiOdom(data.Dataset):
    def __init__(self, root, train=True, sequence_length = 2):
        def collect_n_scenes():
            # operates only for sequence_length==2
            seq_frames=[]
            last_frame = None
            for frame in open(self.frame_list_path):
                if last_frame is None:
                    last_frame=[self.root + "/" + frame[:-8], frame[-7:-1]]
                    continue

                new_scene = self.root + "/" + frame[:-8]
                if(last_frame[0] == new_scene):
                    new_frame = [new_scene, frame[-7:-1]]
                    seq_frame=[
                        last_frame,
                        new_frame
                    ]
                    seq_frames.append(seq_frame)
                    last_frame = new_frame
                else:
                    # print("scene changing")
                    last_frame=[self.root + "/" + frame[:-8], frame[-7:-1]]
            return seq_frames

        self.samples = None
        self.root = Path(root)
        self.sequence_length = sequence_length
        self.frame_list_path = self.root/'train_tmp.txt' if train else self.root/'val_tmp.txt'        
        self.two_frames = collect_n_scenes()
        self.crawl_folders()

        
    def __getitem__(self, index: int):
        sample = self.samples[index]
        # print(sample["img_path"][0]) #  [[Path('/data/formatted_kitti_odom/00_02/000000.jpg'), Path('/data/formatted_kitti_odom/00_02/000000.jpg')]]
        img1 = io.imread(sample["img_path"][0][0])         
        img2 = io.imread(sample["img_path"][0][1])  

        # zoom_y = 164/img1.shape[0]
        # zoom_x = 164/img1.shape[1]
        
        # intrinsic = sample["intrinsics"][0]
        # print("handle intrinsic",intrinsic)
        # print("handle intrinsic",intrinsic[0])
        # print("handle intrinsic",intrinsic[1])
        
        # intrinsic[0] *= zoom_x
        # intrinsic[1] *= zoom_y
        # s = self.supervised
        # rescaled_intrinsic = intrinsic
        # print("img shape: ", img1.shape) #img shape:  (376, 1241, 3)
        return img1, img2, sample["rel_pose"], sample["intrinsics"][0]
    # return batch["image_1"], batch["image_2"], shape, batch["rel_pose"], batch["intrinsics"]


    def __len__(self):
        return len(self.samples)

    def crawl_folders(self):
        sequence_set = []
        Ks = {}
        poses = {}
        extrinsic = {}
        scenes = list(set([two_frame[0][0] for two_frame in self.two_frames]))
        scenes.sort()
        # sort() is needed
       
        for scene in tqdm(scenes): # scene: '/data/formatted_kitti_odom/00_02'
            Ks[scene] = (
                load_as_array(scene + "/cam.npy").astype(np.float32).reshape((3, 3))
            )
            poses_tmp= (
                load_as_array(scene + "/poses.npy")
                .astype(np.float32)
                .reshape(-1, 3, 4)
            )
            poses[scene] = np.zeros((poses_tmp.shape[0], 4, 4)).astype(np.float32)
            poses[scene][ : , :3] = poses_tmp
            poses[scene][ : , 3, 3] = 1
            #poses are 4D now
            extrinsic[scene] = (
                load_as_array(scene + "/Rt_cam2_gt.npy")
                .astype(np.float32)
                .reshape(-1, 4, 4)
            )
           
        for two_frame in tqdm(self.two_frames):
            scene, frame_1, frame_2 = two_frame[0][0], two_frame[0][1], two_frame[1][1]
            # print(scene, frame_1, frame_2) /data/formatted_kitti_odom/00_02 000000 000002
            frame_1_id, frame_2_id = int(frame_1), int(frame_2)   
            sample = { # dict type
                "scene": scene, #str
                "img_path": [], # two str
                "abs_pose": [], # abs_pose
                "rel_pose": [],
                "extrinsic": [],
                "intrinsics": [],
            }
            # img_file = scene + "/%s.jpg" % ("%06d" % frame_num)
            img_file_1, img_file_2 = scene + "/%s.jpg" % (frame_1), scene + "/%s.jpg" % (frame_2)
            sample["img_path"].append([img_file_1, img_file_2])
            sample["abs_pose"].append(poses[scene][frame_1_id]) # sample["abs_pose"] is not for training, maybe used for val? test?
            sample["abs_pose"].append(poses[scene][frame_2_id])

            # this is so right!!!! from frame_1->frame_2 transformation mat
            relative_scene_pose_forward = np.linalg.inv(poses[scene][frame_1_id]) @ poses[scene][frame_2_id]    
            # this is so right!!!! from frame_2->frame_1 transformation mat
            relative_scene_pose_backward = np.linalg.inv(poses[scene][frame_2_id]) @ poses[scene][frame_1_id]    
            
            relative_scene_pose_forward = ( #clement pinard did this when dumping poses but deepFEPE did not do it. so im following FEPE
                extrinsic[scene]
                @ relative_scene_pose_forward
                @ np.linalg.inv(extrinsic[scene])
            )
            relative_scene_pose_backward = ( #clement pinard did this when dumping poses but deepFEPE did not do it. so im following FEPE
                extrinsic[scene]
                @ relative_scene_pose_backward
                @ np.linalg.inv(extrinsic[scene])
            )
            sample["rel_pose"].append(
                relative_scene_pose_forward
            )  # [4, 4]
            sample["rel_pose"].append(
                relative_scene_pose_backward
            )  # [4, 4]

            sample["extrinsic"].append(
                extrinsic[scene]
            )  
            sample["intrinsics"].append(
                Ks[scene]
            )  

            sequence_set.append(sample)
        self.samples = sequence_set

# if __name__ == "__main__":
#     print("test dataset loader for inference")
#     DATASET_PATH = "/data/formatted_kitti_odom"
#     # OUTPUT_IMAGE_PATH = "./inference_data_loader_img.png"

#     framework = formattedKittiOdom(DATASET_PATH)
#     print('{} files to test'.format(len(framework))) # 1591 files to test

#     for sample in tqdm(framework):
#         print(sample[0])
#     #     path = sample['path']
#     #     print(path)
