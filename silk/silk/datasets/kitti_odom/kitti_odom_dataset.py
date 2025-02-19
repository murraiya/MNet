# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch
import os
from path import Path
import skimage.io as io
import torch.utils.data as data

class KittiOdom(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.img_files, self.poses = read_scene_data(self.root)

    def generator(self):
        for img_list, pose in zip(self.img_files, self.poses):
            imgs = io.imread(img_list)   
            # print(imgs.shape) #torch.Size([1, 1, 128, 416])

            yield {'img': imgs,
                    'path': img_list[0],
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
    
    train_seqs = [0, 1, 2, 3, 4, 5, 6, 7, 8]                            
    img_dir = Path(data_root+'/sequences/'+sequence+'/image_2/')
    
    imgs = sum([list(img_dir.walkfiles('*.{}'.format("png")))], [])
    # print('{} files to test'.format(len(test_files))) # 1591 files to test
    
    poses = np.genfromtxt(data_root/'poses'/'{}.txt'.format(sequence)).astype(np.float64).reshape(-1, 3, 4)
    # print(poses.shape) (1591, 3, 4)


    return im_sequences, poses_sequences