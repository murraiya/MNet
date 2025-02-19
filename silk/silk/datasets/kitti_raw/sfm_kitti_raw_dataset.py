import torch
import torch.utils.data as data
import numpy as np
# from imageio import imread
import skimage.io as io

from path import Path
from silk.cv.homography import resize_homography
from silk.models.superpoint_utils import load_image
from typing import Any, Iterable, Tuple, Union



class KittiRaw(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """

    def __init__(self, kitti_raw_path, train=True, transform=None, target_transform=None):
        # print("33333333")
        self.root = Path(kitti_raw_path)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.crawl_folders()
        # print("crawl folders done")

    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            img = sorted(scene.files('*.jpg'))
            for i in range(len(img)):
                sample = {'intrinsics': intrinsics, 'img': img[i]}
                sample = {'img': img[i]}
                sequence_set.append(sample)
        self.samples = sequence_set


    # author says numpy array with dtype=uint8, shape(H,W,3) is needed for image augmentation pipeline
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        # print("in getitem")
        sample = self.samples[index]
        img = io.imread(sample['img'])       
        # intrinsics = sample['intrinsics']   
        
        # print("dataloader: img ", type(img), type(img[0][0][0]), img.shape) # 128x416x3

        return img, None #make it tuple

    def __len__(self):
        return len(self.samples)
