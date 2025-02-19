import torch.utils.data as data
import numpy as np
import skimage.io as io

from path import Path
from silk.cv.homography import resize_homography
from silk.models.superpoint_utils import load_image
from typing import Any, Iterable, Tuple, Union



class KittiRaw(data.Dataset):

    def __init__(self, kitti_raw_path, train=True, transform=None, target_transform=None):
        self.root = Path(kitti_raw_path)
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            img = sorted(scene.files('*.jpg'))
            for i in range(len(img)):
                sample = {'img': img[i]}
                sequence_set.append(sample)
        self.samples = sequence_set

    # author says numpy array with dtype=uint8, shape(H,W,3) is needed for image augmentation pipeline
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        sample = self.samples[index]
        img = io.imread(sample['img'])               
        # print("dataloader: img ", type(img), type(img[0][0][0]), img.shape) # 128x416x3
        
        io.imsave("./from_dataloader.png", img)
        return img, None #make it tuple

    def __len__(self):
        return len(self.samples)
    