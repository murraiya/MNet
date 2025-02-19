# Mostly based on the code written by Clement Godard:
# https://github.com/mrharicot/monodepth/blob/master/utils/evaluation_utils.py
import numpy as np
import torch
import os
from path import Path
from imageio import imread
from tqdm import tqdm
from util import convert_points_from_homogeneous, get_model, load_images, SILK_MATCHER

# for inference
class test_framework_sfm_KITTI(object):
    def __init__(self, root):
        # print("33333333")
        self.root = Path(root)
        scene_list_path = self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)]
        self.crawl_folders()

    def crawl_folders(self):
        sequence_set = []
        for scene in self.scenes:
            poses = np.genfromtxt(scene/'poses.txt') #.reshape((-1, 3, 4))
            intrinsic = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3))
            print(poses.shape)
            imgs = sorted(scene.files('*.jpg'))
            assert(len(imgs) == poses.shape[0])
            for i in range(len(imgs)):
                img = imgs[i]
                sample = {'intrinsic': intrinsic, 'img': img, 'pose': []}
                sample['pose'] = poses[i].reshape((3,4))
                # print(type(sample))
                # sample['poses'] = np.stack(sample['poses'])
                sequence_set.append(sample)
        self.samples = sequence_set
        # print(len(sequence_set), type(self.samples))


    def generator(self):
        for i in range(len(self.samples)):
            sample = self.samples[i]
            img = load_images(sample['img'])

            yield {'img': img,
                   'intrinsic': sample['intrinsic'],
                   'pose': sample['pose']
                   }

    def __iter__(self):
        return self.generator()

    def __len__(self):
        return len(self.samples)


