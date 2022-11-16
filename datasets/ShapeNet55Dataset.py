import os
import torch
import numpy as np
import torch.utils.data as data
from .build import DATASETS
from utils.logger import *
from .group_utils import Group, Mask
from datasets import data_transforms
from torchvision import transforms

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(alternate_rot_axis=True),
        #data_transforms.PointcloudJitter(std=0.001, clip=0.003),
        #data_transforms.PointcloudRandomInputDropout(max_dropout_ratio=0.3),
        data_transforms.PointcloudScaleAndTranslate(scale_low=2./3., scale_high=3./2., translate_range=0.2),
        data_transforms.RandomHorizontalFlip()
    ])

test_transforms = transforms.Compose([])



@DATASETS.register_module()
class ShapeNet(data.Dataset):
    def __init__(self, args, config):
        self.data_root = config.DATA_PATH
        self.pc_path = config.PC_PATH
        self.subset = config.subset
        self.npoints = config.npoints
        self.transforms = train_transforms if self.subset == "train" else test_transforms

        self.data_list_file = os.path.join(self.data_root, f'{self.subset}.txt')
        test_data_list_file = os.path.join(self.data_root, 'test.txt')
        
        self.sample_points_num = config.npoints
        self.whole = config.get('whole')
        self.center = True
        print_log(f'[DATASET] sample out {self.sample_points_num} points', logger = 'ShapeNet-55')
        print_log(f'[DATASET] Open file {self.data_list_file}', logger = 'ShapeNet-55')
        with open(self.data_list_file, 'r') as f:
            lines = f.readlines()
        if self.whole:
            with open(test_data_list_file, 'r') as f:
                test_lines = f.readlines()
            print_log(f'[DATASET] Open file {test_data_list_file}', logger = 'ShapeNet-55')
            lines = test_lines + lines
        self.file_list = []
        for line in lines:
            line = line.strip()
            taxonomy_id = line.split('-')[0]
            model_id = line.split('-')[1].split('.')[0]
            self.file_list.append({
                'taxonomy_id': taxonomy_id,
                'model_id': model_id,
                'file_path': line
            })
        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger = 'ShapeNet-55')
         # grouping config
        num_group, group_size = config.model.num_group, config.model.group_size
        mask_type, mask_ratio = config.model.transformer_config.mask_type, config.model.transformer_config.mask_ratio
        num_group, group_size, sampling_method = config.model.num_group, config.model.group_size, config.sampling_method
        self.grouper = Group(num_group, group_size, sampling_method)
        self.masker = Mask(mask_ratio, num_group, group_size, mask_type)

        self.permutation = np.arange(self.npoints)
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        np.random.shuffle(self.permutation)
        pc = pc[self.permutation[:num]]
        return pc
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(os.path.join(self.pc_path, sample['file_path'])).astype(np.float32)
        points = self.random_sample(data, self.sample_points_num)
        points = self.pc_norm(points)
        points = self.transforms(points)
        neighborhood, center, idx = self.grouper.group(points)
        mask = self.masker.mask(neighborhood, center)
        neighborhood = points[idx]
        if self.center:
            neighborhood = neighborhood - center.reshape(-1, 1, 3)
        neighborhood = torch.from_numpy(neighborhood).float()
        center = torch.from_numpy(center).float()
        mask = torch.from_numpy(mask).bool()
        return neighborhood, center, mask

    def __len__(self):
        return len(self.file_list)


import h5py
import numpy as np
# import open3d
import os

class IO:
    @classmethod
    def get(cls, file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension in ['.npy']:
            return cls._read_npy(file_path)
        # elif file_extension in ['.pcd']:
        #     return cls._read_pcd(file_path)
        elif file_extension in ['.h5']:
            return cls._read_h5(file_path)
        elif file_extension in ['.txt']:
            return cls._read_txt(file_path)
        else:
            raise Exception('Unsupported file extension: %s' % file_extension)

    # References: https://github.com/numpy/numpy/blob/master/numpy/lib/format.py
    @classmethod
    def _read_npy(cls, file_path):
        return np.load(file_path)
       
    # References: https://github.com/dimatura/pypcd/blob/master/pypcd/pypcd.py#L275
    # Support PCD files without compression ONLY!
    # @classmethod
    # def _read_pcd(cls, file_path):
    #     pc = open3d.io.read_point_cloud(file_path)
    #     ptcloud = np.array(pc.points)
    #     return ptcloud

    @classmethod
    def _read_txt(cls, file_path):
        return np.loadtxt(file_path)

    @classmethod
    def _read_h5(cls, file_path):
        f = h5py.File(file_path, 'r')
        return f['data'][()]