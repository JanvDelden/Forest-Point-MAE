from re import S
import numpy as np
import os, sys, h5py
from torch.utils.data import Dataset
import torch
from .build import DATASETS
from utils.logger import *
from torchvision import transforms
from datasets import data_transforms
from .group_utils import Group, Mask

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(alternate_rot_axis=True),
        #data_transforms.PointcloudJitter(std=0.001, clip=0.003),
        #data_transforms.PointcloudRandomInputDropout(max_dropout_ratio=0.3),
        data_transforms.PointcloudScaleAndTranslate(scale_low=2./3., scale_high=3./2., translate_range=0.2),
        data_transforms.RandomHorizontalFlip()
    ])
token_transforms = transforms.Compose(
    [
        data_transforms.PatchDropout(max_ratio=0.9),
    ])

test_transforms = transforms.Compose([])

@DATASETS.register_module()
class ScanObjectNN_all(Dataset):
    def __init__(self, args, config):
        super().__init__()
        self.subset = config.subset
        self.task = args.task 
        self.use_color = False
        num_group, group_size, sampling_method = config.model.num_group, config.model.group_size, config.sampling_method
        if self.task == "pretrain":
            mask_type, mask_ratio = config.model.transformer_config.mask_type, config.model.transformer_config.mask_ratio
            self.masker = Mask(mask_ratio, num_group, group_size, mask_type)
        self.grouper = Group(num_group, group_size, sampling_method)

        self.npoints = config.npoints
        self.center = True

        if self.subset == "train":
            self.plot_folders = config.plot_folders
            self.transforms = train_transforms
            if self.task == "cls":
                self.token_transforms =transforms.Compose([data_transforms.PatchDropout(max_ratio=args.patch_dropout)])
            else:
                self.token_transforms =  test_transforms 
        else: 
            self.plot_folders = config.test_folders
            self.transforms = test_transforms
            self.token_transforms = test_transforms

        self.samples_list = []
        for folder in self.plot_folders:
            files = os.listdir(folder)
            files = [os.path.join(folder, file) for file in files]
            self.samples_list += files
        self.samples_list = np.array(self.samples_list)
        print("ScanobjectNN", self.subset, len(self), "samples")

    def __getitem__(self, idx):
        points, label = np.load(self.samples_list[idx], allow_pickle=True)
        if self.use_color:
            points = np.hstack((points[:, :3], points[:,6:9] / 256))
        else:
            points = points[:,:3]
        points = self.random_sample(points, self.npoints)
        #if self.task == "pretrain":
        #    points[:,:3], _ = self.normalize(points[:,:3])
        points[:, :3] = self.transforms(points[:, :3])
        neighborhood, center, idx = self.grouper.group(points[:, :3])
        if self.task == "pretrain":
            mask = self.masker.mask(neighborhood, center)
        if self.center:
            neighborhood = points[idx]
            neighborhood[:,:,:3] = neighborhood[:,:,:3] - center.reshape(-1, 1, 3)
        neighborhood, center = self.token_transforms((neighborhood, center))
        neighborhood = torch.from_numpy(neighborhood).float()
        center = torch.from_numpy(center).float()
        if self.task == "pretrain":
            mask = torch.from_numpy(mask).bool()
            return neighborhood, center, mask
        elif self.task == "cls":
            label = torch.from_numpy(np.array(label)).int()
            return neighborhood, center, label

    def normalize(self, pc, mean=None):
        if mean is None:
            mean = pc.mean(axis=0)
        pc = pc - mean
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc, mean

    def __len__(self):
        return len(self.samples_list)

    def random_sample(self, pc, num):
        if len(pc) < num:
            choice = np.random.choice(len(pc), num, replace=True)
        else:
            choice = np.random.choice(len(pc), num, replace=False)
        return pc[choice]



######################################### presampled Data
pretrain_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(alternate_rot_axis=True),
        #data_transforms.PointcloudJitter(std=0.001, clip=0.005),
        #data_transforms.PointcloudRandomInputDropout(max_dropout_ratio=0.3),
        data_transforms.PointcloudScaleAndTranslate(scale_low=2./3., scale_high=3./2., translate_range=0.2),
        data_transforms.RandomHorizontalFlip()
    ])
@DATASETS.register_module()
class ScanObjectNN_presampled(Dataset):
    def __init__(self, args, config):
        super().__init__()
        self.subset = config.subset
        self.root = config.ROOT
        self.task = args.task 
        num_group, group_size, sampling_method = config.model.num_group, config.model.group_size, config.sampling_method
        if self.task == "pretrain":
            mask_type, mask_ratio = config.model.transformer_config.mask_type, config.model.transformer_config.mask_ratio
            self.masker = Mask(mask_ratio, num_group, group_size, mask_type)

        self.grouper = Group(num_group, group_size, sampling_method)
        self.npoints = config.npoints
        self.center = True

        if self.subset == 'train':
            data_file = config.train_file
            self.transforms = train_transforms
            if self.task == "cls":
                self.token_transforms =transforms.Compose([data_transforms.PatchDropout(max_ratio=args.patch_dropout)])
            else:
                self.token_transforms = test_transforms
        elif self.subset == 'test':
            data_file = config.test_file
            self.transforms = test_transforms
            self.token_transforms = test_transforms
        else:
            raise NotImplementedError()
        h5 = h5py.File(os.path.join(self.root, data_file), 'r')
        self.points = np.array(h5['data']).astype(np.float32)
        self.labels = np.array(h5['label']).astype(int)
        h5.close()

        print(f'Successfully load ScanObjectNN shape of {self.points.shape} from {data_file}')

    def normalize(self, pc, mean=None):
        if mean is None:
            mean = pc.mean(axis=0)
        pc = pc - mean
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc, mean

    def random_sample(self, pc, num):
        if len(pc) < num:
            choice = np.random.choice(len(pc), num, replace=True)
        else:
            choice = np.random.choice(len(pc), num, replace=False)
        return pc[choice]

    def __getitem__(self, idx):
        points = self.points[idx].copy()
        label = self.labels[idx].copy()
        if self.npoints != 2048:
            points = self.random_sample(points, self.npoints)
        #if self.task == "pretrain":
        #    points, _ = self.normalize(points)
        points = self.transforms(points)
        neighborhood, center, idx = self.grouper.group(points)
        if self.task == "pretrain":
            mask = self.masker.mask(neighborhood, center)

        if self.center:
            neighborhood = neighborhood - center.reshape(-1, 1, 3)
        neighborhood, center = self.token_transforms((neighborhood, center))
        neighborhood = torch.from_numpy(neighborhood).float()
        center = torch.from_numpy(center).float()
        if self.task == "pretrain":
            mask = torch.from_numpy(mask).bool()
            return neighborhood, center, mask
        elif self.task == "cls":
            label = torch.from_numpy(np.array(label)).int()
            return neighborhood, center, label
    
    def __len__(self):
        return self.points.shape[0]
