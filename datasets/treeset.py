import os
import torch
import numpy as np
import torch.utils.data as data
from .group_utils import Group, Mask
from utils.logger import *
import tqdm
from .build import DATASETS
from torchvision import transforms
from datasets import data_transforms

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(),
        data_transforms.PointcloudScaleAndTranslate(scale_low=2./3., scale_high=3./2., translate_range=0.2),
        data_transforms.RandomHorizontalFlip()
    ])

test_transforms = transforms.Compose([])

VAR = "treeset"
@DATASETS.register_module()
class Treeset(data.Dataset):
    def __init__(self, args, config):
        if config.subset == "train":
            self.plot_folders = config.plot_folders
            self.transforms = train_transforms
        else: 
            self.plot_folders = config.test_folders
            self.transforms = test_transforms

        self.npoints = config.npoints
        self.normalization = config.normalization
        self.normalization_pars = np.array(config.normalization_pars)
        self.in_memory = config.in_memory
        self.center, self.remove_ground = True, True

        # grouping config
        num_group, group_size = config.model.num_group, config.model.group_size
        mask_type, mask_ratio = config.model.transformer_config.mask_type, config.model.transformer_config.mask_ratio
        num_group, group_size, sampling_method = config.model.num_group, config.model.group_size, config.sampling_method
        self.grouper = Group(num_group, group_size, sampling_method)
        self.masker = Mask(mask_ratio, num_group, group_size, mask_type)

        self.samples_list = []
        for folder in self.plot_folders:
            files = os.listdir(folder)
            files = [os.path.join(folder, file) for file in files]
            self.samples_list += files
        paths = self.samples_list
        if self.in_memory: 
            self.samples_list = []
            for sample in tqdm.tqdm(paths):
                self.samples_list.append(np.load(sample).astype(np.float32))
        if config.validate_samples:
            for idx, sample in enumerate(self.samples_list):
                if len(sample) < self.npoints:
                    print(paths[idx], "has length", len(sample))

        print_log(f'[DATASET] sample out {self.npoints} points', logger = VAR)
        print_log(f'[DATASET] {len(self.samples_list)} samples from {len(self.plot_folders)} forest plots were loaded', logger = VAR)

    def normalize(self, pc):
        """ files were already centered, only correct by axis values and correct z axis to have data in unit cube"""
        pc = pc / self.normalization_pars
        pc[:, 2] = pc[:, 2] - 1
        return pc

    def random_sample(self, pc, num):
        if len(pc) < num:
            choice = np.random.choice(len(pc), num, replace=True)
        else:
            choice = np.random.choice(len(pc), num, replace=False)
        return  pc[choice]

    def __getitem__(self, idx):
        data = self.samples_list[idx]
        if not self.in_memory:
            data = np.load(data, allow_pickle=True).astype(np.float32)
        if self.remove_ground:
            data = data[data[:,3] != 9999]

        points = self.random_sample(data, self.npoints)[:, 0:3]
        if self.normalization:
            points = self.normalize(points)
        points = self.transforms(points)
        neighborhood, center, idx = self.grouper.group(points)
        mask = self.masker.mask(neighborhood, center)
        if self.center:
            neighborhood = neighborhood - center.reshape(-1, 1, 3)
        neighborhood = torch.from_numpy(neighborhood).float()
        center = torch.from_numpy(center).float()
        mask = torch.from_numpy(mask).bool()
        return neighborhood, center, mask


    def __len__(self):
        return len(self.samples_list)

