import os
import torch
import numpy as np
import torch.utils.data as data
from utils.logger import *
from .build import DATASETS
from numpy.random import default_rng
from .group_utils import Group
from torchvision import transforms
from datasets import data_transforms


train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(),
        #data_transforms.PointcloudJitter(std=0.01, clip=0.03),
        data_transforms.PointcloudScaleAndTranslate(scale_low=2./3., scale_high=3./2., translate_range=0.2),
        data_transforms.RandomHorizontalFlip()
    ])
    
test_transforms = transforms.Compose([])

VAR = "transfer_treeset"
@DATASETS.register_module()
class Transfer_Treeset(data.Dataset):
    def __init__(self, args, config):
        self.offset = config.target_type == "offset"
        if config.subset == "train":
            self.plot_folders = config.plot_folders
            self.transforms = train_transforms
            self.token_transforms =transforms.Compose([data_transforms.PatchDropout(max_ratio=args.patch_dropout)])
        else: 
            self.plot_folders = config.test_folders
            self.transforms,self.token_transforms = test_transforms, test_transforms

        self.npoints = config.npoints
        self.normalization = config.normalization
        self.normalization_pars = np.array(config.normalization_pars)
        self.remove_ground = True

        self.samples_list = []
        for folder in self.plot_folders:
            files = os.listdir(folder)
            files = [os.path.join(folder, file) for file in files]
            self.samples_list += files
        self.samples_list = np.array(self.samples_list)
        if hasattr(config, "max_num_samples"):
            if len(self.samples_list) > config.max_num_samples:
                print("shortened samples list")
                rng = default_rng(seed=config.seed)
                self.samples_list = rng.permutation(self.samples_list)
                self.samples_list = self.samples_list[:config.max_num_samples]
        # grouping config
        num_group, group_size, sampling_method = config.model.num_group, config.model.group_size, config.sampling_method
        self.grouper = Group(num_group, group_size, sampling_method)
        self.samples_path = self.samples_list.copy()
        self.rng = default_rng(seed=args.seed)
        self.center = True
        #self.samples_list = [np.load(sample, allow_pickle=True) for sample in self.samples_list]
        #if self.remove_ground:
        #    self.samples_list =  [sample[sample[:,3] != 9999] for sample in self.samples_list]
    def normalize(self, pc, mean=None):
        if mean is None:
            mean = pc.mean(axis=0)
        pc = pc - mean
        pc = pc / self.normalization_pars
        return pc, mean

    def random_sample(self, pc, num):
        if len(pc) < num:
            choice = self.rng.choice(len(pc), num, replace=True)
        else:
            choice = self.rng.choice(len(pc), num, replace=False)
        return pc[choice]

    def __getitem__(self, idx):
        points = np.load(self.samples_list[idx], allow_pickle=True)
        if self.remove_ground:
            points = points[points[:,3] != 9999]
        points = self.random_sample(points, self.npoints)
        points, label = points[:,:3], points[:,3]  
        filterall = filter_to_inner_quadrant(points)
        if self.normalization:
            points, _ = self.normalize(points)
        points = self.transforms(points)
        neighborhood, center, idx = self.grouper.group(points)
        neighborhood, center = self.token_transforms((neighborhood, center))
        if self.center:
            neighborhood = neighborhood - center.reshape(-1, 1, 3)
        #neighborhood = np.zeros(neighborhood.shape)
        neighborhood = torch.from_numpy(neighborhood).float()
        center = torch.from_numpy(center).float()
        label = torch.from_numpy(np.array(label)).int()
        filterall = torch.from_numpy(filterall).bool()
        target = torch.from_numpy(get_offset(points, label)).float()
        return neighborhood, center, target, torch.from_numpy(np.array(points)).float(), idx, label, filterall

    def __len__(self):
        return len(self.samples_list)


if __name__ == '__main__':
    pass


def get_offset(points, label):
    offset = np.empty((len(points), 3))
    instances = np.unique(label)
    for i_ in instances: 
        instance_points_idx = np.where(label == i_)
        instance_points = points[instance_points_idx]
        if i_ == 9999:
            instance_mean = np.array([0, 0, -1])
        else:
            min_z = np.min(instance_points[:, 2])
            z_thresh = min_z + 0.30 / 25
            lowest_instance_points = instance_points[instance_points[:,2] < z_thresh]
            instance_mean = np.mean(lowest_instance_points, axis=0)
            instance_mean[2] = -1
        offset[instance_points_idx] = instance_points - instance_mean
    return offset


def filter_to_inner_quadrant(points, val=4):
    filterx = np.logical_and(points[:,0] < val, points[:,0] > -val)
    filtery = np.logical_and(points[:,1] < val, points[:,1] > -val)
    filterall = np.logical_and(filterx, filtery)
    return filterall