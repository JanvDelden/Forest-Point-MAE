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


biomass_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(),
        #data_transforms.PointcloudJitter(std=0.01, clip=0.05),
        data_transforms.RandomHorizontalFlip(),
        data_transforms.PointcloudTranslate(translate_range=0.2)
    ])

train_transforms = transforms.Compose(
    [
        data_transforms.PointcloudRotate(),
        data_transforms.PointcloudScaleAndTranslate(scale_low=0.9, scale_high=1.1, translate_range=0.1),
        data_transforms.RandomHorizontalFlip(),
    ])

token_transforms = transforms.Compose(
    [
        data_transforms.PatchDropout(max_ratio=0.95)
    ])

test_transforms = transforms.Compose([])

VAR = "transfer_treeset"
@DATASETS.register_module()
class Cls_Treeset(data.Dataset):
    def __init__(self, args, config):
        self.plot_folders = config.plot_folders
        self.npoints = config.npoints
        self.normalization = config.normalization
        self.normalization_pars = np.array(config.normalization_pars)
        self.fewshot = config.few_shot if hasattr(config, "few_shot") else None
        self.target_label = config.target_type == "label"
        self.biomass = config.target_type == "biomass"
        assert config.target_type == "biomass" or config.target_type == "label"

        self.samples_list = []
        for folder in self.plot_folders:
            files = os.listdir(folder)
            files = [os.path.join(folder, file) for file in files]
            self.samples_list += files
        self.samples_list = np.array(self.samples_list)
        # grouping config
        num_group, group_size, sampling_method = config.model.num_group, config.model.group_size, config.sampling_method
        self.grouper = Group(num_group, group_size, sampling_method)

        if self.fewshot is None:
            rng = default_rng(seed=config.seed)
            self.samples_list = rng.permutation(self.samples_list)
            splitidx = np.floor(config.train_ratio * len(self.samples_list)).astype(int)
            print_log("SHUFFLED THE DATA", logger = VAR+ config.subset)
        
            if config.subset == "train":
                self.samples_list = self.samples_list[0:splitidx]
            else:
                self.samples_list = self.samples_list[splitidx:]
        else: # if self.fewshot is smth
            if config.subset == "train":
                idx = np.load(config.few_shot_train_path)[self.fewshot]
            else:
                idx = np.load(config.few_shot_eval_path)[self.fewshot]
            self.samples_list = self.samples_list[idx.reshape(-1)]
            print_log(f'[DATASET]use fewshot {self.fewshot}', logger = VAR + config.subset)
        self.samples_path = self.samples_list.copy()
        self.samples_list = [np.load(sample, allow_pickle=True) for sample in self.samples_list]
        if config.validate_samples:
            for idx, (mass, sample) in enumerate(self.samples_list):
                if len(sample) < self.npoints:
                    print(idx, "has length", len(sample))
                    
        self.token_transforms =transforms.Compose([data_transforms.PatchDropout(max_ratio=args.patch_dropout)])

        if self.target_label:
            self.transforms = train_transforms

        elif self.biomass:
            print("use biomass transforms")
            self.transforms = biomass_transforms
        else:
            raise NotImplementedError
        if config.subset != "train":
            self.transforms, self.token_transforms = test_transforms, test_transforms
        self.center = True
        
    def normalize(self, pc):
        pc = pc - pc.mean(axis=0)
        pc = pc / self.normalization_pars
        return pc

    def random_sample(self, pc, num):
        if len(pc) < num:
            choice = np.random.choice(len(pc), num, replace=True)
        else:
            choice = np.random.choice(len(pc), num, replace=False)
        return pc[choice]

    def __getitem__(self, idx):
        target, points = self.samples_list[idx]
        points = self.random_sample(points, self.npoints)
        if self.normalization:
            points = self.normalize(points)
        points = self.transforms(points)
        neighborhood, center, idx = self.grouper.group(points)
        if self.center:
            neighborhood = neighborhood - center.reshape(-1, 1, 3)
        neighborhood, center = self.token_transforms((neighborhood, center))
        #neighborhood = np.zeros(neighborhood.shape)
        neighborhood = torch.from_numpy(neighborhood).float()
        center = torch.from_numpy(center).float()
        if self.target_label:
            target = torch.from_numpy(np.array(target)).int()
        elif self.biomass: 
            target = torch.from_numpy(np.array(target)).float()
        return neighborhood, center, target

    def __len__(self):
        return len(self.samples_list)

if __name__ == '__main__':
    pass
