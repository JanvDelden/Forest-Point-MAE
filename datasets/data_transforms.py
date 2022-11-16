import numpy as np
import torch
import random


class PointcloudRotate(object):
    def __init__(self, alternate_rot_axis=False):
        self.alternate_rot_axis= alternate_rot_axis

    def __call__(self, pc):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cos = np.cos(rotation_angle)
        sin = np.sin(rotation_angle)
        R = np.array([  [cos, sin, 0], 
                        [-sin, cos, 0], 
                        [0, 0, 1]])
        if self.alternate_rot_axis:
            R = np.array([  [cos, 0, sin],
                            [0, 1, 0],
                            [-sin, 0, cos]])
        R = R.astype(np.float32)
        pc = np.matmul(pc, R)
        return pc


class PointcloudScaleAndTranslate(object):
    def __init__(self, scale_low=0.8, scale_high=1.2, translate_range=1, return_scalers=False):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.translate_range = translate_range

    def __call__(self, pc):
        scaler = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        translater = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
        pc[:, 0:3] = pc[:, 0:3] * scaler + translater
        return pc

class PointcloudJitter(object):
    def __init__(self, std=0.03, clip=0.08):
        self.std, self.clip = std, clip

    def __call__(self, pc):
        jittered_data = np.random.randn(*pc.shape) * self.std
        jittered_data[jittered_data > self.clip] = self.clip
        pc[:, 0:3] += jittered_data
            
        return pc


class PointcloudRandomInputDropout(object):
    def __init__(self, max_dropout_ratio=0.3):
        assert max_dropout_ratio >= 0 and max_dropout_ratio < 1
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, pc):
        dropout_ratio = np.random.random() * self.max_dropout_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx.tolist(), 0:3] = pc[:1, 0:3].repeat(len(drop_idx), 0)  # set to the first point
        return pc


class RandomHorizontalFlip(object):
  def __init__(self):
    pass
  def __call__(self, pc):
    if np.random.choice(a=[False, True]):
        pc[:, 0] = -pc[:,0]
    return pc
    

class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc):
        scaler = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        pc[:, 0:3] = pc[:, 0:3] * scaler 
        return pc


class PointcloudTranslate(object):
    def __init__(self, translate_range=0.2):
        self.translate_range = translate_range

    def __call__(self, pc):
        xyz2 = np.random.uniform(low=-self.translate_range, high=self.translate_range, size=[3])
        pc[:, 0:3] = pc[:, 0:3] + xyz2
        return pc


class PointCloudandTargetScale():
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, pc, target):
        bsize = pc.size()[0]
        for i in range(bsize):
            scaler = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[1])
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], torch.from_numpy(scaler).float().cuda())
            target[i] = target[i] * torch.from_numpy(scaler).float().cuda()
            
        return pc, target


class PatchDropout(object):
    # tokenwise transformation
    def __init__(self, max_ratio=0.2):
        assert max_ratio >= 0 and max_ratio < 1
        self.max_ratio = max_ratio

    def __call__(self, pc):

        group, center = pc
        dropout_ratio = np.random.random() * self.max_ratio  # 0~0.875
        drop_idx = np.where(np.random.random((group.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            group[drop_idx.tolist()] = group[:1].repeat(len(drop_idx), 0)  # set to the first token
            center[drop_idx.tolist()] = center[:1].repeat(len(drop_idx), 0)  # set to the first token
        return group, center


class ColorDropping(object):
    # all color transformation
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, pc):
        group, center = pc
        if np.random.binomial(n=1, p=self.p):
            group[:,:,3:6] = 0
        return group, center