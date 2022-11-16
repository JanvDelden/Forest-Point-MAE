import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import abc

def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def build_lambda_sche(opti, config):
    if config.get('decay_step') is not None:
        lr_lbmd = lambda e: max(config.lr_decay ** (e / config.decay_step), config.lowest_decay)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opti, lr_lbmd)
    else:
        raise NotImplementedError()
    return scheduler

def build_lambda_bnsche(model, config):
    if config.get('decay_step') is not None:
        bnm_lmbd = lambda e: max(config.bn_momentum * config.bn_decay ** (e / config.decay_step), config.lowest_decay)
        bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd)
    else:
        raise NotImplementedError()
    return bnm_scheduler
    
def set_random_seed(seed, deterministic=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.

    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True

    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def is_seq_of(seq, expected_type, seq_type=None):
    """Check whether it is a sequence of some type.
    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type.
    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum
    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)




class Acc_Metric:
    def __init__(self, acc = 0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def generate_few_shot_data(shot, nruns, savepath="/user/jschnei2/data/few_shot/", eval=20):
    import pandas as pd

    np.random.seed(0)
    savedir = "../data/trees/seidel-trees2"
    trees = [np.load(os.path.join(savedir, path), allow_pickle=True) for path in os.listdir(savedir)]
    df = pd.DataFrame(np.array(trees), columns=["cls", "pc"])
    df = df[df.cls !=3]
    
    cls = df.cls.unique()
    train_idx = np.empty((nruns, len(cls), shot))
    eval_idx = np.empty((nruns, len(cls), eval))
    for s_id in range(nruns):
        for i, cl in enumerate(cls):
            subset = df[df.cls == cl]
            sample = subset.sample(n=shot + eval, replace=False)
            train_sample =  sample.index[0:shot]
            eval_sample = sample.index[shot:]
            train_idx[s_id, i] = train_sample
            eval_idx[s_id, i] = eval_sample

    np.save(os.path.join(savepath, "train_idx.npy"), train_idx.astype(int))
    np.save(os.path.join(savepath, "eval_idx.npy"), eval_idx.astype(int))
    np.savetxt(os.path.join(savepath, "train_idx.txt"), train_idx.astype(int).reshape(-1, nruns))
    np.savetxt(os.path.join(savepath, "eval_idx.txt"), eval_idx.astype(int).reshape(-1, nruns))
    return (train_idx, eval_idx)


def parse_few_shot_results(path, npeaks):
    results = np.loadtxt(path)
    # find peaks in 1d data
    peaks = []
    for i in range(len(results)):
        if i == (len(results) - 1):
            peaks.append(results[i])            
        elif results[i] >= results[i+1]:
            peaks.append(results[i])
    print(len(peaks))
    assert len(peaks) == npeaks
    return np.mean(peaks), np.std(peaks)