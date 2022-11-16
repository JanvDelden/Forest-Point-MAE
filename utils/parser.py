import os
import argparse
from pathlib import Path

def get_args(argv = None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',         type = str,         help = 'yaml config file')
    parser.add_argument('--num_workers', type=int, default=8)
    # seed 
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic',        action='store_true',        help='whether to set deterministic options for CUDNN backend.')      
    # some args
    parser.add_argument('--exp_name', type = str, default='default', help = 'experiment name')
    parser.add_argument('--ckpts', type = str, default=None, help = 'test used ckpt path')
    parser.add_argument('--patch_dropout', type = float, default=0, help = "none")
    
    # task
    parser.add_argument('--test',         action='store_true', default=False, help = 'test mode for certain ckpt')
    parser.add_argument('--freeze_encoder',        action='store_true',        default=False,        help = 'freeze transformer encoder')
    parser.add_argument('--sampling_method', choices=['fps', 'rand', "slice_fps", "kmeans", "kmeans_jitter", "fpskmeans"], default="kmeans",         help = 'find centers method')
    parser.add_argument('--task', choices=['pretrain', 'cls', "segmentation", "regression", "offset"], default="pretrain", help = 'choose task')   

    # few shot cls
    parser.add_argument('--fewshot',        action='store_true',        default=False,        help = 'whether to do fewshot classification')
    parser.add_argument('--shot',   type = int,        default=0,        help = 'how many samples for fewshot classification')
    parser.add_argument('--nruns',   type = int,        default=0,        help = 'how many indepenent experiments for fewshot classification')
    parser.add_argument('--modelnet_shot',   type = int,        default=-1,        help = 'how many indepenent experiments for fewshot classification')
    parser.add_argument('--modelnet_fold',   type = int,        default=-1,        help = 'how many indepenent experiments for fewshot classification')
    parser.add_argument('--modelnet_way',   type = int,        default=-1,        help = 'how many indepenent experiments for fewshot classification')

    # rm probably
    parser.add_argument('--resume',         action='store_true',         default=False,         help = 'autoresume training (interrupted by accident)')

    if argv is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(argv)
    if args.test and args.resume:
        raise ValueError(            '--test and --resume cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(         'ckpts shouldnt be None while test mode')

    args.experiment_path = os.path.join('./experiments', Path(args.config).parts[1], Path(args.config).stem, args.exp_name)
    args.log_name = Path(args.config).stem
    if args.exp_name != "None":
        create_experiment_dir(args)
    return args

def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
