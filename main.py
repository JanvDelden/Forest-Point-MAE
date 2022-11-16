from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as classification
from tools import regression as regression
from tools import segmentation as segmentation
from tools import offset as offset
from utils import parser, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
import numpy as np
import wandb

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # config
    config = get_config(args, logger = logger)
    # batch size
    config.dataset.train.others.bs = config.total_bs
    config.dataset.val.others.bs = config.total_bs
    if config.dataset.get('test'):
        config.dataset.test.others.bs = config.total_bs 
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    start_wandb(args, config)
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic) # seed + rank, for augmentation

    if args.modelnet_way != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.modelnet_way
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.modelnet_way

    if args.task == "pretrain":
        pretrain(args, config)
    elif args.task == "cls":
        if args.fewshot:
            shot = args.shot
            nruns = args.nruns
            np.savetxt(os.path.join(args.experiment_path, "fewshot.txt"), np.empty(0))
            train_idx, eval_idx = misc.generate_few_shot_data(shot, nruns, savepath=args.experiment_path)
            config.dataset.train.others.few_shot_train_path =  os.path.join(args.experiment_path, "train_idx.npy") 
            config.dataset.val.others.few_shot_train_path = os.path.join(args.experiment_path, "train_idx.npy")
            config.dataset.test.others.few_shot_train_path = os.path.join(args.experiment_path, "train_idx.npy")
            config.dataset.train.others.few_shot_eval_path =  os.path.join(args.experiment_path, "eval_idx.npy")
            config.dataset.val.others.few_shot_eval_path = os.path.join(args.experiment_path, "eval_idx.npy")
            config.dataset.test.others.few_shot_eval_path = os.path.join(args.experiment_path,"eval_idx.npy")
            for i in range(nruns):
                config.dataset.train.others.few_shot =  i 
                config.dataset.val.others.few_shot = i 
                config.dataset.test.others.few_shot = i 
                classification(args, config)
            logger.info(misc.parse_few_shot_results(os.path.join(args.experiment_path, "fewshot.txt"), npeaks=nruns))
        else:
            classification(args, config)

    elif args.task == "regression":
        regression(args, config)
    elif args.task == "segmentation":
        segmentation(args, config)
    elif args.task == "offset":
        offset(args, config)
    else:
        raise NotImplementedError()

def start_wandb(args, config):
    wandb.init(project=args.task, entity="jans")
    wandb.config.update(args)
    wandb.config.update(config)
    wandb.run.name = args.exp_name + str(time.time())
    wandb.run.log_code("datasets")

if __name__ == '__main__':
    main()


