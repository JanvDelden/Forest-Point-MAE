# from tools import run_net
from tools import test_net, test_offset, test_seg, test_cls
from utils import parser, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch

def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # config
    config = get_config(args, logger = logger)
    # batch size
    config.dataset.train.others.bs = config.total_bs
    config.dataset.val.others.bs = 1
    config.dataset.test.others.bs = 1
    # log 
    log_args_to_file(args, 'args', logger = logger)
    log_config_to_file(config, 'config', logger = logger)
    # exit()
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        misc.set_random_seed(args.seed, deterministic=args.deterministic) # seed + rank, for augmentation
    # run
    if args.task == "pretrain":
        test_net(args, config)
    elif args.task == "offset":
        test_offset(args, config)
    elif args.task == "segmentation":
        test_seg(args, config)
    elif args.task == "cls":
        test_cls(args, config)
    else:
        raise NotImplementedError


if __name__ == '__main__':
    main()
