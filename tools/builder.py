import os, sys
# online package
import torch
# optimizer
import torch.optim as optim
# dataloader
from datasets import build_dataset_from_cfg
from models import build_model_from_cfg
# utils
from utils.logger import *
from utils.misc import *
from timm.scheduler import CosineLRScheduler
from torch.utils.data import ConcatDataset

def dataset_builder(args, config):
    config.others.padding = args.sampling_method == "variable_tokens"
    config.others.sampling_method = args.sampling_method
    dataset = build_dataset_from_cfg(args, config._base_, config.others)
    basekeys = config.keys()
    for basekey in basekeys:
        if "base" in basekey and basekey != "_base_":
            joinset = build_dataset_from_cfg(args, config[basekey], config.others)
            dataset = ConcatDataset((dataset, joinset))
            print("joined dataset", basekey)
    shuffle = config.others.subset == 'train'
    drop_last = config.others.subset == 'train' and not args.fewshot
    sampler = None
    collate_fn = pad_sequence if config.others.padding else None
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.others.bs,
                                            shuffle = shuffle, 
                                            drop_last = drop_last,
                                            num_workers = int(args.num_workers),
                                            worker_init_fn=worker_init_fn, collate_fn=collate_fn)
    return sampler, dataloader

def model_builder(args, config):
    model = build_model_from_cfg(args, config)
    return model

def build_opti_sche(base_model, config):
    opti_config = config.optimizer
    if opti_config.type == 'AdamW':
        def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
            decay = []
            no_decay = []
            for name, param in model.module.named_parameters():
                if not param.requires_grad:
                    continue  # frozen weights
                if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
                    # print(name)
                    no_decay.append(param)
                else:
                    decay.append(param)
            return [
                {'params': no_decay, 'weight_decay': 0.},
                {'params': decay, 'weight_decay': weight_decay}]
        param_groups = add_weight_decay(base_model, weight_decay=opti_config.kwargs.weight_decay)
        optimizer = optim.AdamW(param_groups, **opti_config.kwargs)
    elif opti_config.type == 'Adam':
        optimizer = optim.Adam(base_model.parameters(), **opti_config.kwargs)
    elif opti_config.type == 'SGD':
        optimizer = optim.SGD(base_model.parameters(), nesterov=False, **opti_config.kwargs)
    else:
        raise NotImplementedError()

    sche_config = config.scheduler
    if sche_config.type == 'LambdaLR':
        scheduler = build_lambda_sche(optimizer, sche_config.kwargs)  # misc.py
    elif sche_config.type == 'CosLR':
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                t_mul=1,
                lr_min=1e-6,
                decay_rate=0.1,
                warmup_lr_init=1e-6,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=1,
                t_in_epochs=True)
    elif sche_config.type == "RestartCosLR":
        scheduler = CosineLRScheduler(optimizer,
                t_initial=sche_config.kwargs.epochs,
                t_mul=1,
                lr_min=1e-6,
                decay_rate=0.7,
                warmup_lr_init=1e-6,
                warmup_t=sche_config.kwargs.initial_epochs,
                cycle_limit=3,
                t_in_epochs=True)
    elif sche_config.type == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **sche_config.kwargs)
    elif sche_config.type == 'function':
        scheduler = None
    elif sche_config.type == "Plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                patience=sche_config.patience,
                min_lr= 1e-6,
                factor=0.1)
    else:
        raise NotImplementedError()
    
    if config.get('bnmscheduler') is not None:
        bnsche_config = config.bnmscheduler
        if bnsche_config.type == 'Lambda':
            bnscheduler = build_lambda_bnsche(base_model, bnsche_config.kwargs)  # misc.py
        scheduler = [scheduler, bnscheduler]
    
    return optimizer, scheduler

def resume_model(base_model, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0
    print_log(f'[RESUME INFO] Loading model weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path)
    base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    base_model.load_state_dict(base_ckpt, strict = True)

    # parameter
    start_epoch = state_dict['epoch'] + 1
    best_metrics = state_dict['best_metrics']
    if not isinstance(best_metrics, dict):
        best_metrics = best_metrics.state_dict()
    # print(best_metrics)

    print_log(f'[RESUME INFO] resume ckpts @ {start_epoch - 1} epoch( best_metrics = {str(best_metrics):s})', logger = logger)
    return start_epoch, best_metrics

def resume_optimizer(optimizer, args, logger = None):
    ckpt_path = os.path.join(args.experiment_path, 'ckpt-last.pth')
    if not os.path.exists(ckpt_path):
        print_log(f'[RESUME INFO] no checkpoint file from path {ckpt_path}...', logger = logger)
        return 0, 0, 0
    print_log(f'[RESUME INFO] Loading optimizer from {ckpt_path}...', logger = logger )
    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # optimizer
    optimizer.load_state_dict(state_dict['optimizer'])

def save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, prefix, args, logger = None):
    torch.save({
                'base_model' : base_model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                'epoch' : epoch,
                'metrics' : metrics.state_dict() if metrics is not None else dict(),
                'best_metrics' : best_metrics.state_dict() if best_metrics is not None else dict(),
                }, os.path.join(args.experiment_path, prefix + '.pth'))
    print_log(f"Save checkpoint at {os.path.join(args.experiment_path, prefix + '.pth')}", logger = logger)

def load_model(base_model, ckpt_path, logger = None):
    if not os.path.exists(ckpt_path):
        raise NotImplementedError('no checkpoint file from path %s...' % ckpt_path)
    print_log(f'Loading weights from {ckpt_path}...', logger = logger )

    # load state dict
    state_dict = torch.load(ckpt_path, map_location='cpu')
    # parameter resume of base model
    if state_dict.get('model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['model'].items()}
    elif state_dict.get('base_model') is not None:
        base_ckpt = {k.replace("module.", ""): v for k, v in state_dict['base_model'].items()}
    else:
        raise RuntimeError('mismatch of ckpt weight')
    base_model.load_state_dict(base_ckpt, strict = True)

    epoch = -1
    if state_dict.get('epoch') is not None:
        epoch = state_dict['epoch']
    if state_dict.get('metrics') is not None:
        metrics = state_dict['metrics']
        if not isinstance(metrics, dict):
            metrics = metrics.state_dict()
    else:
        metrics = 'No Metrics'
    print_log(f'ckpts @ {epoch} epoch( performance = {str(metrics):s})', logger = logger)
    return 


def pad_sequence(batch_list):
    batch_list = torch.nn.utils.rnn.pad_sequence(batch_list, batch_first=True, padding_value=999)
    return batch_list


def prepare_process(args, config):
    logger = get_logger(args.log_name)
    # build dataset
    config.dataset.train.others.model = config.model
    config.dataset.val.others.model = config.model
    (train_sampler, train_dataloader), (_, test_dataloader),= dataset_builder(args, config.dataset.train), \
                                                            dataset_builder(args, config.dataset.val)
    #print_log(f'[TRAINSET] sample out {train_dataloader.dataset.npoints} points', logger=logger)
    print_log(f'[TRAINSET] {len(train_dataloader)} times bs samples were loaded', logger=logger)
    #print_log(f'[VALSET] sample out {test_dataloader.dataset.npoints} points', logger=logger)
    print_log(f'[VALSET] {len(test_dataloader)} times bs samples were loaded', logger=logger)
    # build model
    base_model = model_builder(args, config.model)
    # resume ckpts
    if args.resume:
        start_epoch, _ = resume_model(base_model, args, logger = logger)
    else:
        if args.ckpts is not None:
            yes = config.model.group_size != 32 
            base_model.load_model_from_ckpt(args.ckpts, args.freeze_encoder, logger=logger)
        else:
            print_log('Training from scratch', logger = logger)
        start_epoch = 0
    if args.use_gpu:    
        base_model.cuda()
    print_log('Using Data parallel ...' , logger = logger)
    base_model = nn.DataParallel(base_model).cuda()
    # optimizer & scheduler
    if hasattr(config, "max_step"):
        config.max_epoch = int(np.ceil(config.max_step / (len(train_dataloader))))
        config.scheduler.kwargs.epochs = config.max_epoch
        config.scheduler.kwargs.initial_epochs = int(np.round(config.max_epoch * config.scheduler.initial_epochs_ratio, 0))
    optimizer, scheduler = build_opti_sche(base_model, config)
    if args.resume:
        resume_optimizer(optimizer, args, logger = logger)
    base_model.zero_grad()

    return base_model, train_dataloader, test_dataloader, optimizer, scheduler, logger, config, start_epoch
