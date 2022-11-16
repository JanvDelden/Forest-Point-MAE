import torch
import torch.nn as nn
from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import os
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import wandb


scaler = GradScaler()
def run_net(args, config):
    wandb.init(project="offset", entity="jans")
    base_model, train_dataloader, test_dataloader, optimizer, scheduler, logger, config, start_epoch\
         = builder.prepare_process(args, config)
    # parameter setting
    eval_value = []
    for epoch in range(start_epoch, config.max_epoch + 1):

        epoch_start_time = time.time()
        argmax = train(base_model, train_dataloader, epoch, args, optimizer, config, logger=logger) 
        eval_val = validate(base_model, test_dataloader, epoch, args, config, argmax, logger=logger)
        eval_value.append(eval_val)
        schedulerval = eval_val if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else epoch + 1
        epoch_end_time = time.time()
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) lr = %.6f' % (epoch, epoch_end_time - epoch_start_time, optimizer.param_groups[0]['lr']), logger = logger) 
        scheduler.step(schedulerval)
        if epoch % 25 == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-last', args, logger = logger)      
        if eval_val <= torch.min(torch.tensor(eval_value)):
            builder.save_checkpoint(base_model, optimizer, epoch, None, None, f'ckpt-best', args, logger=logger)
            print_log("--------------------------------------------------------------------------------------------", logger=logger)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if scheduler.num_bad_epochs > scheduler.patience * 2:
                print_log('Patience exhausted, early stopping', logger = logger) 
                break


def train(base_model, train_dataloader, epoch, args, optimizer, config, logger=None):
    base_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(['Loss'])
    n_batches = len(train_dataloader)
    losses = []
    batch_start_time = time.time()
    for i, (neighborhood, center, true_offset, points, idx, label, filterpre) in enumerate(train_dataloader):
        n_itr = epoch * n_batches + i
        data_time.update(time.time() - batch_start_time)
        neighborhood, center, true_offset, points = neighborhood.cuda(), center.cuda(), true_offset.cuda(), points.cuda()
        with autocast():
            if config.fast_filter:
                _, true_offset, points, label = filter_to_inner_quadrant(true_offset, true_offset, label, points, filterpre)
            prediction = base_model(neighborhood, center, points, idx)
            if config.slow_filter:
                prediction, true_offset, points, label = filter_to_inner_quadrant(prediction, true_offset, label, points)
            prediction = prediction * torch.tensor(config.dataset.val._base_.normalization_pars).cuda()
            true_offset = true_offset * torch.tensor(config.dataset.val._base_.normalization_pars).cuda()
            loss = base_model.module.loss_ce(prediction, true_offset, get_l1=True)
        scaler.scale(loss).backward()
        if config.get('grad_norm_clip') is not None:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        base_model.zero_grad()
        losses.append(loss.item())
        wandb.log({'Loss / Batch': loss.item(), 'LR / Batch': optimizer.param_groups[0]['lr'], 'Step': n_itr})

        batch_time.update(time.time() - batch_start_time)
        if i % 20 == 0:
            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %.3f' %
                        (epoch, config.max_epoch, i + 1, n_batches, batch_time.val(), data_time.val(),
                        loss.item()), logger = logger)
        if hasattr(config, "max_num_batches"):
            if i > config.max_num_batches:
                break
        
        batch_start_time = time.time()
    eval_value = -np.mean(losses)
    print_log("TRAINING Loss: %.4f" % np.mean(losses), logger=logger)
    wandb.log({'Training Loss': np.mean(losses), 'Epoch': epoch})

    return eval_value


def validate(base_model, test_dataloader, epoch, args, config, argmax, logger = None):
    base_model.eval()  # set model to eval mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = []
    batch_start_time = time.time()
    with torch.no_grad():
        for i, (neighborhood, center, true_offset, points, idx, label, filterpre) in enumerate(test_dataloader):
            data_time.update(time.time() - batch_start_time)
            neighborhood, center, true_offset, points = neighborhood.cuda(), center.cuda(), true_offset.cuda(), points.cuda()
            with autocast():
                if config.fast_filter:
                    _, true_offset, points, label = filter_to_inner_quadrant(true_offset, true_offset, label, points, filterpre)
                prediction = base_model(neighborhood, center, points, idx)
                if config.slow_filter:
                    prediction, true_offset, points, label = filter_to_inner_quadrant(prediction, true_offset, label, points)
                prediction = prediction * torch.tensor(config.dataset.val._base_.normalization_pars).cuda()
                true_offset = true_offset * torch.tensor(config.dataset.val._base_.normalization_pars).cuda()
                loss = base_model.module.loss_ce(prediction, true_offset, get_l1=True)
            losses.append(loss.item())
            batch_time.update(time.time() - batch_start_time)
            if i % 20 == 0:
                print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses = %.3f' %
                            (epoch, config.max_epoch, i + 1, len(test_dataloader), batch_time.val(), data_time.val(),
                            loss.item()), logger = logger)
            batch_start_time = time.time()
    eval_value = np.mean(losses)
    print_log("VALIDATION Loss: %.4f" % np.mean(losses), logger=logger)
    wandb.log({'Validation Loss': np.mean(losses), 'Epoch': epoch})

    return eval_value 


def test_offset(args, config):
    base_model, train_dataloader, test_dataloader, optimizer, scheduler, logger, config, start_epoch\
         = builder.prepare_process(args, config)    
    base_model.eval()  # set model to eval mode
    losses = []
    with torch.no_grad():
        # assume batch size 1
        for ids, (neighborhood, center, true_offset, points, idx, label, filterpre) in enumerate(test_dataloader):
            neighborhood, center, true_offset, points = neighborhood.cuda(), center.cuda(), true_offset.cuda(), points.cuda()
            if config.fast_filter:
                _, true_offset, points, label = filter_to_inner_quadrant(true_offset, true_offset, label, points, filterpre)
            prediction = base_model(neighborhood, center, points, idx)
            if config.slow_filter:
                prediction, true_offset, points, label = filter_to_inner_quadrant(prediction, true_offset, label, points)
            data_path = f'./vis/treestring_{ids}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            prediction = prediction * torch.tensor(config.dataset.val._base_.normalization_pars).cuda()
            true_offset = true_offset * torch.tensor(config.dataset.val._base_.normalization_pars).cuda()
            #print(torch.tensor(config.dataset.val._base_.normalization_pars).cuda())
            loss = base_model.module.loss_ce(prediction, true_offset, get_l1=True)
            losses.append(loss.item())

            offset = prediction.detach().cpu().numpy()[0]
            true_offset = true_offset.detach().cpu().numpy()[0]
            points = points.detach().cpu().numpy()[0]
            label = label.cpu().numpy()[0][:, np.newaxis]
            if ids <= 10:
                points[:,2] = points[:,2] + 1
                points[:,:3] = points[:,:3] * config.dataset.val._base_.normalization_pars

                cluster = np.hstack((points - offset, label))
                points = np.hstack((points, label))

                np.save(os.path.join(data_path,'cluster.npy'), cluster)
                np.save(os.path.join(data_path,'offset.npy'), offset)
                np.save(os.path.join(data_path,'points.npy'), points)
                np.save(os.path.join(data_path,'true_offset.npy'), true_offset)
                print(data_path, np.round(loss.item(), 4))
        print_log("TEST Loss: %.4f" % np.mean(losses), logger=logger)


def filter_to_inner_quadrant(predictions, true_offset, labels, batch_points, filterpre=None, val=4/15):

    pred_list, true_offset_list, points_list, label_list = [], [], [], []
    i=0
    for points, prediction, label, offset in zip(batch_points, predictions, labels, true_offset):
        if filterpre is None:
            label = label.cuda()
            filter9 = label != 9999
            filterx = torch.logical_and(points[:,0] < val, points[:,0] > -val)
            filtery = torch.logical_and(points[:,1] < val, points[:,1] > -val)
            filterall = torch.logical_and(filter9, filterx)
            filterall = torch.logical_and(filterall, filtery)
        else: 
            filterall = filterpre[i]
            i+=1
        pred_list.append(prediction[filterall])
        true_offset_list.append(offset[filterall])
        points_list.append(points[filterall])
        label_list.append(label[filterall])
    
    return builder.pad_sequence(pred_list), builder.pad_sequence(true_offset_list), builder.pad_sequence(points_list),\
        builder.pad_sequence(label_list)

