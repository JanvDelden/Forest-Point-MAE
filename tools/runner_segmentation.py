import torch
import torch.nn as nn
from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import os
import numpy as np
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
def run_net(args, config, train_writer=None, val_writer=None):
    
    # parameter setting
    eval_value = []

    base_model, train_dataloader, test_dataloader, optimizer, scheduler, logger, config, start_epoch\
         = builder.prepare_process(args, config)
    for epoch in range(start_epoch, config.max_epoch + 1):

        epoch_start_time = time.time()
        argmax = train(base_model, train_dataloader, epoch, train_writer, args, optimizer, config, logger=logger) 
        eval_val = validate(base_model, test_dataloader, epoch, val_writer, args, config, argmax, logger=logger)
        eval_value.append(eval_val)
        schedulerval = eval_val if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else epoch
        scheduler.step(schedulerval)
        if epoch % 25 == 0:
            builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-last', args, logger = logger)      
        if eval_val >= torch.max(torch.tensor(eval_value)):
            builder.save_checkpoint(base_model, optimizer, epoch, None, None, f'ckpt-best', args, logger=logger)
            print_log("--------------------------------------------------------------------------------------------", logger=logger)
        epoch_end_time = time.time()
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) lr = %.6f' % (epoch, epoch_end_time - epoch_start_time, optimizer.param_groups[0]['lr']), logger = logger) 
        train_writer.flush()
        val_writer.flush()

    if train_writer is not None:
        train_writer.close()
    if val_writer is not None:
        val_writer.close()


def train(base_model, train_dataloader, epoch, train_writer, args, optimizer, config, logger=None):
    base_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(['Loss'])
    n_batches = len(train_dataloader)
    tracker, losses = MetricsTracker(string = "TRAINING", logger=logger), []
    batch_start_time = time.time()
    for i, (neighborhood, center, label, points, idx, _) in enumerate(train_dataloader):
        n_itr = epoch * n_batches + i
        data_time.update(time.time() - batch_start_time)
        neighborhood, center, label, points = neighborhood.cuda(), center.cuda(), label.cuda(), points.cuda()
        with autocast():
            prediction = base_model(neighborhood, center, points, idx)
            tracker(prediction, label)
            loss = base_model.module.loss_ce(prediction, label)
        scaler.scale(loss).backward()
        if config.get('grad_norm_clip') is not None:
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
        scaler.step(optimizer)
        scaler.update()
        base_model.zero_grad()

        losses.append(loss.item())
        if train_writer is not None:
            train_writer.add_scalar('Loss/Batch', loss.item(), n_itr)
            train_writer.add_scalar('Loss/Batch', optimizer.param_groups[0]['lr'], n_itr)
        batch_time.update(time.time() - batch_start_time)
        batch_start_time = time.time()
    acc, precision, recall, f1score, argmax = tracker.get_metrics()
    tracker.print_metrics(np.mean(losses), acc, precision, recall, f1score, argmax)
    eval_value = argmax

    if train_writer is not None:
        train_writer.add_scalar('Loss/Epoch', np.mean(losses), epoch)
    return eval_value


def validate(base_model, test_dataloader, epoch, val_writer, args, config, argmax, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode
    batch_time = AverageMeter()
    data_time = AverageMeter()
    tracker, losses = MetricsTracker(string = "VALIDATION", logger=logger), []
    batch_start_time = time.time()
    with torch.no_grad():
        for i, (neighborhood, center, label, points, idx, _) in enumerate(test_dataloader):
            data_time.update(time.time() - batch_start_time)
            neighborhood, center, label, points = neighborhood.cuda(), center.cuda(), label.cuda(), points.cuda()
            with autocast():
                prediction = base_model(neighborhood, center, points, idx)
                tracker(prediction, label)
                loss = base_model.module.loss_ce(prediction, label)
            losses.append(loss.item())
            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
    eval_value = -np.mean(losses)
    acc, precision, recall, f1score, _ = tracker.get_metrics(argmax)
    tracker.print_metrics(np.mean(losses), acc, precision, recall, f1score, argmax)
    eval_value = f1score

    # Add testing results to TensorBoard
    if val_writer is not None:
        val_writer.add_scalar('Metric/loss', loss, epoch)
    return eval_value 


class MetricsTracker:
    def __init__(self, string, logger):
        self.correct = []
        self.ntotal = []
        self.pre = []
        self.rec = []
        self.f1 = []
        self.string = string
        self.logger = logger
        self.thresholds = torch.tensor([0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5,
                                    0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72,
                                    0.74, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98]).reshape(1, 1, 35)
    
    def __call__(self, prediction, target):
        """
        prediction: b n 2
        target: b n 
        """
        prediction = prediction.cpu()[:,:,1]
        target = target.cpu().unsqueeze(2).numpy()
        # the higher the threshold the less  points are assigned to belong to the tree class and not the background
        pred_choice_varying_threshold = (torch.exp(prediction.unsqueeze(2)) >= self.thresholds).numpy()

        # get confusion values for different thresholds
        tp = np.sum(np.logical_and(target == 1, pred_choice_varying_threshold == 1), axis=1)
        fn = np.sum(np.logical_and(target == 1, pred_choice_varying_threshold == 0), axis=1)
        fp = np.sum(np.logical_and(target == 0, pred_choice_varying_threshold == 1), axis=1)
        tn = np.sum(np.logical_and(target == 0, pred_choice_varying_threshold == 0), axis=1)
        if not np.all(tp + fn > 0):
            print("Logicerror")

        # calculate confusion values for every sample in batch for the different thresholds
        precision_varying_threshold = tp / (tp + fp) # tp / all rated positive
        idx = np.isnan(precision_varying_threshold)
        precision_varying_threshold[idx] = 0
        recall_varying_threshold = tp / (tp + fn) # tp / all actual positive
        idx = np.isnan(recall_varying_threshold)
        recall_varying_threshold[idx] = 0

        # append precision
        self.pre.append(precision_varying_threshold)
        self.rec.append(recall_varying_threshold)

        # append correct classifications and count number of points
        self.correct.append(np.sum(pred_choice_varying_threshold == target, axis=1))
        self.ntotal.append(target.shape[0] * target.shape[1])

    def get_metrics(self,argmax=None):
        precision = np.vstack(self.pre)
        recall = np.vstack(self.rec)
        correct = np.vstack(self.correct)
        f1scores = 2 * (precision * recall) / (precision + recall)
        f1scores[np.isnan(f1scores)] = 0
        f1scores = np.mean(f1scores, axis=0)
        if argmax is None:
            argmax = np.argmax(f1scores)

        acc = np.round(np.sum(correct[:, argmax]) / np.sum(self.ntotal), 5)
        precision = np.round(np.mean(precision[:, argmax]), 5)
        recall = np.round(np.mean(recall[:, argmax]), 5)
        f1score = np.round(f1scores[argmax], 5)

        return acc, precision, recall, f1score, argmax
    
    def print_metrics(self, loss, acc, precision, recall, f1score, argmax):
        string = self.string
        logger = self.logger
        print_log('[%s] Loss = %.4f acc = %.4f pre = %.4f rec = %.4f f1 = %.4f, th = % .2f' % (string, loss, acc, precision, recall, f1score, self.thresholds[0,0,argmax]), logger=logger)


def test_seg(args, config, train_writer=None, val_writer=None):

    base_model, train_dataloader, test_dataloader, optimizer, scheduler, logger  = builder.prepare_process(args, config)
    tracker, losses = MetricsTracker(string = "VALIDATION", logger=logger), []
    base_model.eval()  # set model to eval mode
    with torch.no_grad():
        # assume batch size 1
        for ids, (neighborhood, center, label, points, idx, _) in enumerate(test_dataloader):
            neighborhood, center, label, points = neighborhood.cuda(), center.cuda(), label.cuda(), points.cuda()
            prediction = base_model(neighborhood, center, points, idx)
            data_path = f'./vis/treestring_{ids}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)
            tracker(prediction, label)

            prediction = np.exp(prediction.detach().cpu().numpy()[0])[:, 1:]
            points = points.detach().cpu().numpy()[0]
            label = label.detach().cpu().numpy()[0][:, np.newaxis]
            gt = np.hstack((points, label))
            prediction = np.hstack((points, prediction))

            np.save(os.path.join(data_path,'points.npy'), gt)
            np.save(os.path.join(data_path,'prediction.npy'), prediction)

            if ids > 10: 
                break
        
        acc, precision, recall, f1score, _ = tracker.get_metrics(16)
        tracker.print_metrics(0, acc, precision, recall, f1score, 16)
    if train_writer is not None:
        train_writer.close()
    if train_writer is not None:
        val_writer.close()
