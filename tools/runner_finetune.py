import torch
from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.misc import Acc_Metric
import os
import numpy as np
import wandb
from sklearn import metrics


def run_net(args, config):
    wandb.init(project="cls", entity="jans")
    base_model, train_dataloader, test_dataloader, optimizer, scheduler, logger, config, start_epoch\
         = builder.prepare_process(args, config)

    # parameter setting
    best_metrics = Acc_Metric(0.)
    metrics = Acc_Metric(0.)

    # trainval
    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter(['loss', 'acc'])
        base_model.train()  # set model to training mode
        n_batches = len(train_dataloader)
        
        for idx, (neighborhood, center, label) in enumerate(train_dataloader):
            n_itr = epoch * n_batches + idx
            data_time.update(time.time() - batch_start_time)
            
            neighborhood, center, label = neighborhood.cuda(), center.cuda(), label.cuda()
            if args.fewshot:
                label[label == 4] = 3
            prediction = base_model(neighborhood, center)
            loss, acc = base_model.module.get_loss_acc(prediction, label)
            lossval = loss.item()
            loss.backward()

            if config.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
            optimizer.step()
            base_model.zero_grad()
            losses.update([lossval, acc.item()])
            wandb.log({'Loss / Batch': lossval, 'Accuracy / Batch': acc.item()}, step=n_itr)

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()

        wandb.log({'Training Loss': losses.avg(0), 'Training Accuracy': losses.avg()[1], 'Epoch': epoch}, step=n_itr)

        # Validate the current models
        metrics, predictions, labels, lossval = validate(base_model, test_dataloader, epoch, args, config, logger=logger)
        schedulerval = lossval if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else epoch+1
        better = metrics.better_than(best_metrics)
        # Save checkpoints
        if better:
            best_metrics = metrics
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args, logger = logger)
            path = os.path.join(args.experiment_path)
            np.save(os.path.join(path, "predictions.npy"), predictions.cpu().numpy())
            np.save(os.path.join(path, "labels.npy"), labels.cpu().numpy())
            if args.fewshot:
                path = os.path.join(path, "fewshot.txt")
                with open(path, "a") as f:
                    f.write("\n")
                    f.write(f"{best_metrics.acc.item()}")
            print_log("--------------------------------------------------------------------------------------------", logger=logger)
        epoch_end_time = time.time()
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()],optimizer.param_groups[0]['lr']), logger = logger)
        scheduler.step(schedulerval)



def validate(base_model, test_dataloader, epoch, args, config, logger = None):
    base_model.eval()  # set model to eval mode
    test_pred, test_label, losses  = [], [], []
    with torch.no_grad():
        for idx, (neighborhood, center, label) in enumerate(test_dataloader):
            neighborhood, center, label = neighborhood.cuda(), center.cuda(), label.cuda()
            if args.fewshot:
                label[label == 4] = 3
            prediction = base_model(neighborhood, center)
            loss, acc = base_model.module.get_loss_acc(prediction, label)
            losses.append(loss.item())
            target = label.view(-1)
            pred = prediction.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        oAcc = metrics.accuracy_score(test_label.detach().cpu().numpy(), test_pred.detach().cpu().numpy()) * 100
        mAcc = metrics.balanced_accuracy_score(test_label.detach().cpu().numpy(), test_pred.detach().cpu().numpy()) * 100
        print_log('[Validation] EPOCH: %d  oAcc = %.4f, mAcc = %.4f,loss = %.4f' % (epoch, oAcc, mAcc, np.mean(losses)), logger=logger)

    wandb.log({'Validation Accuracy':oAcc, 'Validation Loss': np.mean(losses), 'Validation mean Accuracy': mAcc, 'Epoch': epoch})
    return Acc_Metric(oAcc), test_pred, test_label, np.mean(losses)


def test_cls(args, config):
    base_model, train_dataloader, test_dataloader, optimizer, scheduler, logger, config, start_epoch\
         = builder.prepare_process(args, config)
    times = 10
    vote = False
    base_model.eval()  
    test_pred, test_label, losses  = [], [], []

    with torch.no_grad():
        for idx, (neighborhood, center, label) in enumerate(test_dataloader):
            neighborhood, center, label = neighborhood.cuda(), center.cuda(), label.cuda()
            if args.fewshot:
                label[label == 4] = 3
            prediction = base_model(neighborhood, center)
            loss, acc = base_model.module.get_loss_acc(prediction, label)
            losses.append(loss.item())
            target = label.view(-1)
            pred = prediction.argmax(-1).view(-1)

            test_pred.append(pred.detach())
            test_label.append(target.detach())

        test_pred = torch.cat(test_pred, dim=0)
        test_label = torch.cat(test_label, dim=0)
        oAcc = metrics.accuracy_score(test_label.detach().cpu().numpy(), test_pred.detach().cpu().numpy()) * 100
        mAcc = metrics.balanced_accuracy_score(test_label.detach().cpu().numpy(), test_pred.detach().cpu().numpy()) * 100
        path = os.path.join(args.experiment_path)
        np.save(os.path.join(path, "predictions.npy"), test_pred.cpu().numpy())
        np.save(os.path.join(path, "labels.npy"), test_label.cpu().numpy())
        print_log('[Testing]  oAcc = %.4f, mAcc = %.4f,loss = %.4f' % (oAcc, mAcc, np.mean(losses)), logger=logger)

        wandb.log({'Test Accuracy':oAcc, 'Test Loss': np.mean(losses), 'Test mean Accuracy': mAcc})
    return 

