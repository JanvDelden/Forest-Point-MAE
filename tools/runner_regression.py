import torch
from tools import builder
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
import os
import numpy as np
import wandb


def run_net(args, config, train_writer=None, val_writer=None):
    wandb.init(project="regression", entity="jans")
     # parameter setting
    best_metrics = 9000000
    metrics = 900000

    base_model, train_dataloader, test_dataloader, optimizer, scheduler, logger, config, start_epoch\
         = builder.prepare_process(args, config)

    # training
    for epoch in range(start_epoch, config.max_epoch + 1):
        base_model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = []
        base_model.train()  # set model to training mode
        for idx, (neighborhood, center, label) in enumerate(train_dataloader):
            data_time.update(time.time() - batch_start_time)
            neighborhood, center, label = neighborhood.cuda(), center.cuda(), label.cuda()
            prediction = base_model(neighborhood, center)
            loss = base_model.module.loss_ce(prediction.squeeze(), label)
            loss.backward()
            if config.get('grad_norm_clip') is not None:
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), config.grad_norm_clip, norm_type=2)
            optimizer.step()
            base_model.zero_grad()
            losses.append(loss.item())

            batch_time.update(time.time() - batch_start_time)
            batch_start_time = time.time()
    
        losses = np.array(losses)
        epoch_end_time = time.time()
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %.4f lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, losses.mean(),optimizer.param_groups[0]['lr']), logger = logger)
        wandb.log({'Training Loss': losses.mean(), 'Epoch': epoch})
        # Validate the current models
        metrics, predictions = validate(base_model, test_dataloader, epoch, val_writer, args, config, logger=logger)
        schedulerval = metrics if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else epoch+1
        scheduler.step(schedulerval)
        if metrics < best_metrics:
            best_metrics = metrics
            path = os.path.join(args.experiment_path)
            np.save(os.path.join(path, "predictions.npy"), predictions.numpy())
            #builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-best', args, logger = logger)
            print_log("--------------------------------------------------------------------------------------------", logger=logger)
        #if epoch % 100 == 0:
        #    builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-last', args, logger = logger)      

def validate(base_model, test_dataloader, epoch, val_writer, args, config, logger = None):
    # print_log(f"[VALIDATION] Start validating epoch {epoch}", logger = logger)
    base_model.eval()  # set model to eval mode

    losses = []
    predictions = []
    with torch.no_grad():
        for idx, (neighborhood, center, label) in enumerate(test_dataloader):
            neighborhood, center, label = neighborhood.cuda(), center.cuda(), label.cuda()
            prediction = base_model(neighborhood, center)
            loss = base_model.module.loss_ce(prediction.squeeze(), label)

            losses.append(loss.detach())
            predictions.append(prediction.detach())
        if len(losses) == 1:
            losses = torch.tensor(losses)
            predictions = predictions[0].cpu()
        else:
            losses = torch.cat(losses, dim=0)

        print_log('[Validation] EPOCH: %d  loss = %.4f' % (epoch, losses.mean()), logger=logger)
        wandb.log({'Validation Loss': losses.mean(), 'Epoch': epoch})


    return losses.mean(), predictions





