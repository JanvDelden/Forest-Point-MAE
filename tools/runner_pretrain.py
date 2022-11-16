import torch, os, time
import numpy as np
from tools import builder
from utils.logger import *
from utils.AverageMeter import AverageMeter
import wandb

def run_net(args, config):
    wandb.init(project="pretrain", entity="jans")
    base_model, train_dataloader, test_dataloader, optimizer, scheduler, logger, config, start_epoch\
         = builder.prepare_process(args, config)
    # parameter setting
    lossvals = []
    for epoch in range(start_epoch, config.max_epoch + 1):

        epoch_start_time = time.time()
        losses = train(base_model, train_dataloader, epoch, args, optimizer, config, logger=logger)  
        lossval = validate(base_model, test_dataloader, epoch, args, config, logger=logger)
        lossvals.append(lossval)
        epoch_end_time = time.time()
        print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %.4f lr = %.6f' %
            (epoch,  epoch_end_time - epoch_start_time, np.mean(losses),
             optimizer.param_groups[0]['lr']), logger = logger)
        scheduler.step(epoch+1)

        if epoch % 100 == 0 or epoch == config.max_epoch:
            builder.save_checkpoint(base_model, optimizer, epoch, None, None, 'ckpt-last', args, logger = logger)
        if lossval <= torch.min(torch.tensor(lossvals)):
            builder.save_checkpoint(base_model, optimizer, epoch, None, None, f'ckpt-best', args, logger=logger)
        
        if hasattr(config, "masking_ratio_freq"):
            if epoch % config.masking_ratio_freq == 0 and epoch != 0:
                config.dataset.train.others.model.transformer_config.mask_ratio += config.masking_ratio_reduction
                print_log(f"updated to masking ratio \
                        {config.dataset.train.others.model.transformer_config.mask_ratio}", logger=logger)
                train_sampler, train_dataloader = builder.dataset_builder(args, config.dataset.train)
                #test_dataloader.dataset.masker.masking_ratio = test_dataloader.dataset.masker.masking_ratio + 0.1




def train(base_model, train_dataloader, epoch, args, optimizer, config, logger=None):
    base_model.zero_grad()
    base_model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = []

    base_model.train()  # set model to training mode
    n_batches = len(train_dataloader)
    for idx, (neighborhood, center, mask) in enumerate(train_dataloader):
        batch_start_time = time.time()
        n_itr = epoch * n_batches + idx
        data_time.update(time.time() - batch_start_time)            
        neighborhood, center, mask = neighborhood.cuda(), center.cuda(), mask.cuda()
        
        loss = base_model(neighborhood, center, mask)
        loss = loss.mean()
        loss.backward()
        # forward
        optimizer.step()
        base_model.zero_grad()
        losses.append(loss.item()* 1000) 

        wandb.log({'Loss / Batch': loss.item()*1000, 'LR / Batch': optimizer.param_groups[0]['lr'], 'Step': n_itr})

        batch_time.update(time.time() - batch_start_time)
        if idx % 20 == 0:
            print_log('[Epoch %d/%d][Batch %d/%d] BatchTime = %.2f (s) DataTime = %.2f (s) Losses = %.4f lr = %.6f' %
                        (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                        np.mean(losses), optimizer.param_groups[0]['lr']), logger = logger)
        if hasattr(config, "max_num_batches"):
            if idx > config.max_num_batches:
                break
        
    wandb.log({'Training Loss': np.mean(losses), 'Epoch': epoch})
    return losses


def validate(base_model, test_dataloader, epoch, args, config, logger=None):
    losses = []
    base_model.eval()  # set model to eval mode
    with torch.no_grad():
        n_batches = len(test_dataloader)
        for idx, (neighborhood, center, mask) in enumerate(test_dataloader):
            n_itr = epoch * n_batches + idx
            neighborhood, center, mask = neighborhood.cuda(), center.cuda(), mask.cuda()
            loss = base_model(neighborhood, center, mask)
            loss = loss.mean()
            losses.append(loss.item()* 1000) 

        print_log('[Validation] Losses = %.4f' % np.mean(losses), logger=logger)
        wandb.log({'Validation Loss': np.mean(losses), 'Epoch': epoch})

        return np.mean(losses)

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger = logger)
    config.dataset.test._base_.model = config.model
    _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    config.model.sampling_method = args.sampling_method
    base_model = builder.model_builder(args, config.model)
    builder.load_model(base_model, args.ckpts, logger = logger)

    if args.use_gpu:
        base_model.cuda()
    test(base_model, test_dataloader, args, config, logger=logger)

def test(base_model, test_dataloader, args, config, logger = None):

    base_model.eval()  # set model to eval mode
    target = './vis'
    with torch.no_grad():
        for idx, (neighborhood, center, mask) in enumerate(test_dataloader):
            neighborhood, center, mask = neighborhood.cuda(), center.cuda(), mask.cuda()
            full_original, full_rebuilt, visible, rebuilt_points, masked = base_model(neighborhood, center, mask, vis=True)
            data_path = f'./vis/treestring_{idx}'
            if not os.path.exists(data_path):
                os.makedirs(data_path)

            np.save(os.path.join(data_path,'full_visible.npy'), visible.detach().cpu().numpy())
            np.save(os.path.join(data_path,'full_rebuilt.npy'), full_rebuilt.detach().cpu().numpy())
            np.save(os.path.join(data_path,'full_original.npy'), full_original.detach().cpu().numpy())
            np.save(os.path.join(data_path,'only_rebuilt.npy'), rebuilt_points.detach().cpu().numpy())
            np.save(os.path.join(data_path,'masked_points.npy'), masked.detach().cpu().numpy())
            print(data_path)

            if idx > 10:
                break
        return