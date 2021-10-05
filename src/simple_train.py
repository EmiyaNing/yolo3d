'''
Description:
    This script for training.
    It will training the yolo3d model in a single GPU.
    It will save the best result model after 50 epochs.
    This script will use the simplest way to construct the code.
'''
import sys
import os
import random
sys.path.append('..')

import time
import numpy as np
import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from data_process.kitti_dataloader import create_train_dataloader, create_val_dataloader
from models.model_utils import create_model
from utils.train_utils import create_optimizer, create_lr_scheduler, get_saved_state, save_checkpoint
from utils.train_utils import to_python_float, get_tensorboard_log
from utils.misc import AverageMeter, ProgressMeter
from config.train_config import parse_train_configs
from evaluate import evaluate_mAP

def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer):
    '''
    Description:
        This function used to train a model one epoch.
    '''
    batch_time = AverageMeter('Time',':6.3f')
    data_time  = AverageMeter('Data',':6.3f')
    losses     = AverageMeter('Loss',':.4e')
    progress   = ProgressMeter(len(train_dataloader), [batch_time, data_time, losses],
                               prefix='Train - Epoch: [{}/{}]'.format(epoch, configs.num_epochs))
    num_iters_per_epoch = len(train_dataloader)

    model.train()
    start_time = time.time()
    for batch_idx, batch_data  in enumerate(tqdm(train_dataloader)):
        # init some statistics code
        data_time.update(time.time() - start_time)
        _, imgs, targets = batch_data
        global_step = num_iters_per_epoch * (epoch - 1) + batch_idx + 1
        batch_size = imgs.size(0)

        # transfer the data from cpu memory to gpu memory
        targets = targets.to(configs.device, non_blocking=True)
        imgs = imgs.to(configs.device, non_blocking=True)
        # forward the model
        total_loss, outputs = model(imgs, targets)
        # backward the model
        total_loss.backward()
        # optimizer's update
        if global_step % configs.subdivisions == 0:
            optimizer.step()
            lr_scheduler.step()
            if tb_writer is not None:
                tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], global_step)
            # zero the parameter gradients
            optimizer.zero_grad()

        # update the static information
        losses.update(to_python_float(total_loss.data), batch_size)
        batch_time.update(time.time() - start_time)

        if tb_writer is not None:
            if (global_step % configs.tensorboard_freq) == 0:
                tensorboard_log = get_tensorboard_log(model)
                tb_writer.add_scalar('avg_loss', losses.avg, global_step)
                for layer_name, layer_dict in tensorboard_log.items():
                    tb_writer.add_scalars(layer_name, layer_dict, global_step)

        if logger is not None:
            if (global_step % configs.print_freq) == 0:
                logger.info(progress.get_message(batch_idx))
        # upadate the start time
        start_time = time.time()

def main():
    configs = parse_train_configs()
    configs.subdivisions = int(64 / configs.batch_size)
    global_ap        = None


    logger = Logger(configs.logs_dir, configs.saved_fn)
    logger.info('>>> Created a new logger')
    logger.info('>>> configs: {}'.format(configs))
    tb_writer = SummaryWriter(log_dir=os.path.join(configs.logs_dir, 'tensorboard'))

    model  = create_model(configs)
    # load the pretrain checkpoint
    if configs.pretrained_path is not None:
        assert os.path.isfile(configs.pretrained_path), "=> no checkpoint found at '{}'".format(configs.pretrained_path)
        model.load_state_dict(torch.load(configs.pretrained_path))
        if logger is not None:
            logger.info('loaded pretrained model at {}'.format(configs.pretrained_path))

    # resume weights of model from a checkpoint
    if configs.resume_path is not None:
        assert os.path.isfile(configs.resume_path), "=> no checkpoint found at '{}'".format(configs.resume_path)
        model.load_state_dict(torch.load(configs.resume_path))
        if logger is not None:
            logger.info('resume training model from checkpoint {}'.format(configs.resume_path))

    # create optimizer && lr_scheduler
    optimizer = create_optimizer(configs, model)
    lr_scheduler = create_lr_scheduler(optimizer, configs)

    # resume optimizer && lr_scheduler
    if configs.resume_path is not None:
        utils_path = configs.resume_path.replace('Model_', 'Utils_')
        assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
        utils_state_dict = torch.load(utils_path, map_location='cuda:0')
        optimizer.load_state_dict(utils_state_dict['optimizer'])
        lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
        configs.start_epoch = utils_state_dict['epoch'] + 1

    # logger the training parameters of model
    if configs.is_master_node:
        num_parameters = get_num_parameters(model)
        logger.info('number of trained parameters of the model: {}'.format(num_parameters))

    # create train/val dataloader
    train_dataloader, train_sampler = create_train_dataloader(configs)
    val_dataloader = create_val_dataloader(configs)

    # epoch training loop. training model, eval model, save best model
    for epoch in range(configs.start_epoch, configs.num_epochs + 1):
        if logger is not None:
            logger.info('{}'.format('*-' * 40))
            logger.info('{} {}/{} {}'.format('=' * 35, epoch, configs.num_epochs, '=' * 35))
            logger.info('{}'.format('*-' * 40))
            logger.info('>>> Epoch: [{}/{}]'.format(epoch, configs.num_epochs))

        train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, logger, tb_writer)
        if (epoch > 50) and (epoch % configs.checkpoint_freq == 0):
            print('number of batches in val_dataloader: {}'.format(len(val_dataloader)))
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs, logger)
            val_metrics_dict = {
                'precision': precision.mean(),
                'recall': recall.mean(),
                'AP': AP.mean(),
                'f1': f1.mean(),
                'ap_class': ap_class.mean()
            }
            if global_ap is not None:
                if AP.mean() > global_ap:
                    model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
                    save_best_checkpoint(configs.best_dir, AP.mean(), model_state_dict, utils_state_dict, epoch)
                global_ap = AP.mean()
            else:
                global_ap = AP.mean()
                model_state_dict, utils_state_dict = get_saved_state(model, optimizer, lr_scheduler, epoch, configs)
                save_checkpoint(configs.checkpoint_dir, configs.saved_fn, model_state_dict, utils_state_dict, epoch)
            if tb_writer is not None:
                tb_writer.add_scalars('Validation', val_metrics_dict, epoch)

            if not configs.step_lr_in_epoch:
                lr_scheduler.step()
                if tb_writer is not None:
                    tb_writer.add_scalar('LR', lr_scheduler.get_lr()[0], epoch)

    # close the tbwriter
    if tb_writer is not None:
        tb_writer.close()

if __name__ == '__main__':
    main()