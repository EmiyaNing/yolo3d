"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Kangling Ning
# DoC: 2021.10.2
# email: ningkangl@icloud.com
-----------------------------------------------------------------------------------
# Description: Test the model in test dataset, and save the result to a .txt file.
"""

import argparse
import sys
import os
import time

from easydict import EasyDict as edict
import cv2
import torch
import numpy as np

sys.path.append('../')

import config.kitti_config as cnf
from data_process import kitti_data_utils, kitti_bev_utils
from data_process.kitti_dataloader import create_test_dataloader
from models.model_utils import create_model
from utils.misc import make_folder
from utils.evaluation_utils import post_processing, rescale_boxes, post_processing_v2
from utils.misc import time_synchronized
from utils.visualization_utils import show_image_with_boxes, merge_rgb_to_bev, predictions_to_kitti_format

def parse_test_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for YOLO3D Implementation')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/yolo3d_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')


    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf_thresh', type=float, default=0.5,
                        help='the threshold for conf')
    parser.add_argument('--nms_thresh', type=float, default=0.5,
                        help='the threshold for conf')



    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.root_dir = '../'
    configs.dataset_dir = os.path.join(configs.root_dir, 'dataset', 'kitti')


    return configs

def test(model, dataloader, configs):
    model.eval()
    for batch_idx, (img_paths, imgs_bev) in enumerate(dataloader):
        input_imgs = imgs_bev.to(device=configs.device).float()
        # start to forward
        t1         = time_synchronized()
        outputs    = model(input_imgs)
        t2         = time_synchronized()
        # forward end
        # Do post processing
        detections = post_processing_v2(outputs, conf_thresh=configs.conf_thresh,
                                        nms_thresh=configs.nms_thresh)
        img_detections = []
        img_detections.extend(detections)
        img_rgb = cv2.imread(img_paths[0])
        calib = kitti_data_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
        # convert model output to kitti data formation
        objects_pred = predictions_to_kitti_format(img_detections, calib, img_rgb.shape, 
                                                   configs.img_size)
        print('\tIn batch ', batch_idx, 'the models output is:')
        for objects in objects_pred:
            print(objects)
            print('\n')
        print('\n\n\n')
        print('\tDone testing the {}th sample, time: {:.1f}ms, speed {:.2f}FPS'.format(batch_idx,
                    (t2 - t1) * 1000,1 / (t2 - t1)))
        print('\n\n\n')

if __name__ == '__main__':
    configs = parse_test_configs()
    configs.distributed = False  # For testing
    model = create_model(configs)
    print('\n\n' + '-*=' * 30 + '\n\n')
    assert os.path.isfile(configs.pretrained_path), "No file at {}".format(configs.pretrained_path)

    model.load_state_dict(torch.load(configs.pretrained_path))

    configs.device = torch.device('cuda:0')
    model = model.to(device=configs.device)
    test_dataloader = create_test_dataloader(configs)

    test(model, test_dataloader, configs)