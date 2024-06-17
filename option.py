###########################################################################
# Created by: YI ZHENG
# Email: yizheng@bu.edu
# Copyright (c) 2020
###########################################################################

import os
import argparse
import torch

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(description='PyTorch Classification')
        # model and dataset 
        parser.add_argument('--n_class', type=int, default=4, help='classification classes')
        parser.add_argument('--n_features', type=int, default=512, help='number of features per node in the graph')
        parser.add_argument('--data_path', type=str, help='path to dataset where images store')
        parser.add_argument('--train_set', type=str, help='train')
        parser.add_argument('--val_set', type=str, help='validation')
        parser.add_argument('--model_path', type=str, help='path to trained model')
        parser.add_argument('--log_path', type=str, help='path to log files')
        parser.add_argument('--task_name', type=str, help='task name for naming saved model files and log files')
        parser.add_argument('--train', action='store_true', default=False, help='train only')
        parser.add_argument('--test', action='store_true', default=False, help='test only')
        parser.add_argument('--batch_size', type=int, default=6, help='batch size for origin global image (without downsampling)')
        parser.add_argument('--log_interval_local', type=int, default=10, help='classification classes')
        parser.add_argument('--resume', type=str, default="", help='path for model')
        parser.add_argument('--graphcam', action='store_true', default=False, help='GraphCAM')
        parser.add_argument('--site', type=str, default="LUAD", help='site of the dataset, if using a canonical dataset')

        # the parser
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()
        # default settings for epochs and lr

        args.num_epochs = 120
        args.lr = 1e-3             

        if args.test:
            args.num_epochs = 1

        if args.site.strip().lower() == 'none':
            args.site = None

        return args
