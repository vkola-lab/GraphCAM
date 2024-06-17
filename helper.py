#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from utils.metrics import ConfusionMatrix
from PIL import Image
import os

# torch.cuda.synchronize()
# torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True

def collate(batch):
    image = [ b['image'] for b in batch ] # w, h
    label = [ b['label'] for b in batch ]
    id = [ b['id'] for b in batch ]
    adj_s = [ b['adj_s'] for b in batch ]
    return {'image': image, 'label': label, 'id': id, 'adj_s': adj_s}

def preparefeatureLabel(batch_graph, batch_label, batch_adjs, n_features: int = 512):
    batch_size = len(batch_graph)
    labels = torch.LongTensor(batch_size)
    max_node_num = 0

    for i in range(batch_size):
        labels[i] = batch_label[i]
        max_node_num = max(max_node_num, batch_graph[i].shape[0])
    
    masks = torch.zeros(batch_size, max_node_num)
    adjs =  torch.zeros(batch_size, max_node_num, max_node_num)
    batch_node_feat = torch.zeros(batch_size, max_node_num, n_features)

    for i in range(batch_size):
        cur_node_num =  batch_graph[i].shape[0]
        #node attribute feature
        tmp_node_fea = batch_graph[i]
        batch_node_feat[i, 0:cur_node_num] = tmp_node_fea

        #adjs
        adjs[i, 0:cur_node_num, 0:cur_node_num] = batch_adjs[i]
        
        #masks
        masks[i,0:cur_node_num] = 1  

    node_feat = batch_node_feat.cuda()
    labels = labels.cuda()
    adjs = adjs.cuda()
    masks = masks.cuda()

    return node_feat, labels, adjs, masks

class Trainer(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)

    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()
    
    def plot_cm(self):
        self.metrics.plotcm()

    def train(self, sample, model, n_features: int = 512):
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'], n_features=n_features)
        pred,labels,loss = model.forward(node_feat, labels, adjs, masks)

        return pred,labels,loss

class Evaluator(object):
    def __init__(self, n_class):
        self.metrics = ConfusionMatrix(n_class)
    
    def get_scores(self):
        acc = self.metrics.get_scores()

        return acc

    def reset_metrics(self):
        self.metrics.reset()

    def plot_cm(self):
        self.metrics.plotcm()

    def eval_test(self, sample, model, graphcam_flag=False, n_features : int = 512):
        node_feat, labels, adjs, masks = preparefeatureLabel(sample['image'], sample['label'], sample['adj_s'], n_features=n_features)
        if not graphcam_flag:
            with torch.no_grad():
                pred,labels,loss = model.forward(node_feat, labels, adjs, masks)
        else:
            torch.set_grad_enabled(True)
            pred,labels,loss= model.forward(node_feat, labels, adjs, masks, graphcam_flag=graphcam_flag)
        return pred,labels,loss