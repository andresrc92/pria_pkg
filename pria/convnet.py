import torch.utils.data
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
import random
import os
from matplotlib import pyplot as plt
from PIL import Image
import cv2

class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.conv1 = nn.Conv2d(3, 32, kernel_size=7)
    self.pool1 = nn.MaxPool2d(3, 3)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=7)
    self.drp1 = nn.Dropout2d(0.25)
    self.pool2 = nn.MaxPool2d(3, 3)
    self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
    self.drp2 = nn.Dropout2d(0.25)
    self.pool3 = nn.MaxPool2d(2, 2)
    self.lin1 = nn.Linear(21120, 7)

    self.loss_coefficient = 0.01

  def forward(self, x):
    # print(x.shape)
    x = self.conv1(x)
    # print(x.shape)
    x = torch.relu(x)
    # print(x.shape)
    x = self.pool1(x)
    # print(x.shape)
    x = self.conv2(x)
    # print(x.shape)
    x = self.drp1(x)
    # print(x.shape)
    x = torch.relu(x)
    # print(x.shape)
    x = self.pool2(x)
    # print(x.shape)
    x = self.conv3(x)
    # print(x.shape)
    x = self.drp2(x)
    # print(x.shape)
    x = torch.relu(x)
    # print(x.shape)
    x = self.pool3(x)
    # print(x.shape)
    # x = x.view(-1, 21120)
    x = x.reshape(-1, 21120)
    # print(x.shape)
    x = self.lin1(x)
    # print(x.shape)
    return x
  
  def compute_loss(self, pred, gt):
    translation_loss = nn.MSELoss(reduction='none')(pred[:,:3], gt[:,:3]).mean()
    rotation_loss = nn.MSELoss(reduction='none')(pred[:,3:], gt[:,3:]).mean()
    loss = translation_loss + self.loss_coefficient * rotation_loss
    return loss
