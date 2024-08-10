import torch.utils.data
import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
from pria.network_modules import *
import random
import os
from matplotlib import pyplot as plt
from PIL import Image
import cv2

class Se3Ish(nn.Module):
  def __init__(self):
    super(Se3Ish, self).__init__()
    self.rot_dim = 4

    self.convB1 = ConvBNReLU(C_in=3,C_out=64,kernel_size=7,stride=2)
    self.poolB1 = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
    self.convB2 = ResnetBasicBlock(64,64,bias=True)
    self.convB3 = ResnetBasicBlock(64,64,bias=True)

    self.convAB1 = ConvBNReLU(128,256,kernel_size=3,stride=2)
    self.convAB2 = ResnetBasicBlock(256,256,bias=True)
    self.convAB2 = ResnetBasicBlock(256,256,bias=True)

    self.trans_conv1 = ConvBNReLU(64,512,kernel_size=3,stride=2)
    self.trans_conv2 = ResnetBasicBlock(512,512,bias=True)
    self.trans_pool1 = nn.AdaptiveAvgPool2d(1)
    self.trans_out = nn.Sequential(nn.Linear(512,3),nn.Tanh())

    self.rot_conv1 = ConvBNReLU(64,512,kernel_size=3,stride=2)
    self.rot_conv2 = ResnetBasicBlock(512,512,bias=True)
    self.rot_pool1 = nn.AdaptiveAvgPool2d(1)
    self.rot_out = nn.Sequential(nn.Linear(512,self.rot_dim),nn.Tanh())

    self.loss = nn.MSELoss(reduction='none')
    self.loss_coefficient = 0.01

  def forward(self, B):
    batch_size = B.shape[0]
    # print(B.shape)
    output = {}
    # print(B.shape)
    # a = self.convA1(A)
    # a = self.poolA1(a)
    # a = self.convA2(a)

    b = self.convB1(B)
    # print("b ", b.shape)
    b = self.poolB1(b)
    # print("b ", b.shape)
    b = self.convB2(b)
    # print("b ", b.shape)
    b = self.convB3(b)
    # print("b ", b.shape)

    # ab = torch.cat((a,b),1).contiguous()
    # ab = self.convAB1(ab)
    # ab = self.convAB2(ab)
    # output['feature'] = ab

    trans = self.trans_conv1(b)
    # print("trans ", trans.shape)
    trans = self.trans_conv2(trans)
    # print("trans ", trans.shape)
    trans = self.trans_pool1(trans)
    # print("trans ", trans.shape)
    trans = trans.reshape(batch_size,-1)
    # print("trans ", trans.shape)
    trans = self.trans_out(trans).contiguous()
    # print("trans ", trans.shape)
    output['trans'] = trans

    rot = self.rot_conv1(b)
    # print("rot ", rot.shape)
    rot = self.rot_conv2(rot)
    # print("rot ", rot.shape)
    rot = self.rot_pool1(rot)
    # print("rot ", rot.shape)
    rot = rot.reshape(batch_size,-1)
    # print("rot ", rot.shape)
    rot = self.rot_out(rot).contiguous()
    # print("rot ", rot.shape)
    output['rot'] = rot

    # print("output ", output)
    return torch.cat((output['trans'], output['rot']), 1)

  # def loss(self, predictions, targets):
  #   output = {}
  #   trans_loss = nn.MSELoss()(predictions['trans'].float(), targets[:,:3].float())
  #   rot_loss = nn.MSELoss()(predictions['rot'].float(), targets[:,3:].float())
  #   output['trans'] = trans_loss
  #   output['rot'] = rot_loss

  #   return output
  def compute_loss(self, pred, gt):
      translation_loss = torch.nn.MSELoss(reduction='none')(pred[:,:3], gt[:,:3]).mean()
      rotation_loss = torch.nn.MSELoss(reduction='none')(pred[:,3:], gt[:,3:]).mean()
      loss = translation_loss + self.loss_coefficient * rotation_loss
      print(loss.shape)
      return loss
