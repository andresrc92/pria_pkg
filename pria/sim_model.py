# import tensorflow as tf
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
# from se3ish import *
from pria.convnet import *
from pria.rotations import *
import yaml

class Trainer():
  def __init__(self, data_folder):
    self.data_folder = data_folder
    self.imgs_folder = os.path.join(data_folder,'imgs')
    self.gt_path = os.path.join(data_folder,'gt.yaml')
    self.norm_data = os.path.join(data_folder,'convnet_files/norm_param.npy')
    self.model_path = os.path.join(data_folder,'convnet_files/state_dict.pt')
        
    # with open(self.gt_path, 'r') as file:
    #   self.gt_file = yaml.safe_load(file)

    # self.total_images = self.gt_file['initial_pose']['total_images']
    # self.total_images = int(self.total_images)
    # self.width = self.gt_file['initial_pose']['width']
    # self.height = self.gt_file['initial_pose']['height']

    # self.even_total = int(self.total_images / 16)
    # self.even_total *= 16

    # # set trainnig to 90%
    # self.split_index = 1000 #int(self.even_total * 0.9)

    # self.imgs = np.zeros((self.total_images,240,320,3), dtype=float)
    # self.labels = np.zeros((self.total_images,7), dtype=float)

    # self.norm_imgs = np.zeros((self.total_images,240,320), dtype=float)
    # self.norm_labels = np.zeros((self.total_images,7), dtype=float)
   
  def read_imgs(self):
    """
    Read image files and labels into np array
    """

    for index in range(self.total_images):
      file_name = '{}.png'.format(index)
      full_path = os.path.join(self.imgs_folder,file_name)
      if self.width != 320 or self.height != 240:
        im = Image.open(full_path).resize((320,240))
      else:
        im = Image.open(full_path)

      self.imgs[index] = np.asarray(im)[:,:,:3] # discard the alpha channel

      t = np.array(self.gt_file[index]['translation'], dtype='f')
      r = np.array(self.gt_file[index]['rotation'], dtype='f')
      r = self.convert_rotation(r)
      self.labels[index] = np.concatenate((t,r))

  def show_image(self, index):
    index = int(index)
    if index < 0 or index > self.total_images:
      return
    
    file_name = '{}.png'.format(index)
    full_path = os.path.join(self.imgs_folder,file_name)
    im = Image.open(full_path)
    im_array = np.asarray(im)[:,:,:3]/255

    t = np.array(self.gt_file[index]['translation'], dtype='f')
    r = np.array(self.gt_file[index]['rotation'], dtype='f')
    r = self.convert_rotation(r)
    y_hat = np.concatenate((t,r))  

    print("inference label ", y_hat)

    self.Xi = torch.tensor(im_array, dtype=torch.float32)
    self.Xi = self.Xi.permute(2,0,1)
    self.Xi.unsqueeze(1)
    self.Xi = self.Xi.to(self.gpu, dtype=torch.float)

    y = self.cnn(self.Xi)
    y.squeeze(1)
    y = y.cpu()
    y_u = y.detach().numpy() * (self.max_labels - self.min_labels) + self.min_labels
    print("inference ", y_u[0])
    # Compute distance 3D
    diff = y_u - y_hat
    print("error ",diff[0])

    dist = np.power(diff[0][:3],2)
    dist = np.sum(dist)

    print("Cartesian distance: ", dist, " m")

    # Predicted quaternion norm
    norm = np.power(y_u[0][3:],2)
    norm = np.sum(norm)
    print("Predicted quaternion norm: ", norm)
    norm = np.power(y_hat[3:],2)
    norm = np.sum(norm)
    print("Label quaternion norm: ", norm)

    # print(im_array.shape)
    # im.show()

    # print(im_array[0,300:305,:])

    # plt.imshow(im_array, interpolation='nearest')
    # plt.show()

  def infere_from_image(self, image, show=False):
    h, w, c = image.shape     
    if w != 320 or h != 240:
      im = cv2.resize(image, dsize=(320,240), interpolation=cv2.INTER_CUBIC)
    else:
      im = image

    if show:
      cv2.imshow('input', im)
      cv2.waitKey(1)

    im_array = np.asarray(im)[:,:,:3]/255
    self.Xi = torch.tensor(im_array, dtype=torch.float32)
    self.Xi = self.Xi.permute(2,0,1)
    self.Xi.unsqueeze(1)
    self.Xi = self.Xi.to(self.gpu, dtype=torch.float)

    y = self.cnn(self.Xi)
    y.squeeze(1)
    y = y.cpu()
    y_u = y.detach().numpy() * (self.max_labels - self.min_labels) + self.min_labels
    
    # print("Translation ", np.round(y_u[0][:3],3))
    return np.round(y_u[0],6)



  def convert_rotation(self, flatten_matrix):
    matrix = np.reshape(flatten_matrix,(3,3))
    
    r = Rotations()
    r.from_matrix(matrix)
  
    return r.as_quat()

  def normalize_data(self):
    """
    Normalize data, shuffle and save into pytorch tensors.
    It also saves the maximum and minimum values as .npy
    """

    self.max_imgs = np.max(self.imgs[:,:,:,:],axis=0)
    self.min_imgs = np.min(self.imgs[:,:,:,:],axis=0)

    self.max_labels = np.max(self.labels[:,:], axis=0)
    self.min_labels = np.min(self.labels[:,:], axis=0)
    # print(self.max_imgs.shape)

    with open(self.norm_data, 'wb') as f:
       np.savez(f,self.max_imgs, self.min_imgs, self.max_labels,self.min_labels)

    # norm_imgs = (self.imgs - self.min_imgs)/(self.max_imgs - self.min_imgs)
    norm_imgs = self.imgs / 255
    norm_labels = (self.labels - self.min_labels)/(self.max_labels - self.min_labels)

    # Shuffle the data
    idxs = list(range(len(self.labels)))
    np.random.shuffle(idxs)

    norm_labels = norm_labels[idxs]
    norm_imgs = norm_imgs[idxs]

    # Train - Validation splitting
    self.Xt = norm_imgs[0:self.split_index,:,:,:]
    self.Yt = norm_labels[0:self.split_index,:]

    self.Xv = norm_imgs[self.split_index:,:,:,:]
    self.Yv = norm_labels[self.split_index:,:]

    self.Xt = torch.tensor(self.Xt, dtype=torch.float32)
    self.Xt = self.Xt.permute(0,3,1,2)
    self.Yt = torch.tensor(self.Yt, dtype=torch.float32)

    self.Xv = torch.tensor(self.Xv, dtype=torch.float32)
    self.Xv = self.Xv.permute(0,3,1,2)
    self.Yv = torch.tensor(self.Yv, dtype=torch.float32)

    # plt.imshow(norm_imgs[5], interpolation='nearest')
    # plt.show()

    
  def train_se3ish(self):
    # self.cnn = Se3Ish()
    self.cnn = ConvNet()

    self.gpu = torch.device("cuda:0")
    self.cnn = self.cnn.to(self.gpu)
    self.Xt = self.Xt.to(self.gpu, dtype=torch.float)
    self.Yt = self.Yt.to(self.gpu, dtype=torch.float)
    self.Xv = self.Xv.to(self.gpu, dtype=torch.float)
    self.Yv = self.Yv.to(self.gpu, dtype=torch.float)

    # Initialize optimizer after sending model to gpu
    opt = optim.Adam(self.cnn.parameters(), lr=0.0001)

    self.total_epochs = 60
    self.train_error = [0]*self.total_epochs
    self.val_error = [0]*self.total_epochs
    batch_size = 16

    for j in range(self.total_epochs):
      
      # Training data

      self.cnn.train(True) # Turn on dropout and gradient tracking

      for i in range(0,len(self.Yt),batch_size):
        # Take a batch
        x = self.Xt[i:i+batch_size,:,:,:]
        # print("x: ", x.shape)
        y_hat = self.Yt[i:i+batch_size]
        # print("y_hat ", y_hat)


        # Make prediction for the batch
        y = self.cnn(x)
        # print("y ",y)

        # Compute loss and gradients
        loss = self.cnn.loss(y, y_hat)

        # Zero grad on evey batch, to use gradients of the current batch
        opt.zero_grad()
        
        loss.mean().backward()

        # Adjust learning weights
        opt.step()

        self.train_error[j] += loss.mean().item()


      # Validation data

      self.cnn.eval() # Turn off dropout

      # Disable gradient computation and reduce memory consumption
      with torch.no_grad():
        for i in range(0,len(self.Yv),batch_size):
          # Take a batch
          vx = self.Xv[i:i+batch_size,:,:,:]
          vy_hat = self.Yv[i:i+batch_size]

          vy = self.cnn(vx)
          vloss = self.cnn.loss(vy, vy_hat)

          self.val_error[j] += vloss.mean().item()
      
      print('{} Train: {} - Validation: {}'.format(j, self.train_error[j], self.val_error[j]))


  def plot_losses(self):
    epochs = np.arange(0,self.total_epochs,1)
    plt.plot(epochs, self.train_error, label='train')
    plt.plot(epochs, self.val_error, label='val')
    plt.legend()
    plt.show()

  def save_model(self):
    torch.save(self.cnn.state_dict(), self.model_path)

  def open_model(self):
    """
    self.cnn = ConvNet()

    self.cnn.load_state_dict(torch.load(self.model_path))
    
    self.cnn.eval()

    self.gpu = torch.device("cuda:0")
    
    self.cnn = self.cnn.to(self.gpu)
    """
    self.cnn = ConvNet()
    self.cnn.load_state_dict(torch.load(self.model_path))
    self.cnn.eval()

    self.gpu = torch.device("cuda:0")
    self.cnn = self.cnn.to(self.gpu)

  def open_normalization(self):
    with open(self.norm_data, 'rb') as f:
      npzfile = np.load(f)
      max_imgs = npzfile.files[0]
      min_imgs = npzfile.files[1]
      max_labels = npzfile.files[2]
      min_labels = npzfile.files[3]

      self.max_imgs = npzfile[max_imgs]
      self.min_imgs = npzfile[min_imgs]
      self.max_labels = npzfile[max_labels]
      self.min_labels = npzfile[min_labels]

  def run(self):
    print("Step 1: Reading dataset")
    self.read_imgs()
    print("Step 2: Normalize data")
    self.normalize_data()
    print("Step 3: Training loop")
    self.train_se3ish()
    print("Step 4: Save model")
    self.save_model()
    print("Step 5: Plot loss")
    self.plot_losses()
    print("Step 6: Infere")
    self.show_image(0)

  def infere(self):
    print("Step 1: Loading model")
    self.open_model()
    print("Step 2: Open normalization data")
    self.open_normalization()
    print("Step 3: Open image and infere")
    self.show_image(900)

  def metrics(self):
      
      self.cnn.eval() # Turn off dropout
      values = np.empty((0,7), float)

      batch_size = 16

      # Disable gradient computation and reduce memory consumption
      with torch.no_grad():
        for i in range(0,len(self.Yv),batch_size):
          # Take a batch
          vx = self.Xv[i:i+batch_size,:,:,:]
          vy_hat = self.Yv[i:i+batch_size]

          vy_hat_u = vy_hat * (self.max_labels - self.min_labels) + self.min_labels

          vy = self.cnn(vx)

          vy_u = vy * (self.max_labels - self.min_labels) + self.min_labels
          # vloss = self.cnn.loss(vy, vy_hat)

          np.append(values, np.array((vy_u - vy_hat_u)), axis=0)

  

          # self.val_error[j] += vloss.mean().item()

if __name__=="__main__":
  data_folder = './cube_3_dof_v2'
  # model = Trainer(data_folder).show_image(30)

  # Trainer(data_folder).run()
  Trainer(data_folder).infere()
