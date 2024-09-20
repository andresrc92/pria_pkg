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
from pria.se3ish import *
from pria.convnet import *
from pria.rotations import *
from pria.image_to_coarse import *
from pria.image_to_coarse_simple import *
import yaml
import sys
import time


class Trainer:
    def __init__(self, data_folder, model_key, epochs):
        model_key = int(model_key)
        if model_key == 1:
            model_name = "convnet_"
            # self.cnn = ConvNet()
        elif model_key == 2:
            model_name = "coarsenet_"
            self.cnn = NetworkCoarse()
        elif model_key == 3:
            model_name = "tracknet_"
            # self.cnn = Se3Ish()
        elif model_key == 4:
            model_name = "simplenet_"
            # self.cnn = NetworkCoarseSimple()

        self.data_folder = data_folder
        self.imgs_folder = os.path.join(data_folder, "imgs")
        self.gt_path = os.path.join(data_folder, "gt.yaml")
        self.model_folder = os.path.join(data_folder, "{}{}".format(model_name, epochs))
        self.norm_data = os.path.join(self.model_folder, "norm_param.npy")
        self.model_path = os.path.join(self.model_folder, "state_dict.pt")
        self.loss_data = os.path.join(self.model_folder, "loss.npy")
        self.plot_path = os.path.join(self.model_folder, "loss.png")

        self.total_epochs = int(epochs)
        self.batch_size = 32

        self.parent = os.getcwd()
        self.path = os.path.join(self.parent, self.model_folder)

        try:
            os.mkdir(self.path)
        except FileExistsError:
            pass

    def gt_converter(self):
        with open(self.gt_path, "r") as file:
            self.gt_file = yaml.safe_load(file)

        self.next_gt_file = {}
        self.next_gt_file.update({"initial_pose": self.gt_file["initial_pose"]})

        r = Rotations()
        for i in range(self.gt_file["initial_pose"]["total_images"]):
            
            m = np.array(self.gt_file[i]["rotation"], dtype="f")

            r = self.convert_rotation(m)

            self.next_gt_file.update(
                {
                    i: {
                    "translation": self.gt_file[i]["translation"],
                    "rotation": r.tolist()
                    }
                }
            )

        next_gt_path = '{}{}'.format(self.gt_path,'2')
        with open(next_gt_path, 'w') as file:
            yaml.dump(self.next_gt_file, file,  default_flow_style=False)

    def get_initial_pose(self):
        t = np.array(self.gt_file["initial_pose"]["translation"], dtype="f")
        r = np.array(self.gt_file["initial_pose"]["rotation"], dtype="f")
        pose = np.concatenate((t, r))
        return pose

    def read_meta_data(self):
        with open(self.gt_path, "r") as file:
            self.gt_file = yaml.safe_load(file)

        self.total_images = self.gt_file["initial_pose"]["total_images"]
        self.total_images = int(self.total_images)
        self.width = self.gt_file["initial_pose"]["width"]
        self.height = self.gt_file["initial_pose"]["height"]

        self.train_index = self.total_images * 80 // 100
        self.val_size = self.total_images * 15 // 100
        self.test_size = self.total_images * 5 // 100
        self.val_index = self.val_size + self.train_index

        # print("Train index ", self.train_index)
        # print("Val index ", self.val_index)

        self.imgs = np.zeros((self.total_images, 240, 320, 3), dtype=float)
        self.labels = np.zeros((self.total_images, 7), dtype=float)
    
        gt_labels = np.zeros((self.total_images, 7), dtype=float)
        for i in range(self.total_images):
            t = np.array(self.gt_file[i]["translation"], dtype="f")
            r = np.array(self.gt_file[i]["rotation"], dtype="f")
            gt_labels[i] = np.concatenate((t, r))

        gt_labels_ = np.array(gt_labels)
        self.gt_min = np.min(gt_labels_, axis=0)
        # self.gt_min[-1] = 0.999999999998 
        self.gt_max = np.max(gt_labels_, axis=0)

        print("Total images", self.total_images)
        # print("Ground truth max", self.gt_max)
        # print("Ground truth min", self.gt_min)

        # self.norm_imgs = np.zeros((self.total_images,240,320,3), dtype=float)
        # self.norm_labels = np.zeros((self.total_images,7), dtype=float)
        return self.gt_file["initial_pose"]["translation"][2]

    def read_imgs(self):
        """
        Read image files and labels into np array
        """

        for index in range(self.total_images):
            file_name = "{}.png".format(index)
            full_path = os.path.join(self.imgs_folder, file_name)
            if self.width != 320 or self.height != 240:
                im = Image.open(full_path).resize((320, 240))
            else:
                im = Image.open(full_path)

            self.imgs[index] = np.asarray(im)[:, :, :3]  # discard the alpha channel

            t = np.array(self.gt_file[index]["translation"], dtype="f")
            r = np.array(self.gt_file[index]["rotation"], dtype="f")
            self.labels[index] = np.concatenate((t, r))

    def read_batch_imgs(self, indices):
        """
        Read and Normalize image files and labels into np array
        """

        imgs = np.zeros((len(indices), 240, 320, 3), dtype=float)
        labels = np.zeros((len(indices), 7), dtype=float)

        for i, index in enumerate(indices):
            file_name = "{}.png".format(index)
            full_path = os.path.join(self.imgs_folder, file_name)
            if self.width != 320 or self.height != 240:
                im = Image.open(full_path).resize((320, 240))
            else:
                im = Image.open(full_path)

            imgs[i] = np.asarray(im)[:, :, :3]  # discard the alpha channel
        

            t = np.array(self.gt_file[index]["translation"], dtype="f")
            r = np.array(self.gt_file[index]["rotation"], dtype="f")
            labels[i] = np.concatenate((t, r))
            
        # Normalize batch
        imgs_norm = imgs/255
        labels_norm = (labels - self.gt_min) / (self.gt_max - self.gt_min)

        return imgs_norm, labels_norm

    def show_image(self, index):
        index = int(index)
        if index < 0 or index > self.total_images:
            return
            
        error = []
        norm_ = []
        cycle_time = []
        max_dist = -np.inf
        min_dist = np.inf     
        for index in range(self.total_images):

            file_name = "{}.png".format(index)
            full_path = os.path.join(self.imgs_folder, file_name)
            im = Image.open(full_path)
            im_array = np.asarray(im)[:, :, :3] / 255

            t = np.array(self.gt_file[index]["translation"], dtype="f")
            r = np.array(self.gt_file[index]["rotation"], dtype="f")
            y_hat = np.concatenate((t, r))

            # print("inference label ", y_hat)
            start = time.time()

            self.Xi = torch.tensor(im_array, dtype=torch.float32)
            self.Xi = self.Xi.permute(2, 0, 1)
            self.Xi = self.Xi.unsqueeze(0)
            self.Xi = self.Xi.to(self.gpu, dtype=torch.float)

            y = self.cnn(self.Xi)
            y.squeeze(0)
            y = y.cpu()
            y_u = y.detach().numpy() * (self.max_labels - self.min_labels) + self.min_labels

            stop = time.time()
            cycle_time.append(stop - start)


            # print("inference ", y_u[0])
            # Compute distance 3D
            diff = y_u - y_hat
            # print("error ", diff[0])

            dist = np.power(diff[0][:3], 2)
            dist = np.sum(dist)

            if dist > max_dist:
                max_dist = dist

            if dist < min_dist:
                min_dist = dist

            # print("Cartesian distance: ", dist, " m")

            # print("Predicted quaternion norm: ", norm)
            norm = np.power(y_hat[3:], 2)
            norm = np.sum(norm)
            # Predicted quaternion norm
            norm = np.power(y_u[0][3:], 2)
            norm = np.sum(norm)
            # print("Label quaternion norm: ", norm)
            error.append(dist)
            norm_.append(norm)

        print(np.mean(error), max_dist, min_dist, np.mean(norm_), 1/np.mean(cycle_time))
        # print(im_array.shape)
        # im.show()

        # print(im_array[0,300:305,:])

        # plt.imshow(im_array, interpolation='nearest')
        # plt.show()

    def show_single_image(self, index):
        index = int(index)
        if index < 0 or index > self.total_images:
            return
            
        file_name = "{}.png".format(index)
        full_path = os.path.join(self.imgs_folder, file_name)
        print(full_path)
        im = Image.open(full_path)
        im_array = np.asarray(im)[:, :, :3]/ 255

        # cv2.imshow("input2", im_array)
        # cv2.waitKey(1)

        t = np.array(self.gt_file[index]["translation"], dtype="f")
        r = np.array(self.gt_file[index]["rotation"], dtype="f")
        y_hat = np.concatenate((t, r))

        print("inference label ", y_hat)
        start = time.time()

        y_u = self.infer_from_image(im_array, True)

        stop = time.time()
        cycle_time = stop - start


        print("inference ", y_u)
        # Compute distance 3D
        diff = y_u - y_hat
        # print("error ", diff[0])

        dist = np.power(diff[:3], 2)
        dist = np.sum(dist)

        # if dist > max_dist:
        #     max_dist = dist

        # if dist < min_dist:
        #     min_dist = dist

        # print("Cartesian distance: ", dist, " m")

        # print("Predicted quaternion norm: ", norm)
        norm = np.power(y_hat[3:], 2)
        norm = np.sum(norm)
        # Predicted quaternion norm
        norm = np.power(y_u[3:], 2)
        norm = np.sum(norm)
        
        print("dist: ", dist, "norm: ", norm, "freq: ", 1/cycle_time)
        # print(im_array.shape)
        # im.show()

        # print(im_array[0,300:305,:])

        labels = ['x','y','z','qx','qy','qz','qw']
        plt.plot(labels, (y_u - y_hat), marker='o')

        # plt.imshow(im_array, interpolation='nearest')
        plt.show()

    def infer_from_image(self, image, show=False):
        
        im_array = image

        # print(im_array.shape)

        if show:
            plt.imshow(im_array, interpolation='nearest')
            plt.show()

        self.Xi = torch.tensor(im_array, dtype=torch.float32)
        self.Xi = self.Xi.permute(2, 0, 1)
        self.Xi = self.Xi.unsqueeze(0)
        self.Xi = self.Xi.to(self.gpu, dtype=torch.float)

        y = self.cnn(self.Xi)
        y.squeeze(0)
        y = y.cpu()
        y_u = y.detach().numpy()[0] * (self.max_labels - self.min_labels) + self.min_labels

        # print(np.round(y.detach().numpy()[:3], 2))
        # print("Prediction ", np.round(y_u,3))

        # Normalize quaternion output
        mod_ = np.power(y_u[3:], 2)
        mod_ = np.sum(mod_)
        mod_ = np.sqrt(mod_)

        y_u[3:] /= mod_

        return np.round(y_u, 6)

    def convert_rotation(self, flatten_matrix):
        matrix = np.reshape(flatten_matrix, (3, 3))

        r = Rotations()
        r.from_matrix(matrix)

        return r.as_quat()

    def normalize_data(self):
        """
        Normalize data, shuffle and save into pytorch tensors.
        It also saves the maximum and minimum values as .npy
        """
        # Shuffle the data
        idxs = list(range(len(self.labels)))
        np.random.shuffle(idxs)

        self.shuffled_imgs = self.imgs#[idxs]
        self.shuffled_labels = self.labels#[idxs]

        self.max_imgs = np.max(self.shuffled_imgs[:self.train_index, :, :, :], axis=0)
        self.min_imgs = np.min(self.shuffled_imgs[:self.train_index, :, :, :], axis=0)

        self.max_labels = np.max(self.shuffled_labels[:self.train_index, :], axis=0)
        self.min_labels = np.min(self.shuffled_labels[:self.train_index, :], axis=0)
        # print(self.max_imgs.shape)

        with open(self.norm_data, "wb") as f:
            np.savez(f, self.max_imgs, self.min_imgs, self.max_labels, self.min_labels)

        # norm_imgs = (self.imgs - self.min_imgs)/(self.max_imgs - self.min_imgs)
        norm_imgs = self.shuffled_imgs / 255
        norm_labels = (self.shuffled_labels - self.min_labels) / (
            self.max_labels - self.min_labels
        )

        # Train - Validation - Testing (m) splitting
        self.Xt = norm_imgs[0 : self.train_index, :, :, :]
        self.Yt = norm_labels[0 : self.train_index, :]

        self.Xv = norm_imgs[self.train_index : self.val_index, :, :, :]
        self.Yv = norm_labels[self.train_index : self.val_index, :]

        self.Xm = norm_imgs[self.val_index :, :, :, :]
        self.Ym = norm_labels[self.val_index :, :]

        # Convert to Pytorch tensors
        self.Xt = torch.tensor(self.Xt, dtype=torch.float32)
        self.Xt = self.Xt.permute(0, 3, 1, 2)
        self.Yt = torch.tensor(self.Yt, dtype=torch.float32)

        self.Xv = torch.tensor(self.Xv, dtype=torch.float32)
        self.Xv = self.Xv.permute(0, 3, 1, 2)
        self.Yv = torch.tensor(self.Yv, dtype=torch.float32)

        self.Xm = torch.tensor(self.Xm, dtype=torch.float32)
        self.Xm = self.Xm.permute(0, 3, 1, 2)
        self.Ym = torch.tensor(self.Ym, dtype=torch.float32)

        # plt.imshow(norm_imgs[5], interpolation='nearest')
        # plt.show()

    def train(self):

        self.gpu = torch.device("cuda:0")
        self.cnn = self.cnn.to(self.gpu)
        self.Xt = self.Xt.to(self.gpu, dtype=torch.float)
        self.Yt = self.Yt.to(self.gpu, dtype=torch.float)
        self.Xv = self.Xv.to(self.gpu, dtype=torch.float)
        self.Yv = self.Yv.to(self.gpu, dtype=torch.float)

        # Initialize optimizer after sending model to gpu
        opt = optim.Adam(self.cnn.parameters(), lr=0.0001) #, weight_decay=0.001)

        self.train_error = [0] * self.total_epochs
        self.val_error = [0] * self.total_epochs

        for j in range(self.total_epochs):

            # Training data

            self.cnn.train(True)  # Turn on dropout and gradient tracking

            for i in range(0, len(self.Yt), self.batch_size):
                # Take a batch
                x = self.Xt[i : i + self.batch_size, :, :, :]
                # print("x: ", x.shape)
                y_hat = self.Yt[i : i + self.batch_size]
                # print("y_hat ", y_hat)

                # Make prediction for the batch
                y = self.cnn(x)
                # print("y ",y)

                # Compute loss and gradients
                # loss = self.cnn.loss(y, y_hat)
                loss = self.cnn.compute_loss(y, y_hat)

                # Zero grad on evey batch, to use gradients of the current batch
                opt.zero_grad()

                # loss.mean().backward()
                loss.backward()

                # Adjust learning weights
                opt.step()

                self.train_error[j] += loss.item()

            # Validation data
            self.cnn.eval()  # Turn off dropout

            # Disable gradient computation and reduce memory consumption
            with torch.no_grad():
                for i in range(0, len(self.Yv), self.batch_size):
                    # Take a batch
                    vx = self.Xv[i : i + self.batch_size, :, :, :]
                    vy_hat = self.Yv[i : i + self.batch_size]

                    vy = self.cnn(vx)
                    vloss = self.cnn.compute_loss(vy, vy_hat)

                    self.val_error[j] += vloss.item()

            print(
                "{} Train: {} - Validation: {}".format(
                    j, self.train_error[j], self.val_error[j]
                )
            )

    def plot_losses(self):
        epochs = np.arange(0, self.total_epochs, 1)
        plt.plot(epochs, self.train_error, label="train")
        plt.plot(epochs, self.val_error, label="val")
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
        print(self.model_path)
        self.cnn.load_state_dict(torch.load(self.model_path))

        self.gpu = torch.device("cuda:0")
        self.cnn = self.cnn.to(self.gpu)

        self.cnn.eval()
        
    def open_normalization(self):
        with open(self.norm_data, "rb") as f:
            npzfile = np.load(f)

            max_imgs = npzfile.files[0]
            min_imgs = npzfile.files[1]
            max_labels = npzfile.files[2]
            min_labels = npzfile.files[3]

            self.max_imgs = npzfile[max_imgs]
            self.min_imgs = npzfile[min_imgs]
            self.max_labels = npzfile[max_labels]
            self.min_labels = npzfile[min_labels]

            print("Labels min ", self.min_labels)
            print("Labels max ", self.max_labels)

    def train_batch(self):

        self.gpu = torch.device("cuda:0")
        self.cnn = self.cnn.to(self.gpu)
        # self.Xt = self.Xt.to(self.gpu, dtype=torch.float)
        # self.Yt = self.Yt.to(self.gpu, dtype=torch.float)
        # self.Xv = self.Xv.to(self.gpu, dtype=torch.float)
        # self.Yv = self.Yv.to(self.gpu, dtype=torch.float)

        # Initialize optimizer after sending model to gpu
        opt = optim.Adam(self.cnn.parameters(), lr=0.001)#, weight_decay=0.001)

        self.train_error = [0] * self.total_epochs
        self.val_error = [0] * self.total_epochs
        self.train_dist = []
        self.val_dist = []

        # Randomize index
        self.random_i = list(range(self.total_images))
        np.random.shuffle(self.random_i)

        min_val_error = np.inf
        patience = 15
        lr_patience = 6
        bad_epoch_count = 0
        num_bad_epochs_since_lr_change = 0

        with open(self.norm_data, "wb") as f:
            np.savez(f, self.gt_max, self.gt_min)

        print("Train batch count: ", self.train_index / self.batch_size)
        print("Val batch count: ", self.val_size / self.batch_size)
        print("Test val count: ", self.test_size / self.batch_size)

        for j in range(self.total_epochs):

            # Training data

            self.cnn.train(True)  # Turn on dropout and gradient tracking

            for i in range(0, self.train_index, self.batch_size):
                # Open and normalize current batch 
                Xt, Yt = self.read_batch_imgs(self.random_i[i:i+self.batch_size])

                # Convert to PyTorch tensor and send to GPU
                Xt = torch.tensor(Xt, dtype=torch.float32)
                Xt = Xt.permute(0, 3, 1, 2)
                Yt = torch.tensor(Yt, dtype=torch.float32)

                Xt = Xt.to(self.gpu, dtype=torch.float)
                Yt = Yt.to(self.gpu, dtype=torch.float)

                # Make prediction for the batch
                Y = self.cnn(Xt)

                # Compute loss and gradients
                loss = self.cnn.compute_loss(Y, Yt)

                # Zero grad on evey batch, to use gradients of the current batch
                opt.zero_grad()

                # Adjust learning weights
                loss.backward()
                opt.step()

                self.train_error[j] += loss.item()

                # Compute error
                self.train_dist.append(self.cartesian_dist(Y,Yt))

            self.train_error[j] /= int(self.train_index / self.batch_size)

            # Validation data
            self.cnn.eval()  # Turn off dropout

            # Disable gradient computation and reduce memory consumption
            with torch.no_grad():
                for i in range(0, self.val_size, self.batch_size):
                    # Take a batch
                    i_0 = i + self.train_index
                    i_1 = i_0 + self.batch_size

                    Xv, Yv = self.read_batch_imgs(self.random_i[i_0:i_1])

                    # Convert to PyTorch tensor and send to GPU
                    Xv = torch.tensor(Xv, dtype=torch.float32)
                    Xv = Xv.permute(0, 3, 1, 2)
                    Yv = torch.tensor(Yv, dtype=torch.float32)

                    Xv = Xv.to(self.gpu, dtype=torch.float)
                    Yv = Yv.to(self.gpu, dtype=torch.float)

                    Y = self.cnn(Xv)
                    vloss = self.cnn.compute_loss(Y, Yv)

                    self.val_error[j] += vloss.item()

                    # Compute error
                    self.val_dist.append(self.cartesian_dist(Y,Yv))

            self.val_error[j] /= int(self.val_size / self.batch_size)

            # Adjust LR
            if num_bad_epochs_since_lr_change > lr_patience:
                for p in opt.param_groups:
                    old_lr = p['lr']
                    new_lr = 0.5 * old_lr
                    p['lr'] = new_lr
                print('Dropping learning rate to ' + str(new_lr),end='\r')
                num_bad_epochs_since_lr_change = 0

            # Early stopping
            if self.val_error[j] < 0.99 * min_val_error:
                min_val_error = self.val_error[j]
                bad_epoch_count = 0
                num_bad_epochs_since_lr_change = 0
                print(
                    "{} Train: {} - *Validation*: {}".format(
                        j, self.train_error[j], self.val_error[j]
                    ),end='\r'
                )
                self.save_model()
            else:
                print(
                    "{} Train: {} - Validation: {}".format(
                        j, self.train_error[j], self.val_error[j]
                    ),end='\r'
                )
                bad_epoch_count += 1
                num_bad_epochs_since_lr_change += 1

            self.final_epochs = j
            if bad_epoch_count > patience:
                print()
                print("Early stopping at epoch ", j)
                break;

    def cartesian_dist(self, prediction, groundtruth):
        p = prediction.detach().cpu().tolist()
        p *= (self.gt_max - self.gt_min)
        p += self.gt_min

        gt = groundtruth.detach().cpu().tolist()
        gt *= (self.gt_max - self.gt_min)
        gt += self.gt_min

        dist = np.fabs(p[:,:3] - gt[:,:3])
        dist = np.power(dist, 2)
        dist = np.sum(dist, axis=1)
        dist = np.sqrt(dist)

        r = Rotations()
        euler_array = []
        for i in range(len(p)):
            r.from_array(p[i,3:])
            euler_p = np.array(r.as_euler())

            r.from_array(gt[i,3:])
            euler_gt = np.array(r.as_euler())

            euler_array.append(np.fabs(euler_p - euler_gt))

        angle_delta = np.sum(euler_array, axis=1)
        angle_delta = np.mean(angle_delta)

        return [dist.mean(), angle_delta.mean()]

    def plot_losses(self):
        with open(self.loss_data, "wb") as f:
            np.savez(f, self.train_error[:self.final_epochs+1], self.val_error[:self.final_epochs+1],self.train_dist,self.val_dist)

        epochs = np.arange(0, self.final_epochs+1, 1)
        plt.plot(epochs, self.train_error[:self.final_epochs+1], label="train")
        plt.plot(epochs, self.val_error[:self.final_epochs+1], label="val")
        plt.legend()
        # plt.show()
        plt.savefig(self.plot_path)
        
    def save_model(self):
        torch.save(self.cnn.state_dict(), self.model_path)

    def open_batch_normalization(self):
        with open(self.gt_path, "r") as file:
            self.gt_file = yaml.safe_load(file)

        with open(self.norm_data, "rb") as f:
            npzfile = np.load(f)

            max_labels = npzfile.files[0]
            min_labels = npzfile.files[1]

            self.max_labels = npzfile[max_labels]
            self.min_labels = npzfile[min_labels]

            print("Labels min ", self.min_labels)
            print("Labels max ", self.max_labels)

    def test_the_model(self):
        return

    def run(self):
        print("Step 1: Reading dataset")
        self.read_meta_data()
        self.read_imgs()
        print("Step 2: Normalize data")
        self.normalize_data()
        print("Step 3: Training loop")
        self.train()
        print("Step 5: Plot loss")
        self.plot_losses()
        print("Step 6: Infer")
        self.metrics()

    def run_batch(self):
        print("Step 1: Reading metadata")
        self.read_meta_data()
        print("Step 2: Training loop")
        self.train_batch()
        print("Step 4: Save model")
        self.save_model()
        print("Step 5: Plot loss")
        self.plot_losses()
        print("Step 6: Infer")
        self.batch_metrics()

        # for i in range(5):
        #     os.system('spd-say "your program has finished"')
        #     time.sleep(3)

    def infer(self, index):
        print("Step 1: Loading model")
        self.read_meta_data()
        self.open_model()
        print("Step 2: Open normalization data")
        self.open_batch_normalization()
        print("Step 3: Open image and infer")
        self.show_single_image(index)

    def metrics(self):

        self.cnn.eval()  # Turn off dropout
        dist_array = []
        norm_array = []

        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            self.Xv = self.Xv.to(self.gpu, dtype=torch.float)
            # self.Ym = self.Ym.to(self.gpu, dtype=torch.float)
            self.Yv = self.Yv.cpu()

            for i in range(0, len(self.Yv), self.batch_size):
                # Take a batch
                test_x = self.Xv[i : i + self.batch_size, :, :, :]
                test_y_hat = self.Yv[i : i + self.batch_size, :]

                test_y_hat_u = (
                    test_y_hat * (self.max_labels - self.min_labels) + self.min_labels
                )

                test_y = self.cnn(test_x)

                test_y = test_y.cpu()

                vy_u = (
                    test_y.detach().numpy() * (self.max_labels - self.min_labels)
                    + self.min_labels
                )
                # vloss = self.cnn.loss(vy, vy_hat)

                pred_ = test_y_hat_u.detach().numpy()
                pred_cart = pred_[:, :3]
                label_cart = vy_u[:, :3]

                pred_quat = pred_[:,3:]
                label_quat = vy_u[:,3:]

                dist = label_cart - pred_cart
                dist = np.power(dist, 2)
                dist = np.sum(dist, axis=1)
                dist = np.sqrt(dist)

                norm = label_quat - pred_quat
                norm = np.power(pred_quat, 2)
                norm = np.sum(norm, axis=1)
                norm = np.sqrt(norm)
                # print(dist)
                # print(label_cart.shape)

                # np.append(dist_array, [[np.mean(dist)]], axis=0)
                dist_array.append(np.mean(dist))
                norm_array.append(np.mean(norm))
            

            # print(dist_array)
            print("mean distance", np.mean(dist_array))
            print("mean quat norm", np.mean(norm_array))

        with open(self.loss_data, "wb") as f:
            np.savez(f, self.train_error, self.val_error, np.mean(dist_array))

    def batch_metrics(self):

        test_dist = []

        self.cnn.eval()  # Turn off dropout
        
        # Disable gradient computation and reduce memory consumption
        with torch.no_grad():
            for i in range(self.val_index, self.total_images, self.batch_size):
                # Take a batch
                i_0 = i
                i_1 = i_0 + self.batch_size

                Xt, Yt = self.read_batch_imgs(self.random_i[i_0:i_1])

                # Convert to PyTorch tensor and send to GPU
                Xt = torch.tensor(Xt, dtype=torch.float32)
                Xt = Xt.permute(0, 3, 1, 2)
                Yt = torch.tensor(Yt, dtype=torch.float32)

                Xt = Xt.to(self.gpu, dtype=torch.float)
                Yt = Yt.to(self.gpu, dtype=torch.float)

                Y = self.cnn(Xt)
                # vloss = self.cnn.compute_loss(Y, Yt)

                # self.val_error[j] += vloss.item()

                # Compute error
                test_dist.append(self.cartesian_dist(Y,Yt))
            
            test_dist_mean = np.mean(test_dist, axis=0)
            print("Distancia en test set ", test_dist_mean)


if __name__ == "__main__":
    # print(len(sys.argv))
    if len(sys.argv) >= 4:
        data_folder = sys.argv[1]
        # print(data_folder)
        model = sys.argv[2]
        # print(model)
        epochs = sys.argv[3]
        # print(epochs)

        options = ['', '_twist', '_cone', '_cone_twist']

        if len(sys.argv) == 5:
            Trainer(data_folder, model, epochs).infer(int(sys.argv[4]))
        else:
            Trainer(data_folder, model, epochs).run_batch()
            # for i in range(4):
            #     folder = '{}{}'.format(data_folder, options[i])
            #     print("////////////////////////////////")
            #     print(i, " " + folder)
            #     print("////////////////////////////////")
            #     Trainer(folder, model, epochs).run_batch()

        # Trainer(data_folder, model, epochs).gt_converter()
