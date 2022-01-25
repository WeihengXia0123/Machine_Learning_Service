import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader

import glob
import os
import scipy.io

import utils
import models

batch_size = 32

"""
1. Load data from the dataset folder
"""
DATA_PATH = "../dataset/"
mat_files = glob.glob(DATA_PATH + "**/*.mat", recursive=True)

train_data_list = []
train_label_list = []
for mat_file in mat_files:
    mat = scipy.io.loadmat(mat_file)
    data = mat['X']
    label = mat['y']
    train_data_list.append(data)
    train_label_list.append(label)

train_data_list = np.array(train_data_list)
train_label_list = np.array(train_label_list)

# replace labels of `10`` with `0`
train_label_list[train_label_list==10] = 0
train_label_list = train_label_list.astype(np.long)

train_data_stacked = np.concatenate(train_data_list, axis=3)
train_data_stacked = np.transpose(train_data_stacked, (3,0,1,2)) # (b, h, w, c)
train_label_stacked = np.concatenate(train_label_list, axis=0)
train_label_stacked = train_label_stacked.reshape(train_label_stacked.shape[0])

# convert from int [0,255] to float [0,1]
train_data_stacked = train_data_stacked.astype(np.float32)/255.

train_data = torch.from_numpy(train_data_stacked)
train_label = torch.from_numpy(train_label_stacked)

dataset = TensorDataset(train_data, train_label)
val_size = int(len(dataset)/10)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size*2, num_workers=4, pin_memory=True)

# for images, _ in train_loader:
#     print('images.shape: ', images.shape)
#     print('images.dtype: ', images.dtype)
#     plt.figure(figsize=(16,8))
#     plt.axis('off')
#     images = np.transpose(images, (0,3,1,2)) # (b,c,h,w)
#     plt.imshow(make_grid(images, nrow=4).permute((1,2,0)))
#     plt.show()
#     break

"""
2. Training and Validation
"""
input_size = 3072
num_classes = 10
device = utils.get_default_device() # device: CPU or GPU
print("device: ", device)

model = models.model(input_size, out_size=num_classes)
utils.to_device(model, device) # move model into the same CPU/GPU device as dataloaders
train_dataLoader = utils.DeviceDataLoader(train_loader, device) # wrap dataloder into device_dataloader
valid_dataLoader = utils.DeviceDataLoader(val_loader, device) # wrap dataloder into device_dataloader

history = [utils.evaluate(model, valid_dataLoader)]
print(history)
history += utils.fit(1000,1e-2, model, train_dataLoader, valid_dataLoader)
print(history)

"""
3. Inference 
"""