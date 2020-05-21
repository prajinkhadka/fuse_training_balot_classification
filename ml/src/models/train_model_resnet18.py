import sys
sys.path.insert(1, '/home/prajin/Desktop/balot_final/fuse_training_balot_classification/ml/src/data')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
train_on_gpu = torch.cuda.is_available()
import torchvision.models as models
from tqdm import tqdm_notebook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Data Loaders, Imprted from loading_data.py

from loading_data import loading_datas
train_data, train_loader, validation_data, validation_loader, test_data, test_loader = loading_datas()


def train_resnet18():
    loss_fn_c = nn.CrossEntropyLoss()
    max_epochs =2
    model_c = models.resnet18(pretrained = True)
    model_c.fc = nn.Linear(512,48,bias=True)
    if deivce == True:
        model_c.to(device)

    opt_c = optim.Adam(model_c.parameters())
    loss_arr = []
n_iters = np.ceil(50000/ 64)
i= 0
for epoch in range(max_cpochs):
  for i, data in enumerate(train_loader, 0):
    inputs, labels = data
    if device == True:
        inputs, labels = inputs.to(device), labels.to(device)

    c_out = model_c(inputs)
    opt_c.zero_grad()
    loss =  loss_fn_c(c_out, labels)
    loss.backward()
    opt_c.step()
    del inputs, labels
    loss_arr.append(loss.item())


    if i % 100 == 0:
        print("Iterations %d/%d loss = %0.2f"%(i, n_iters, loss.item()))
        
  print(loss.item()) 


# Saving The Model Architecure and Weights.

checkpoint = {'model': model_c,
          'state_dict': model_c.state_dict(),
          'optimizer' : opt_c.state_dict()}

torch.save(checkpoint, 'checkpoint_resnet.pth')
  
    


