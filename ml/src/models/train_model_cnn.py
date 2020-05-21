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

def train_cnn():
    class Net(nn.Module):
        def __init__(self):
            super(Net, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1) 
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1) 
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1) 
            self.pool = nn.MaxPool2d(2, 2)
            self.fc1 = nn.Linear(64 * 56 * 56, 500)
            self.fc2 = nn.Linear(500, 48)
            self.dropout = nn.Dropout(0.25)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            #print(x.shape)
            x = x.view(-1, 64 * 56 * 56)
            x = self.dropout(x)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

model = Net()
if train_on_gpu:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

n_epochs = 10

valid_loss_min = np.Inf

for epoch in range(1, n_epochs+1):
    train_loss = 0.0
    valid_loss = 0.0
  
    model.train()
    for data, target in train_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
    model.eval()
    for data, target in validation_loader:
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
    
    train_loss = train_loss/len(train_loader.sampler)
    valid_loss = valid_loss/len(validation_loader.sampler)
        
    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        epoch, train_loss, valid_loss))
    
    if valid_loss <= valid_loss_min:
        valid_loss_min = valid_loss




checkpoint = {'model': model,
          'state_dict': model.state_dict(),
          'optimizer' : opttimizer.state_dict()}

torch.save(checkpoint, 'checkpoint_cnn.pth')
