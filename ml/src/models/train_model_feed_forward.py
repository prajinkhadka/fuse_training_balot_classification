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

def train_feed_forward():
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(12288, 6000)
            self.fc2 = nn.Linear(6000, 1200)
            self.fc3 = nn.Linear(1200, 256)
            self.fc4 = nn.Linear(256, 128)
            self.fc5 = nn.Linear(128, 64)
            self.fc6 = nn.Linear(64,48)

        def forward(self,x):
            x = x.view(x.shape[0], -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = F.relu(self.fc3(x))
            x = F.relu(self.fc4(x))
            x = F.relu(self.fc5(x))
            x = F.log_softmax(self.fc6(x), dim =1)

            return x
    model = Classifier()
    # Negative log lokelihood loss, as are are using softmax in the last layers.
    criterian = nn.NLLLoss()
    # using adam optimzier
    optimizer = optim.Adam(model.parameters())

    # Training Feed Forward.
    epochs = 12
    train_loss = 0.0
    valid_loss = 0.0
    for e in range(epochs):
    running_loss = 0 
    for images,labels in train_loader:
        images= images.view(images.shape[0], -1)
        optimizer.zero_grad()
        output = model.forward(images)
        #print(output)
        loss = criterian(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        print(f"training loss:{running_loss /len(train_loader)}")
    model.eval()
    for data, target in validation_loader:
        running_loss_v = 0
        data = data.view(data.shape[0], -1)
        output = model.forward(data)
            # calculate the batch loss
        loss = criterian(output, target)
        running_loss_v += loss.item()
    else:
        print(f"Validiation loss:{running_loss_v /len(validation_loader)}")

checkpoint = {'model': model,
          'state_dict': model.state_dict(),
          'optimizer' : opttimizer.state_dict()}

torch.save(checkpoint, 'checkpoint_feed_forward.pth')