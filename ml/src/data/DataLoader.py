import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transformsi

class VowelConsonantDataset(Dataset):
    def __init__(self, file_path,train=True,transform=None):
        # transformation to be applied.
        self.transform = transform
        #path of images.
        self.file_path=file_path
        self.train=train
        # CSV files where labels are given - explicitly to be passed.
        self.data_info = pd.read_csv("/content/testset.csv")
        self.data_info = self.data_info.sort_values(by = ['Data'])
        self.data_info = self.data_info.reset_index()
        self.file_names = [file for _,_,files in os.walk(self.file_path) for file in files]
        self.file_names = sorted(self.file_names)
        self.len = len(self.file_names)
        
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self,  index):
        file_name = self.file_names[index]
        image_data = self.pil_loader(self.file_path +"/" + file_name)
        if self.transform:
            image_data = self.transform(image_data)
        # gives image and label - image in RGB
        if self.train:
            Y1 = self.get_classes(index)
            label = Y1
            return image_data, label
    
    def pil_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            image_data = img.convert('RGB')
            return image_data
    
    def get_classes(self, index):
        classs = self.data_info['Label'][index]
        return classs
    