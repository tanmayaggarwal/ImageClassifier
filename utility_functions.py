# Imports

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms, models
from PIL import Image
import seaborn as sns
import os
from pathlib import Path
import json

def load_data(data_dir):
    
    # Define the data directories
    data_dir = data_dir
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
     
    # Define transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(255),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    # Load the datasets with ImageFolder and pass the transforms
    train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

    # Using the image datasets, define the dataloaders
    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 64)
    testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 64)

    return trainloaders, validloaders, testloaders, train_datasets

def label_mapping(cat_to_name):
        
    with open(cat_to_name, 'r') as f:
        cat_to_name = json.load(f)

    return cat_to_name

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model, and returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
    im = Image.open(image)
    
    im_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
    im = im_transforms(im).float()
    
    np_image = np.array(im)
    return np_image



     