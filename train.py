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
import argparse
from model import *
from utility_functions import *

# Setting up a parser and associated arguments
parser = argparse.ArgumentParser(description='This script helps train a new neural network on a data set')
parser.add_argument('data_dir', action='store', help='Enter the data directory')
parser.add_argument('--save_dir', action='store', dest='save_dir', default=os.getcwd(), help='Enter the directory where you wish to save the checkpoint')
parser.add_argument('--arch', action='store', dest='arch', default='vgg16', help='Choose architecture')
parser.add_argument('--learning_rate', action='store', dest='learning_rate', type=float, default='0.003', help='Set learning rate')
parser.add_argument('--hidden_units', action='store', dest='hidden_units', type=int, default='500', help='Set hidden units')
parser.add_argument('--epochs', action='store', dest='epochs', type=int, default='5', help='Set number of epochs')
parser.add_argument('--gpu', action='store_const', dest='device', const='cuda', default='cpu', help='Use GPU for training')
args = parser.parse_args()

# Build the network
model, criterion, optimizer, device = build_network(args.device, args.arch, args.learning_rate, args.hidden_units)

# Load the data
trainloaders, validloaders, testloaders, train_datasets = load_data(args.data_dir)

# Train the model
model, criterion, optimizer = train_model(device, trainloaders, validloaders, model, criterion, optimizer, args.epochs)

# Test the model
test_model(device, testloaders, model, criterion, args.epochs)

# Save checkpoint
save_checkpoint(args.save_dir, args.arch, model, optimizer, train_datasets)
