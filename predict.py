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
parser = argparse.ArgumentParser(description='This script uses a trained network to predict the class for an input image')
parser.add_argument('image_path', action='store', default='flowers/test/29/image_04137.jpg', help='Enter the path to a single image')
parser.add_argument('filepath', action='store', default='checkpoint.pth', help='Enter the path for the checkpoint')
parser.add_argument('--top_k', action='store', dest='topk', type=int, default='1', help='Return top K most likely classes')
parser.add_argument('--category_names', action='store', dest='cat_to_name', default='cat_to_name.json', help='Use a mapping of categories to real names')
parser.add_argument('--gpu', action='store_const', dest='device', const='cuda', default='cpu', help='Use GPU for inference')
args = parser.parse_args()

# Defining the load checkpoint function
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    
    arch = checkpoint['arch']
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
    elif arch == "alexnet":
        model = models.alexnet(pretrained=True)
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
    else:
        print("Invalid model. Exiting...")
        exit()
    
    for param in model.parameters():
        param.required_grad = False
        
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])

    return model

# Defining the predict function
def predict(image_path, model, device, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
        
    model.eval()
    model.to(device)
    
    image = torch.from_numpy(process_image(image_path)).float().to(device)
    image.unsqueeze_(0)
    image_predict = torch.exp(model(image))
    
    top_p, top_class = image_predict.topk(topk, dim=1)

    return top_p, top_class

# Loading the checkpoint
model = load_checkpoint(args.filepath)

# Running the prediction
probs, classes = predict(args.image_path, model, args.device, args.topk)

# Convert tensors to numpy array
probs = probs.cpu().detach().numpy()
classes = classes.cpu().detach().numpy()

# Convert array to list
probs = probs.tolist()[0]
classes = classes.tolist()[0]

# Mapping the index to class and class names
cat_to_name = label_mapping(args.cat_to_name)
idx_to_class = dict([[v,k] for k, v in model.class_to_idx.items()])
classes = [idx_to_class[i] for i in classes]
class_names = [cat_to_name[i] for i in classes]

# Identifying the correct index label for the image
p = Path(args.image_path)
index = p.parts[2]

# Printing the results
print("The correct picture label is: " + cat_to_name[index])
print("The predicted top {} class probabilities are: ".format(args.topk) + str(probs))
print("The predicted class numbers are: " + str(classes))
print("The corresponding class names are" + str(class_names))


'''
# For Notebook only: 
Display an image along with the top 5 classes

def plot(filepath, probs, classes, cat_to_name, index):
    image = process_image(filepath)
    plt.imshow(image)
    plt.title(cat_to_name[index])
    
    plt.figure(figsize = (5, 10))
    plt.subplot(2, 2, 1)
    sns.barplot(x = probs, y = class_names)
    plt.show()
    
plot(args.image_path, probs, classes, cat_to_name, index)
'''