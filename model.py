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

def build_network(device, arch, learning_rate, hidden_units):
    # This function builds a image classifier neural network with 25088 input nodes and 102 output classes
    # This function uses VGG16 as default for its base feature extraction layers, unless an alternate architecture is specified 
    # This function returns the model, criterion, optimizer, and device

    # Use pre-trained model as the starting point
    if arch == "vgg16":
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif arch == "vgg13":
        model = models.vgg13(pretrained=True)
        input_size = 25088
    elif arch == "vgg19":
        model = models.vgg19(pretrained=True)
        input_size = 25088
    elif arch == "alexnet":
        model = models.alexnet(pretrained=True)
        input_size = 9216
    elif arch == "resnet18":
        model = models.resnet18(pretrained=True)
        input_size = 512
    else:
        print("Invalid model name. Please select one of the following model types: vgg13, vgg16, vgg19, alexnet, or resnet18. Please try again. Exiting...")
        exit()
    
    # Freeze parameters so we don't backprop through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Define and assign new model classifier
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('Dropout', nn.Dropout(0.2)),
                              ('fc2', nn.Linear(hidden_units, 102)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))
    model.classifier = classifier
    
    # Checking device type and setting criterion as Negative Log Likelihood Loss function 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.NLLLoss()

    # Training the classifier parameters with a pre-defined learning rate, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Moving the model to the available device
    model.to(device); 
    
    return model, criterion, optimizer, device

def train_model(device, trainloaders, validloaders, model, criterion, optimizer, epochs):
    # This function trains the network on a set of training data loaded into trainloaders
    # This function also runs a validation pass at the end o
    
    epochs = epochs
    steps = 0
    print_every = 5

    train_losses, valid_losses = [], []
    for e in range(epochs):
        running_loss = 0
        for inputs, labels in trainloaders:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        else:
            #Validation pass and print out the validation accuracy
            valid_loss = 0
            accuracy = 0

            # Turn off gradients for validation
            with torch.no_grad():
                model.eval()
                for inputs, labels in validloaders:
                    inputs, labels = inputs.to(device), labels.to(device)

                    logps = model.forward(inputs)
                    valid_loss += criterion(logps, labels)

                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            train_losses.append(running_loss/len(trainloaders))
            valid_losses.append(valid_loss/len(validloaders))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloaders)),
                  "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloaders)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloaders)))

            model.train()

    return model, criterion, optimizer

def test_model(device, testloaders, model, criterion, epochs):
    epochs = epochs
    steps = 0

    for e in range(epochs):
        test_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in testloaders:
                inputs, labels = inputs.to(device), labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)

                test_loss += batch_loss.item()

                # Calculate accuracy
                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim = 1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

            print(f"Epoch: {e+1}/{epochs}.. "
                  f"Test loss: {test_loss/len(testloaders):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloaders):.3f}")

            model.train()

    return 

def save_checkpoint(save_dir, arch, model, optimizer, train_datasets):
    # Save the checkpoint 
    model.class_to_idx = train_datasets.class_to_idx

    checkpoint = {'input_size': 25088,
                  'output_size': 102,
                  'classifier': model.classifier,
                  'optimizer': optimizer,
                  'arch': arch,
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx,
                  'optimizer_state_dict': optimizer.state_dict()}

    filepath = save_dir + '/checkpoint.pth'
    torch.save(checkpoint, filepath)
    #torch.save(checkpoint, 'checkpoint.pth')
    
    return 




