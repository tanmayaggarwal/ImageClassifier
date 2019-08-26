# ImageClassifier
An image classification application that uses a deep learning model

Overview: 
A Python application that can train an image classifier on a dataset and then predict new images using the trained model

Details:
The image classifier uses transfer learning leveraging pretrained ImageNet models
The application allows users to choose from different ImageNet model architectures vgg13, vgg16, vgg19, resnet, alexnet) as well as set hyperparameters for learning rate, number of hidden units, and training epochs
Users can print out the top K classes along with associated probabilities and load a JSON file that maps the class values to category names
