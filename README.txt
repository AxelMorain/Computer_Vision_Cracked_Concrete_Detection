Created on Sun Mar 31 13:57:05 2024

@author: Axel

Project:
Implement Convolution layers in a deep neural network to spot cracks in 
images of concrete decks.

Dataset:
The dataset was aquire from Kaggle. In this project we are only using the 
desck images.
https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images/code

Project structure and step overview:
    
Importing Picture:
    -Get all the images in numpy arrays

Image Manipulation:
    -resize, rescale, flip 
    -denoise / smooththing
    -contrast enhancement
    -Thresholding
    -data augmentation

Data preparation:
    -split in training and test set
    -shuffle the data

Model Building:
    -instantiate a sequential model
    -MaxPooling
    -Conv2D layer
    -Flatten layer
    -Dense layer
    -activation functions
    -validation set

Model Exploration/Deep Dive:
    -kernel extraction 
    -kernel visualization
    -feature maping visualization


Notes / overview:
    Model1 is looking good ! The accracy plateaus after 5ish epochs. Which is
    quite fast. The peak accuracy keep increasing as we feed it more data.
    This is a sign of a healthy model. With the data from 
    Image_Manipulation_Take_1 we con only achieve an accuracy of arround 80%
    Let's go back and re-work an image manipulation workflow

    Image_Manipulation_take_2 utilized data augmentation techniques allowing
    us to triple the amount of cracked images wich were sparced. This lead to a masive 
    accuracy improvement while still using the same model as before, Model1.
    An accuracy of 100% was achived! Yay !!
