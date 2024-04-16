# -*- coding: utf-8 -*-
"""
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


"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import skimage as ski
import sklearn as skl
import random
from sklearn.preprocessing import normalize


import glob
import os
from PIL import Image

def Image_Display(image, cmap = 'gray', title = 'an image'):
    plt.imshow(image, cmap = cmap)
    plt.title(title)
    plt.show()
    plt.clf()
 
'''
-------------------------------------------------------------------------------
Importing pictures ------------------------------------------------------------
-------------------------------------------------------------------------------
'''

print(os.getcwd())
list_Decks_Cracked = glob.glob('Decks\Cracked'+ '\*.jpg')

# Start with the first 10 images of cracked decks
decks_c = np.array([np.array(Image.open(fname)) \
                    for fname in list_Decks_Cracked[:10]])

decks_c_all = np.array([np.array(Image.open(fname)) \
                    for fname in list_Decks_Cracked[:]])

# display the first 10 images
[Image_Display(decks_c[i, :, :, :], title = 'image ' + str(i)) \
               for i in range(len(decks_c))]
# images 4, 8, 9, 1000 are quite interesting

# Get the solid / non-cracked images
print(os.getcwd())
list_Decks_Solid = glob.glob(r"Decks\Non_cracked" + '\*.jpg')

decks_s_all = np.array([np.array(Image.open(fname)) \
                    for fname in list_Decks_Solid[:]])
# cleaning
del list_Decks_Cracked, list_Decks_Solid


'''
-------------------------------------------------------------------------------
Image manipulation ------------------------------------------------------------
-------------------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
# Take 1 ------------------------------

# Details and thought process about how I made the function bellow
"""
images = decks_c
print(images.shape)

image1 = images[4, :, :, :]
print(image1.shape)
# Each image is made of the 3 rgb layers
plt.imshow(image1)
plt.show()
plt.clf()



# Take 1 - preprocessing overview
# 1. turn the image gray scale
# 2. Contrast enhancement with ski.exposure.
# 3. ski.restoration.denoise_bilateral -- CANCEL
# 4. Thresholding

# Take 1--1.
image1_a = ski.color.rgb2gray(image1)
print(image1_a.shape)
Image_Display(title = 'Before, image1', image = image1)
Image_Display(image1_a,title =  'After, rgb2gray')
# This not much better, let's try something else...
image1_a1 = image1[:, :, 0] # just grab the Red layer
Image_Display(title = 'Before, image1', image = image1)
Image_Display(image1_a1, title =  'After, image1 first layer')
# this is computationaly lighter  and works equaly well!
image1_a1 = normalize(image1_a1)
plt.hist(image1_a1.ravel(), bins = 256)
plt.show()
plt.clf()
image1_a = image1_a1
Image_Display(image1_a, title = 'image1_a')


# Take 1--2. histtogram equalization
# Try a global histogram equalization
image1_b1 = ski.exposure.equalize_hist(image1_a)
plt.hist(image1_b1.ravel(), bins = 256)
plt.show()
plt.clf()
Image_Display(image1_b1, title = 'applied global hist -- nope')
# nope.... This is very ugly....
# Try a local histogram equalization algorithm
image1_b2 = ski.exposure.equalize_adapthist(image1_a)
plt.hist(image1_b2.ravel(), bins = 256)
plt.hist(image1_a.ravel(), bins = 256, color = 'red')
plt.legend(['image1_c2', 'image1_a'])
plt.show()
plt.clf()
Image_Display(image1_a, title = 'image1_a - before')
Image_Display(image1_b2, title = 'image1_c2 - after')
# This is better
image1_b = image1_b2
Image_Display(image1_b, title = 'image1_b')


# Take 1--3. -- cancel!
image1_c = ski.restoration.denoise_bilateral(image1_b )
Image_Display(image1_b, title = 'image1_b -- before')
Image_Display(image1_c, title = 'Denoised, image1_c -- after')
# Nope... this is worst...
del image1_c

# Take 1--4 Thresholding
thresh = ski.filters.threshold_otsu(image1_b)
image1_c = image1_b > thresh
Image_Display(image1_c, title = 'thresholded')

# This is all pretty good, let's make it a function
"""

def Image_Manipulation_Take_1(image):

    # Take 1
    # 1. rgb2gray
    # 2. Contrast enhancement with ski.exposure.
    # 3. ski.restoration.denoise_bilateral -- CANCEL
    # 4. Thresholding
        
    # Take 1--1.
    image1_a1 = image[:, :, 0] # just grab the Red layer

    # this is computationaly lighter  and works equaly well!
    image1_a1 = normalize(image1_a1)
    image1_a = image1_a1

    # Take 1--2. histtogram equalization
    # Try a global histogram equalization
    # Try a local histogram equalization algorithm
    image1_b2 = ski.exposure.equalize_adapthist(image1_a)
    # This is better
    image1_b = image1_b2
    
    # Take 1--4 Thresholding
    thresh = ski.filters.threshold_otsu(image1_b)
    image1_c = image1_b > thresh
    
    return image1_c

# End of Image Manipulation Take 1 ----
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Take 2 ------------------------------

# Image manipulation Take_2 detail and thought process bellow:
"""
# Let's find a handful of interesting cracked images to benchmark our image
# pre-processing work
[Image_Display(decks_c[i, :, :, :], title = 'Image'+ str(i)) \
 for i in range(decks_c.shape[0])]
# images 4, 8, 9, 1000 are the most interesting

goodimages = np.zeros_like((4, 256, 256, 3))
goodimages = decks_c_all[[4, 8, 9, 1000], :, :, :]
# Sanity check =)
np.may_share_memory(goodimages, decks_c)

[Image_Display(goodimages[i, :, :, :], title = 'goodimages'+ str(i)) \
    for i in range(goodimages.shape[0])] 
# Nice!
# Here is an over-view of what I want to pre-precessing I want to do:
# 1. Data augmentation and gray scale 
#   unlike Take_1, we are going to use all chanels of each images
#   chanel1 => keep it as is
#   chanel2 => flip up side down (data augmentation)
#   chanel3 => flip left right (data augmentation)
# 2. contrast enhancing
# That's kinda it ! Binary images are not going to be used. They are going to
# be kept as grayscale.
# Let's see if it makes any difference

# 1.
betterimages = np.zeros(shape = (12, 256, 256), dtype = 'float64')
np.may_share_memory(betterimages, goodimages)

temp = 0
for i in range(goodimages.shape[0]):
    print(str(i))
    for j in range(goodimages.shape[3]):
        print('\t '+ str(j))
#        print('i = {0}, j = {1}'.format(i, j))
        if j == 1 :
            betterimages[temp, :, :] = np.flipud(goodimages[i, :, :, j])
            temp += 1
        elif j == 2:
            betterimages[temp, :, :] = np.fliplr(goodimages[i, :, :, j])
            temp += 1
        else:
            betterimages[temp, :, :] = goodimages[i, :, :, j]
            temp += 1

[Image_Display(betterimages[i, :, :], title = 'betterimages'+ str(i)) \
    for i in range(betterimages.shape[0])] 
# looking good !

# 2.
np.max(betterimages[1, :, :])
np.min(betterimages[1, :, :])

for i in range(betterimages.shape[0]):
    betterimages[i, :, :] = skl.preprocessing.normalize(betterimages[i, :, :])
    
np.max(betterimages[1, :, :])
np.min(betterimages[1, :, :])
# looking better =) 

# Streching the histogram
betterimages = ski.exposure.equalize_adapthist(betterimages)

# looking how it looks now
[Image_Display(betterimages[i, :, :], title = 'betterimages \n with contrast enahncement'+ str(i)) \
    for i in range(betterimages.shape[0])] 
# There is indeed more contrast... it does not look amazingly better, but 
# this is should still be an improvement !
"""


def Image_Manipulation_Take_2(rgb_image_tensor):
    # Here is an over-view of what I want to pre-precessing I want to do:
    # 1. unlike Take_1, we are going to use all chanels of each images
    #   chanel1 => keep it as is
    #   chanel2 => flip up side down (data augmentation)
    #   chanel3 => flip left right (data augmentation)
    # 2. contrast enhancing
    # That's kinda it ! Let's see if it makes any difference

    # 1. Data Augmentation
    
    betterimages_lenght = rgb_image_tensor.shape[0] *3
    betterimages = np.zeros(shape = (betterimages_lenght, 256, 256)\
                            , dtype = 'float64')
    
    temp = 0
    for i in range(rgb_image_tensor.shape[0]):
    #    print(str(i))
        for j in range(rgb_image_tensor.shape[3]):
    #        print('\t '+ str(j))
    #        print('i = {0}, j = {1}'.format(i, j))
            if j == 1 :
                betterimages[temp, :, :] = np.flipud(rgb_image_tensor[i, :, :, j])
                temp += 1
            elif j == 2:
                betterimages[temp, :, :] = np.fliplr(rgb_image_tensor[i, :, :, j])
                temp += 1
            else:
                betterimages[temp, :, :] = rgb_image_tensor[i, :, :, j]
                temp += 1
                
     # 2. Contrast Enhancement
     #normalizing the data
    for i in range(betterimages.shape[0]):
         betterimages[i, :, :] =\
             skl.preprocessing.normalize(betterimages[i, :, :])   
             
     # Applying the contrast enhancement algorithm
    betterimages = ski.exposure.equalize_adapthist(betterimages)
            
    del betterimages_lenght, temp
    
    return betterimages
    
# End of Image Manipulation Take 2 ----
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Image_Manipulation_Take_1 -----------

#Test run on the 10 pictures
images_1 = np.zeros(shape = (len(decks_c), 256, 256))


for i in range(len(decks_c)):
    images_1[i, :, :] = Image_Manipulation_Take_1(decks_c[i, :,:,:])
print('images_1 shape ' + str(images_1.shape))
# display the first 10 images
[Image_Display(images_1[i, :, :], title = 'image ' + str(i)) \
               for i in range(images_1.shape[0])]
# it is all very ugly, but maybe the computer will have an easier time 
# understanding it than me.
    
# Apply the Image_Manipulation_Take_1 to all the pictures in the decks folder
# First the cracked pictures, then the uncracked ones
decks_c_all = np.array([np.array(Image.open(fname)) \
                    for fname in list_Decks_Cracked[:]])
    
print('decks_c_all shape ' + str(decks_c_all.shape))

images_t1_cracked = np.zeros(shape = (decks_c_all.shape[0], 256, 256))
print('images_t1_cracked shape ' + str(images_t1_cracked.shape))

for i in range(decks_c_all.shape[0]):
    images_t1_cracked[i, :, :] = Image_Manipulation_Take_1(decks_c_all[i, :,:,:])
# That took less than a minute to run ! Pretty quick!

Image_Display(images_t1_cracked[100, :, :], title = 'images_t1_cracked 100')
Image_Display(images_t1_cracked[1000, :, :], title = 'images_t1_cracked 1000')
Image_Display(images_t1_cracked[2000, :, :], title = 'images_t1_cracked 2000')
# Looking as expected !

# Now Let's do the same on the uncracked/solid one's
images_t1_solid = np.zeros(shape = (decks_s_all.shape[0], 256, 256))

for i in range(decks_s_all.shape[0]):
    images_t1_solid[i, :, :] = Image_Manipulation_Take_1(decks_s_all[i, :,:,:])
# It only took a few minutes ! This is really fast !

images_t1_solid.shape
Image_Display(images_t1_solid[100, :, :], title = 'images_t1_solid 100')
Image_Display(images_t1_solid[1000, :, :], title = 'images_t1_solid 1000')
Image_Display(images_t1_solid[10000, :, :], title = 'images_t1_solid 10000')
# Looking as expected !

# Cleaning time
del i, decks_c, decks_c_all, decks_s_all, images_1

# End of Image_Manipulation_Take_1 ----
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Image_Manipulation_Take_2 -----------

# Small scale testing
test = decks_c[:10, :, :, :]

test2  = Image_Manipulation_Take_2(test)

[Image_Display(test2[i, :, :], title = 'best images'+ str(i)) \
    for i in range(test2.shape[0])] 
# Looking good    
del test, test2

# Full run
# It is very resource intensive and takes for ever to run
try:
    images_t2_cracked = np.load('images_t2_cracked.npy')
    images_t2_solid = np.load('images_t2_solid.npy')
except:
    # Takes 3 minutes
    images_t2_cracked = Image_Manipulation_Take_2(decks_c_all)
    np.save('images_t2_cracked',images_t2_cracked )
    
    
    # Do the same to the un-cracked images / solid images
    
    # too big to run all the 11k images, 
    # Run each of those chunks one after another, 
    #+/-10 min run time per chunk
    images_t2_solid = Image_Manipulation_Take_2(decks_s_all[0 :5000, :, :, :])
    
    temp = Image_Manipulation_Take_2(decks_s_all[5000: 10000, :, :, :])
    images_t2_solid = np.append(images_t2_solid, temp, axis = 0)
    
    temp1 = Image_Manipulation_Take_2(decks_s_all[10000: , :, :, :])
    images_t2_solid = np.append(images_t2_solid, temp1, axis = 0)
    
    # Let's save the file
    np.save('images_t2_solid',images_t2_solid )
    

# End of Image_Manipulation_Take_2 ----
#------------------------------------------------------------------------------


'''
-------------------------------------------------------------------------------
Data Preparation --------------------------------------------------------------
-------------------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
# Create X for Take 1 -----------------

# The dataset is too large to decently run.
# Let's first only use 4000 images from images_t1_solid
solide_images = 11595
X_lengh = images_t1_cracked.shape[0] + images_t1_solid[:solide_images, :, :].shape[0]
X = np.zeros(shape = (X_lengh, 256, 256, 1))
X[0:images_t1_cracked.shape[0],:,:, :] =\
    np.reshape(images_t1_cracked, newshape = (2025, 256, 256, 1))
X[images_t1_cracked.shape[0]:, :, :] =\
    np.reshape(images_t1_solid[:solide_images, :, :], newshape = (solide_images, 256, 256, 1))



# Create y
y = np.zeros(shape = (X.shape[0],))
y[:images_t1_cracked.shape[0]] = 1

# Create X_train, y_train, X_test, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X
                                                    ,y
                                                    ,test_size = .25
                                                    ,random_state = 3
                                                    ,shuffle = True
                                                    )
# Sanity check
for i in range(16):
    Image_Display(X_train[i, :, :]\
                  , title= 'image #' + str(i) + ' cracked: ' + str(y_train[i]))
# looking good
# End of image preparation for Take 1 -
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Create X for Take 2 -----------------

# The dataset is too large to decently run.
# Let's first only use 12000 images, 6k from cracked and 6k from solid
solide_images = 6000
cracked_images = 6000
X_lengh = solide_images + cracked_images

X = np.zeros(shape = (X_lengh, 256, 256, 1))
X[0:cracked_images,:,:, :] =\
    np.reshape(images_t2_cracked[:cracked_images, :, :]\
               , newshape = (cracked_images, 256, 256, 1))
X[cracked_images:cracked_images + solide_images, :, :, :] =\
    np.reshape(images_t2_solid[:solide_images, :, :], newshape = (solide_images, 256, 256, 1))



# Create y
y = np.zeros(shape = (X.shape[0],))
y[:cracked_images] = 1

del solide_images, cracked_images, X_lengh

# Create X_train, y_train, X_test, y_test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X
                                                    ,y
                                                    ,test_size = .25
                                                    ,random_state = 3
                                                    ,shuffle = True
                                                    )

# Sanity check
for i in range(20):
    Image_Display(X_train[i, :, :, :]\
                  , title= 'image #' + str(i) + ' cracked: ' + str(y_train[i]))
for i in range(20):
    Image_Display(X_test[i, :, :, :]\
                  , title= 'image #' + str(i) + ' cracked: ' + str(y_test[i]))
# looking good

# End of image preparation for Take 2 -
#------------------------------------------------------------------------------



'''
-------------------------------------------------------------------------------
Model Building ! --------------------------------------------------------------
-------------------------------------------------------------------------------
'''

# Let's visualize some kernel size before creating our model:

def Kernel_On_Image(image, kernel_size, top_left_corner_coordinate):
    im = np.zeros_like(image)
    im[:,:] = image
    # Sanity check
    #print(np.may_share_memory(im, images_t1_cracked[1000, :, :]))
    im[top_left_corner_coordinate[0] : top_left_corner_coordinate[0] + kernel_size
       ,top_left_corner_coordinate[1] : top_left_corner_coordinate[1] + kernel_size] = .5
    Image_Display(im, title = 'Image with kernel in gray')
    return

Kernel_On_Image(images_t2_cracked[1000, :, :], kernel_size = 32 \
                , top_left_corner_coordinate = [100, 100])
Image_Display(images_t2_cracked[1000, :, :])


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D,LeakyReLU
from keras import Input 
from keras.metrics import BinaryAccuracy, FalseNegatives
from keras.losses import BinaryCrossentropy
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
import keras.utils



def Model1():
    model1 = Sequential()
    model1.add(Input(shape = (256, 256, 1)))
    model1.add(MaxPooling2D(pool_size = (3, 3)
                            ,padding = 'same'                        
                            ))
    model1.add(Conv2D(filters = 10
                      ,input_shape = (256, 256, 1)
                      ,kernel_size = 16
                      ,strides = (1, 1)
                      ,padding = 'same'
                      ,activation = LeakyReLU()
                      ,))
    model1.add(Conv2D(filters = 10
                      ,kernel_size = 8
                      ,strides = (1, 1)
                      ,padding = 'same'
                      ,activation = LeakyReLU()
                      ))
    model1.add(Flatten())
    model1.add(Dense(units = 10
                     ,activation = LeakyReLU()
                     ))
    model1.add(Dense(units= 1
                     ,activation = 'sigmoid'))
    
    model1.compile(optimizer = SGD(learning_rate = 0.01/1)
                   ,loss = BinaryCrossentropy()
    #               ,metrics = [BinaryAccuracy(), FalseNegatives()]
                    ,metrics = [BinaryAccuracy()]
                   ,)
    print(model1.summary())
    
    # Load initial weights
    #model1.save_weights('model1_initial_weights.h5')
    
    '''
    try:
        model1.load_weights('model1_initial_weights.h5')
    except:
        model1.save_weights('model1_initial_weights.h5')
    '''
    return model1


model1 = Model1()

# Before fitting the model, let's dry run it on the Test data to verify the
# output shape and the over all health and speed of the model
'''
model1.evaluate(X_test,y_test )
'''

# While dry testing the model, some shape issues were found.
# They got fixed!
'''
temp = X_test[:10, :, :]
temp1 = np.reshape(temp, newshape=(10, 256, 256,1))

Image_Display(temp1[0, :, :, :], title = 'temp1 (256, 256, 1)')
Image_Display(temp[0, :, :], title = 'temp (256, 256)')
'''

# We are now ready to fit the model
history = model1.fit(x = X_train
                     ,y = y_train
                     ,batch_size = 512
                     ,epochs = 5
                     ,validation_split = .2
                     )


# Plotting the history
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_binary_accuracy'])
#plt.plot(history.history['val_false_negatives_7'])
plt.legend(['loss', 'Validation Acc'])
plt.title('SGD 0.01, pooling 3x3, LeakyReLU()')
plt.show()
plt.clf()

test_accuracy = model1.evaluate(X_test, y_test, batch_size = 512 )[1]

print('Test set accuracy: {}%'.format(test_accuracy * 100))

# NIIIIIIIIICE!!!!!!
# That is really good, it is actialy perfect !!
# Yay!!!!! Awesome !!!!

# Let's save this beautiful model ! =) 
model1.save('model1_take2.tf', save_format='tf')

#model1_test = tf.keras.models.load_model('model1_take2.tf')

'''
Notes:
    Model1 is looking good ! The accracy plateaus after 5ish epochs. Which is
    quite fast. The peak accuracy keep increasing as we feed it more data.
    This is a sign of a healthy model. With the data from 
    Image_Manipulation_Take_1 we con only achieve an accuracy of arround 80%
    Let's go back and re-work an image amnipulation workflow

    Image_Manipulation_take_2 utilized data augmentation techniques allowing
    us to triple the amount of cracked images wich were sparced. This lead to a masive 
    accuracy improvement while still using the same model as before, Model1.
    An accuracy of 100% was achived! Yay !!
    
'''

'''
-------------------------------------------------------------------------------
Model Exploration/Deep Dive ---------------------------------------------------
-------------------------------------------------------------------------------

Let's go above and beyond by taking a deep dive into the model and see what
those gorgeous kernels are doing ! Yay !!!
'''

# Let's find an interseting image to follow throught the process
Image_Display(images_t2_cracked[1000 , :, : ])
[Image_Display(images_t2_cracked[i, :, :], title = 'image i = '+ str(i))\
 for i in range(50, 100, 3) ]
#i = 33, 74, 80  is pretty good

coolimage = images_t2_cracked[1000 , :, : ]


conv1 = model1.layers[1]
weights1 = conv1.get_weights()
weights1[0].shape
kernel_list1 = weights1[0]
kernel1  = kernel_list1[:, :, 0, 0]
print('Shape of kernel1 = {}'.format(kernel1.shape))

# Let's look for an interesting kernel
[Image_Display(kernel_list1[:, :, :, i]\
               ,title = 'kernel from \n kernel_list1 #' + str(i))\
 for i in range(kernel_list1.shape[3])]
    
# None of them are strikingly interesting...
# Let's re-create the feature maps anyway...

def Convolution_with_Padding(image, kernel):
    results = np.zeros(shape = image.shape)
    
    image1 = np.pad(image
                   ,pad_width = (int(kernel.shape[0]/2)\
                                 , int(kernel.shape[1]/2))
                   ,mode = 'constant'
                   )

    
    #with over lapping
    for i in range(image1.shape[0] - kernel.shape[0]):
        for j in range(image1.shape[1] - kernel.shape[1]):
            chunk = image1[i:i + kernel.shape[0],\
                           j:j + kernel.shape[1]]           
            results[i, j] = (chunk * kernel.T).sum()
        
    return results

# we first need to apply the max pooling layer to an image before runing the
# convolution

def Max_Pooling(image, pool_size):
    h = int(round((image.shape[0] / pool_size[0]) + (image.shape[0] % pool_size[0] > 0), 0))
    w = int(round(image.shape[1] / pool_size[1] + (image.shape[1] % pool_size[1] > 0), 0))
    #The modulus part is to round up
    results = np.zeros(shape = (h, w))
    
    # No overlapping
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            results[i, j] = np.max(image[i * pool_size[0]:i *pool_size[0] + pool_size[0],\
                                         j * pool_size[1]:j *pool_size[1] + pool_size[1]] )

    return results


pooled = Max_Pooling(coolimage, (3, 3))
Image_Display(pooled, title = 'Pooled coolimage')

feature_map = Convolution_with_Padding(pooled, kernel1)
Image_Display(feature_map, title = 'feature map from kernel1')

# Nice !! Let's apply this to all the kernels of first convolution layer to the
# same image

for i in range(kernel_list1.shape[3]):
    kernel_temp = kernel_list1[:, :, 0, i]
    featur_map_temp = Convolution_with_Padding(pooled, kernel_temp)
    Image_Display(featur_map_temp, title = 'feature map from kernel{}'\
                  .format(str(i)))
    
# Let's do this for other images as well 
#i = 33, 74, 80  is pretty good
coolimage = images_t2_cracked[80 , :, : ]
Image_Display(coolimage, title = 'image i = 80')

pooled = Max_Pooling(coolimage, (3, 3))
Image_Display(pooled, title = 'Pooled coolimage')

for i in range(kernel_list1.shape[3]):
    kernel_temp = kernel_list1[:, :, 0, i]
    featur_map_temp = Convolution_with_Padding(pooled, kernel_temp)
    Image_Display(featur_map_temp, title = 'feature map from kernel{}'\
                  .format(str(i)))
