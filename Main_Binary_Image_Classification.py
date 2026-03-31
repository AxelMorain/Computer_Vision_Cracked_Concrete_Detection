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
import pprint


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

try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) #runs on VS Code
except NameError:
    script_dir = os.path.dirname(os.path.abspath(r'C:\Users\morai\Python_Project\Binary Classification Defect Detection\Computer_Vision_Cracked_Concrete_Detection-main\Computer_Vision_Cracked_Concrete_Detection-main\Main_Binary_Image_Classification.py'))  # Runs in Spyder

os.chdir(script_dir)
list_Decks_Cracked = glob.glob('archive (2)/Decks/Cracked/*.jpg')


# Start with the first 10 images of cracked decks
decks_c = np.array([np.array(Image.open(fname)) \
                    for fname in list_Decks_Cracked[:10]])

decks_c_all = np.array([np.array(Image.open(fname)) \
                    for fname in list_Decks_Cracked[:]])

# display the first 10 images
[Image_Display(decks_c[i, :, :, :], title = 'image ' + str(i)) \
               for i in range(len(decks_c))]
Image_Display(decks_c_all[1000, :, :, :])
# images 4, 8, 9, 1000 are quite interesting

# Get the solid / non-cracked images
list_Decks_Solid = glob.glob('archive (2)/Decks/Non-cracked/*.jpg')

decks_s_all = np.array([np.array(Image.open(fname)) \
                    for fname in list_Decks_Solid[:]])

# Display some of the solid/non-cracked images
[Image_Display(decks_s_all[im, :, :, :], title= "Some Solid Images")\
 for im in range(0, 10000, 3000)]

    
# cleaning
del list_Decks_Solid


'''
-------------------------------------------------------------------------------
Image manipulation ------------------------------------------------------------
-------------------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
# Take 1 ------------------------------



def image_manipulation_take_1(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses an RGB image for crack detection.
    Apply global pixael normalization 
    
    Args:
        image: RGB image array of shape (H, W, 3)
    Returns:
        Binary image array of shape (H, W)
    """
    red = image[:, :, 0].astype(np.float32) / 255.0
    enhanced = ski.exposure.equalize_adapthist(red)
    return enhanced > ski.filters.threshold_otsu(enhanced)


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
             
     # Applying the contrast enhancement algorithm per-image to avoid memory error
    for i in range(betterimages.shape[0]):
        betterimages[i, :, :] = ski.exposure.equalize_adapthist(betterimages[i, :, :])
            
    del betterimages_lenght, temp
    
    return betterimages
    
# End of Image Manipulation Take 2 ----
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Image_Manipulation_Take_1 -----------

#Test run on the 10 pictures

images_1 = np.stack([image_manipulation_take_1(im) for im in decks_c])

for i in range(len(images_1)):
    Image_Display(images_1[i, :, :], title= f"Images After Manipulation id: {i}")


# it is all very ugly, but maybe the computer will have an easier time 
# understanding it than me.
    
# Apply the Image_Manipulation_Take_1 to all the pictures in the decks folder
# First the cracked pictures, then the uncracked ones

images_t1_cracked = np.stack([image_manipulation_take_1(im) \
                              for im in decks_c_all])
# That took less than a minute to run ! Pretty quick!

Image_Display(images_t1_cracked[100, :, :], title = 'images_t1_cracked 100')
Image_Display(images_t1_cracked[1000, :, :], title = 'images_t1_cracked 1000')
Image_Display(images_t1_cracked[2000, :, :], title = 'images_t1_cracked 2000')
# Looking as expected !

# Now Let's do the same on the uncracked/solid one's

images_t1_solid = np.stack([image_manipulation_take_1(im) \
                            for im in decks_s_all])


print(images_t1_solid.shape)
Image_Display(images_t1_solid[100, :, :], title = 'images_t1_solid 100')
Image_Display(images_t1_solid[1000, :, :], title = 'images_t1_solid 1000')
Image_Display(images_t1_solid[10000, :, :], title = 'images_t1_solid 10000')
# Looking as expected !


# Cleaning time
del i, images_1

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
# It is very resource intensive and takes forever to run
try:
    images_t2_cracked = np.load('images_t2_cracked.npy')
    images_t2_solid = np.load('images_t2_solid.npy')
except:
    # Takes 3 minutes
    images_t2_cracked = Image_Manipulation_Take_2(decks_c_all)
    np.save('images_t2_cracked',images_t2_cracked )
    
    
    # Do the same to the un-cracked images / solid images
    # Only process 2000 solid images -> 6000 after augmentation,
    # which is exactly what the model uses (solide_images = 6000)
    images_t2_solid = Image_Manipulation_Take_2(decks_s_all[:2000, :, :, :])

    # Let's save the file
    np.save('images_t2_solid', images_t2_solid)
    

# End of Image_Manipulation_Take_2 ----
#------------------------------------------------------------------------------


'''
-------------------------------------------------------------------------------
Data Preparation --------------------------------------------------------------
-------------------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
# Create X and y for Take 1 -----------------

N_SOLID = 11595  # cap solid samples to limit memory usage and reduce class imbalance

cracked = images_t1_cracked[:, :, :, np.newaxis]              # (n_cracked, 256, 256, 1)
solid   = images_t1_solid[:N_SOLID, :, :, np.newaxis]         # (N_SOLID,   256, 256, 1)

X = np.concatenate([cracked, solid], axis=0)                   # (n_cracked + N_SOLID, 256, 256, 1)



# Create y  (1 = cracked, 0 = solid)
y = np.concatenate([np.ones(cracked.shape[0]), np.zeros(solid.shape[0])])


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



#
# Memory check!
# Before training the model I wanna have a look at the memory usage
# The more RAM we have available the larger the batches can be during training 
#

import sys

def var_memory_report():
    total = 0
    rows = []
    for name, obj in sorted(globals().items()):
        if name.startswith('_'):
            continue
        if isinstance(obj, np.ndarray):
            size = obj.nbytes
        else:
            size = sys.getsizeof(obj)
        total += size
        rows.append((name, type(obj).__name__, size))
    
    rows.sort(key=lambda x: -x[2])
    print(f"{'Variable':30s} {'Type':20s} {'Size (MB)':>10}")
    print("-" * 65)
    for name, typ, size in rows:
        print(f"{name:30s} {typ:20s} {size/1e6:>10.4f}")
    print("-" * 65)
    print(f"{'TOTAL':30s} {'':20s} {total/1e6:>10.4f}")

var_memory_report()

#
# I know we are are going to need to re-work the data the source data sets
# The only one I will need again is X and y which will free up almost a GB
#

del X, y



'''
-------------------------------------------------------------------------------
Model Building ! --------------------------------------------------------------
-------------------------------------------------------------------------------
'''

# Let's visualize some kernel size before creating our model:

def visualize_kernel(image: np.ndarray, kernel_size: int, top_left: tuple[int, int]) -> None:
    """Overlay a kernel footprint on an image and display it.

    Args:
        image:       2-D grayscale array (H, W).
        kernel_size: Side length of the square kernel overlay in pixels.
        top_left:    (row, col) of the kernel's top-left corner.
    """
    row, col = top_left
    h, w = image.shape[:2]
    if row < 0 or col < 0 or row + kernel_size > h or col + kernel_size > w:
        raise ValueError(
            f"Kernel [{row}:{row+kernel_size}, {col}:{col+kernel_size}] "
            f"falls outside image bounds ({h}, {w})."
        )
    im = image.copy()
    im[row:row + kernel_size, col:col + kernel_size] = 0.5
    Image_Display(im, title=f'Kernel {kernel_size}x{kernel_size} at ({row}, {col})')


visualize_kernel(images_t1_cracked[1000, :, :], kernel_size=32, top_left=(100, 100))
Image_Display(images_t1_cracked[1000, :, :])


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, LeakyReLU
from tensorflow.keras import Input
from tensorflow.keras.metrics import BinaryAccuracy, F1Score
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD


class F1ScoreBinary(F1Score):
    """F1Score wrapper that handles 1-D label arrays (shape (N,) instead of (N, 1))."""
    def update_state(self, y_true, y_pred, sample_weight=None):
        if len(y_true.shape) == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        if len(y_pred.shape) == 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)


class FocalLoss(tf.keras.losses.Loss):
    """Binary focal loss for imbalanced classification.

    Down-weights easy examples so training focuses on hard/minority ones.

    Args:
        gamma: Focusing parameter — higher = more focus on hard examples (default 2.0).
        alpha: Class balance weight for the positive class (default 0.25).
    """
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_weight = self.alpha * tf.pow(1 - p_t, self.gamma)
        return tf.reduce_mean(focal_weight * bce)


def build_model(
    input_shape: tuple[int, int, int] = (256, 256, 1),
    n_filters: int = 10,
    learning_rate: float = 0.0001,
) -> Sequential:
    """Build and compile the binary crack-detection CNN.

    Architecture: MaxPool → Conv(16×16) → Conv(8×8) → Flatten → Dense → sigmoid.

    Args:
        input_shape:   (H, W, channels) of the input images.
        n_filters:     Number of filters in each Conv2D layer.
        learning_rate: Learning rate for the SGD optimiser.

    Returns:
        Compiled Keras Sequential model ready for training.
    """
    model = Sequential([
        Input(shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), padding='same'),
        Conv2D(filters=n_filters, kernel_size=16, strides=(1, 1), padding='same'),
        LeakyReLU(),
        Conv2D(filters=n_filters, kernel_size=8,  strides=(1, 1), padding='same'),
        LeakyReLU(),
        Flatten(),
        Dense(units=n_filters),
        LeakyReLU(),
        Dense(units=1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=[BinaryAccuracy(), F1ScoreBinary(threshold=0.5)],
    )
    model.summary()
    return model


model1 = build_model()

# Before fitting the model, let's dry run it on the Test data to verify the
# output shape and the over all health and speed of the model

model1.evaluate(X_test,y_test )


# While dry testing the model, some shape issues were found.
# They got fixed!


# We are now ready to fit the model
history = model1.fit(x = X_train
                     ,y = y_train
                     ,batch_size = 512
                     ,epochs = 8
                     ,validation_split = .2
                     )


# Plotting the history
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['binary_accuracy'])
#plt.plot(history.history['val_false_negatives_7'])
plt.legend(['loss', 'binary_accuracy'])
plt.title('SGD 0.0001, pooling 3x3, LeakyReLU()')
plt.show()
plt.clf()

pprint.pprint(history.history)

test_accuracy = model1.evaluate(X_test, y_test, batch_size = 512 )[1]

print('Test set accuracy: {}%'.format(test_accuracy * 100))


'''
Okay, so this is not good, there are a lot of red flags going on.
- val_binary_accuracy is completely stagnant from the first epoch to the very
last one
- binary accuracy is completely stagnant from the 2nd epoch to the last one
- the losses are going down but not by much...

Overall it does not look like the model is learning much...

The last activation function is a sigmoid one, which is good....
What I think is happening is that we have a strong class imbalance and
the model is being very lazy and is just classifying every image as one class.
We have 10 times more images of solid concrete than cracked concrete.

Code update:
    - adding F1score as a metric
    - Change the loss function to FocalLoss, a loss function designed for class
    imbalance

Outcome:
    - we received an f1_score of 0 on the test set... This is bad, it literally
    cannot get any worse. But on the bright side it does prove our hypothesis
    that the model is not learning and is predicting every image as solid.

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
