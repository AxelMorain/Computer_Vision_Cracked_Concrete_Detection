# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 19:24:24 2026

@author: morai

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
from skimage.color import rgb2gray
import sklearn as skl
import random
from sklearn.preprocessing import normalize
import pprint
import cv2


import glob
import os
from PIL import Image

def Image_Display(image, image2=None, image3=None, cmap='gray', title='an image', title2='an image', title3='an image'):
    images = [img for img in [image, image2, image3] if img is not None]
    titles = [title, title2, title3][:len(images)]

    if len(images) == 1:
        plt.imshow(images[0], cmap=cmap)
        plt.title(titles[0])
    else:
        fig, axes = plt.subplots(1, len(images), figsize=(5 * len(images), 5))
        for ax, img, t in zip(axes, images, titles):
            ax.imshow(img, cmap=cmap)
            ax.set_title(t)
        plt.tight_layout()
    plt.show()
    plt.clf()

# I will be running the model on my graphic card with CUDA
import tensorflow as tf
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

 
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

for i in range(len(decks_c)):
    Image_Display(decks_c[i, :, :, :], title = 'Images Cracked ' + str(i))

    
    
Image_Display(decks_c_all[1000, :, :, :])
# images 2, 4, 5, 6, 8, 9, 1000 are quite interesting

best_cracked = np.array([decks_c[i, :, :, :] for i in [2, 4, 5, 6, 8, 9]])

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

Let's rework our image binarysation process as it erases some of faint crack
'''



for i in best_cracked:
    print(ski.exposure.is_low_contrast(i))





#------------------------------------------------------------------------------
# Take 1 ------------------------------

def image_manipulation_take_1_0(image: np.ndarray) -> np.ndarray:
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


def image_manipulation_take_1_1(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses an RGB image for crack detection.
    Apply local exposure equalization pixel normalization 
    Apply local thresholding for binarization
    
    Args:
        image: RGB image array of shape (H, W, 3)
    Returns:
        Binary image array of shape (H, W)
    """
    red = image[:, :, 0].astype(np.float32) / 255.0
    enhanced = ski.exposure.equalize_adapthist(red)
    local_thresh = ski.filters.threshold_local(enhanced, block_size=41)  # tune block_size
    return enhanced > local_thresh


#-------------------------
# Small scale test  ------

best_cracked_manip_1_0 = np.array([image_manipulation_take_1_0(best_cracked[i, :, :, :])\
                    for i in range(len(best_cracked))])
    
best_cracked_manip_1_1 = np.array([image_manipulation_take_1_1(best_cracked[i, :, :, :])\
                    for i in range(len(best_cracked))])


    
    
for i in range(len(best_cracked_manip_1_0)):
    Image_Display(best_cracked[i, :, :],  best_cracked_manip_1_0[i, :, :],\
                  best_cracked_manip_1_1[i, :, :],\
                  title = 'raw image #' + str(i),\
                  title2 = 'image_manipulation_take_1_0 #'+ str(i),\
                  title3 = 'image_manipulation_take_1_1')
#Opinion:
#   the creackes are not more obvious after image_manipulation_take_1_1



# End of Image Manipulation Take 1 ----
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Take 2 ------------------------------



def image_manipulation_take_2(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses an RGB image for crack detection with data augmentation.
    Applies global pixel normalization and mirrors across 4 axes (vertical,
    horizontal, main diagonal, anti-diagonal).
    Increases dataset size 5x (original + 4 flips).

    Args:
        image: RGB image array of shape (H, W, 3)
    Returns:
        Binary image array of shape (5, H, W)
    """
    red = image[:, :, 0].astype(np.float32) / 255.0
    enhanced = ski.exposure.equalize_adapthist(red)

    vertical  = np.flipud(enhanced)            # (i,j) → (N-1-i, j)
    horizontal = np.fliplr(enhanced)           # (i,j) → (i, N-1-j)
    main_diag  = enhanced.T                    # (i,j) → (j, i)
    anti_diag  = enhanced[::-1, ::-1].T        # (i,j) → (N-1-j, N-1-i)  ← fixed

    stack = np.stack([enhanced, vertical, horizontal, main_diag, anti_diag])
    # (5, H, W)
    threshold = ski.filters.threshold_otsu(stack)

    return stack > threshold  # (5, H, W) boolean



def image_augmentation_1(image: np.ndarray) -> np.ndarray:
    """
    Mirrors across 4 axes (vertical,horizontal, main diagonal, anti-diagonal).
    Increases dataset size 5x (original + 4 flips).
    
    Parameters
    ----------
    image : np.ndarray
        an image to be augmented

    Returns
    -------
    5 imges (original + 4 flips)

    """
    vertical  = np.flipud(image)            # (i,j) → (N-1-i, j)
    horizontal = np.fliplr(image)           # (i,j) → (i, N-1-j)
    main_diag  = image.T                    # (i,j) → (j, i)
    anti_diag  = image[::-1, ::-1].T        # (i,j) → (N-1-j, N-1-i)  ← fixed
    
    return np.stack([image, vertical, horizontal, main_diag, anti_diag])


#
# Detailed example
#

"""
image = decks_c[5, :,: , 0].astype(np.float32) / 255.0 # grab one images
image.shape
Image_Display(image, title = "original")
enhanced = ski.exposure.equalize_adapthist(image)
Image_Display(enhanced, title="enhanced")

vertical = np.flipud(enhanced)          # or img[::-1]
Image_Display(vertical,  title ="vertical")


horizontal = np.fliplr(enhanced) 
Image_Display(horizontal, title ="horizontal")
horizontal.shape

main_diag = np.transpose(enhanced)   
Image_Display(main_diag, title = 'main_diag')
main_diag.shape

anti_diag  = enhanced[::-1, ::-1].T
Image_Display(anti_diag, title = 'anti_diag')
anti_diag.shape


images = np.stack([enhanced, vertical, horizontal, main_diag, main_diag])
images.shape
# Small scale test
images_t2_cracked_test = np.concatenate\
    ([image_manipulation_take_2(im) for im in decks_c])
images_t2_cracked_test.shape


del image,enhanced, vertical, horizontal, main_diag, images_t2_cracked_test
"""
    
# End of Image Manipulation Take 2 ----
#------------------------------------------------------------------------------



#------------------------------------------------------------------------------
# Image_Manipulation_Take_1 -----------

"""
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
"""

# End of Image_Manipulation_Take_1 ----
#------------------------------------------------------------------------------


#------------------------------------------------------------------------------
# Image_Manipulation_Take_2 -----------


#
# So we are only going to augment the cracked images. If our predictions are 
# still bad and we need more data, we are going to augment the solid images as
# well
#
# Takes a few minute to run
#
    
#image_manipulation_take_2
images_t2_cracked =  np.concatenate\
    ([image_manipulation_take_2(im) for im in decks_c_all])

# Image solid (non cracked) is the same. it stays unchange
images_t1_solid = np.stack([image_manipulation_take_1(im) \
                            for im in decks_s_all])


# Sanity check
#len(images_t2_cracked) == len(images_t1_cracked) * 5
# looking good,


# End of Image_Manipulation_Take_2 ----
#------------------------------------------------------------------------------


'''
-------------------------------------------------------------------------------
Data Preparation --------------------------------------------------------------
-------------------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
# Create X and y for Take 1 -----------------

"""
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
"""


#------------------------------------------------------------------------------
# Create X and y for Take 2 -----------------


cracked = images_t2_cracked[:, :, :, np.newaxis]              # (n_cracked, 256, 256, 1)
solid   = images_t1_solid[:, :, :, np.newaxis]         # (,   256, 256, 1)

X = np.concatenate([cracked, solid], axis=0)                   # (n_cracked + N_SOLID, 256, 256, 1)


# Create y
y = np.zeros(shape = (X.shape[0],))
y[:len(cracked)] = 1

del solid, cracked

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
# The only one I will need again is X and y which will free up almost 1 GB
#

#del X, y

del X, y, images_t2_cracked, images_t1_solid

var_memory_report()




'''
-------------  Exporting Data  -----------
'''

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)



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


visualize_kernel(decks_c[1, :, :], kernel_size=32, top_left=(100, 100))
Image_Display(decks_c[1, :, :])


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, LeakyReLU\
    ,Dropout, GlobalAveragePooling2D
from tensorflow.keras import Input
try:
    from tensorflow.keras.metrics import F1Score
except:
    from tensorflow_addons.metrics import F1Score

from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD


class F1ScoreBinary(F1Score):
    """F1Score wrapper that handles 1-D label arrays (shape (N,) instead of (N, 1))."""
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(num_classes=1, threshold=threshold, **kwargs)

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


def build_model_A(
    input_shape: tuple[int, int, int] = (256, 256, 1),
    n_filters: int = 10,
    learning_rate: float = 0.001,
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
        Conv2D(filters=n_filters, kernel_size=16, padding='same'),
        LeakyReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=n_filters, kernel_size=8, padding='same'),
        LeakyReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        GlobalAveragePooling2D(),   # 128x128x10 → just 10
        Dense(units=n_filters),
        LeakyReLU(),
        Dense(units=n_filters),
        LeakyReLU(),
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss= 'binary_crossentropy',
        metrics=[BinaryAccuracy(), F1ScoreBinary(threshold=0.5)],
    )
    model.summary()
    return model






model1 = build_model_A()

# Before fitting the model, let's dry run it on the Test data to verify the
# output shape and the over all health and speed of the model

model1.evaluate(X_test,y_test )

# While dry testing the model, some shape issues were found.
# They got fixed!

#
# Prepare datasets in batches for tf
#
split = int(len(X_train) * 0.8)
X_tr, X_val = X_train[:split], X_train[split:]
y_tr, y_val = y_train[:split], y_train[split:]

train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(32)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

history = model1.fit(train_ds, validation_data=val_ds, epochs=8)

# We are now ready to fit the model
history = model1.fit(x = X_train
                     ,y = y_train
                     ,batch_size = 32
                     ,epochs = 30
                     ,validation_split = .2
                     )


# Plotting the history
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['binary_accuracy'])
#plt.plot(history.history['val_false_negatives_7'])
plt.legend(['loss', 'binary_accuracy'])
plt.title('SGD 0.001, pooling 2x2, LeakyReLU()')
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

Changes needed:
    -Data augmentation to have more images of creacked concrete

'''





'''
-------------------------------------------------------------------------------
Model Building  Take 2! --------------------------------------------------------------
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


#visualize_kernel(images_t1_cracked[1000, :, :], kernel_size=64, top_left=(100, 100))
#Image_Display(images_t1_cracked[1000, :, :])


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, LeakyReLU
from tensorflow.keras import Input
try:
    from tensorflow.keras.metrics import F1Score
except ImportError:
    from tensorflow_addons.metrics import F1Score
    
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam, SGD


class F1ScoreBinary(F1Score):
    """F1Score wrapper that handles 1-D label arrays (shape (N,) instead of (N, 1))."""
    def __init__(self, threshold=0.5, **kwargs):
        super().__init__(num_classes=1, threshold=threshold, **kwargs)

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
    learning_rate: float = 0.005,
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
#        MaxPooling2D(pool_size=(3, 3), padding='same'),
        Conv2D(filters=32, kernel_size=16, strides=(1, 1), padding='same'),
        LeakyReLU(),
        
        Conv2D(filters=64, kernel_size=8,  strides=(1, 1), padding='same'),
        LeakyReLU(),
        
        Conv2D(filters=128, kernel_size=4,  strides=(1, 1), padding='same'),
        LeakyReLU(),
        
        GlobalAveragePooling2D(),
        
        Dense(units=32),
        LeakyReLU(),
        
        Dense(units=16),
        LeakyReLU(),
        
        Dense(units=8),
        LeakyReLU(),
        
        Dense(units=1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        #loss=FocalLoss(gamma=2.0, alpha=0.25),
        loss=tf.keras.losses.BinaryCrossentropy(),
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
                     ,batch_size = 1024
                     ,epochs = 100
                     ,validation_split = .2
                     )


# Plotting Loss: train vs validation
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'])
plt.title('Loss — Adam lr=0.005, BinaryCrossentropy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.clf()

# Plotting Accuracy and F1: train vs validation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['binary_accuracy'])
axes[0].plot(history.history['val_binary_accuracy'])
axes[0].set_title('Binary Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(['train', 'validation'])

axes[1].plot(history.history['f1_score'])
axes[1].plot(history.history['val_f1_score'])
axes[1].set_title('F1 Score')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1')
axes[1].legend(['train', 'validation'])

plt.suptitle('Adam lr=0.005 — 3×Conv + 3×Dense')
plt.tight_layout()
plt.show()
plt.clf()

pprint.pprint(history.history)

test_accuracy = model1.evaluate(X_test, y_test, batch_size = 512 )[1]

print('Test set accuracy: {}%'.format(test_accuracy * 100))





""" Previous model that worked well, but slow

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 256, 256, 10)      2570      
                                                                 
 leaky_re_lu (LeakyReLU)     (None, 256, 256, 10)      0         
                                                                 
 max_pooling2d (MaxPooling2D  (None, 128, 128, 10)     0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 128, 128, 10)      6410      
                                                                 
 leaky_re_lu_1 (LeakyReLU)   (None, 128, 128, 10)      0         
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 64, 64, 10)       0         
 2D)                                                             
                                                                 
 global_average_pooling2d (G  (None, 10)               0         
 lobalAveragePooling2D)                                          
                                                                 
 dense (Dense)               (None, 10)                110       
                                                                 
 leaky_re_lu_2 (LeakyReLU)   (None, 10)                0         
                                                                 
 dense_1 (Dense)             (None, 10)                110       
                                                                 
 leaky_re_lu_3 (LeakyReLU)   (None, 10)                0         
                                                                 
 dense_2 (Dense)             (None, 1)                 11        
                                                                 
=================================================================
Total params: 9,211
Trainable params: 9,211
Non-trainable params: 0
"""


"""

# -----------------------------------------------------------------------------
# Lets have a look at hte model I saved 2 years ago
#



model_saved = tf.keras.models.load_model('model1_take2.keras')

model_saved = tf.keras.models.load_model(
    'model1_take2.keras',
    custom_objects={'LeakyReLU': tf.keras.layers.LeakyReLU}
)

# Architecture summary
model_saved.summary()

# Hyperparameters (optimizer, loss, metrics, learning rate)
model_saved.optimizer
model_saved.optimizer.learning_rate
model_saved.loss
model_saved.metrics_names

# Detailed config of every layer
model_saved.get_config()


model_save_history = model_saved.evaluate(X_test,y_test )
model_save_history.history













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


"""