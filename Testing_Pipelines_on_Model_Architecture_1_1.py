# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 10:27:40 2026

@author: morai

Current situation: We just ran "Testing_Images_Pre_Processing_Pipelines.py" 
and now we have 2 different set of images to compare. Yay!



Goal:  We are trying 2 different data preprocessing pipelines to see 
which one lead to better accuracy. To compare them we are going to train a new 
model with the same model architecture for both data sets. Then we are going
to compare the accuracy on the test set. 
    To ensure propper comparaison, the test and training split will be 
done with the same ratio and seed.


Here we will be testing im_c_1_1 and im_s_1_1

RESULTS: We did it !!! Accuracty of 99.9% on the test set !!! After only 25 ish
epoch !!! That is amazing !!


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
Run 2 with im_c_1_1 and im_s_1_1------------------------------------------------------------
-------------------------------------------------------------------------------
'''


#
# Importing Data
#

try:
    script_dir = os.path.dirname(os.path.abspath(__file__)) #runs on VS Code
except NameError:
#    script_dir = os.path.dirname(os.path.abspath(r'C:\Users\morai\Python_Project\Binary Classification Defect Detection\Computer_Vision_Cracked_Concrete_Detection-main\Computer_Vision_Cracked_Concrete_Detection-main\Main_Binary_Image_Classification.py'))  # Runs in Spyder
    script_dir = os.path.dirname(os.path.abspath(r'C:\\Users\\morai\\Python_Project\\Binary Classification Defect Detection\\Computer_Vision_Cracked_Concrete_Detection-main\\Computer_Vision_Cracked_Concrete_Detection-main'))  # Runs in Spyder

#'C:\\Users\\morai\\Python_Project\\Binary Classification Defect Detection\\Computer_Vision_Cracked_Concrete_Detection-main\\Computer_Vision_Cracked_Concrete_Detection-main'

im_c_1_1 = np.load(r"datasets/im_c_1_1.npy")
im_s_1_1 = np.load(r"datasets/im_s_1_1.npy")



#
# Splitting Data
#

# Create X_1_0
X_1_1 = np.concatenate([im_c_1_1, im_s_1_1], axis=0)    

# Create y_1_0
y_1_1 = np.zeros(shape = (X_1_1.shape[0],))
y_1_1[:len(im_c_1_1)] = 1

del im_c_1_1, im_s_1_1

# Create X_train, y_train, X_test, y_test
from sklearn.model_selection import train_test_split
X_train_1_1, X_test_1_1, y_train_1_1, y_test_1_1 = train_test_split(X_1_1
                                                    ,y_1_1
                                                    ,test_size = .25
                                                    ,random_state = 3
                                                    ,shuffle = True
                                                    )

#
# Run the model
#


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
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping



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



def build_model_1(
    input_shape: tuple[int, int, int] = (256, 256, 1),
    n_filters: int = 10,
    learning_rate: float = 0.001,
) -> Sequential:
    """Build and compile the binary crack-detection CNN.
    
    This is the bet archetecture I have so far

    Architecture: Conv(16×16)→ Max Pooling (3x3)
                → Conv(8×8) → Max Pooling (2x2)
                → Conv (4x4) → Flatten (GlobalAveragePooling)
                → Dense (64) → Dense (32) → Dense (32)
                → Dense (32) → Dense (1 -sigmoid)

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
        Conv2D(filters=32, kernel_size=16, strides=(3, 3), padding='same'),
        LeakyReLU(),
        
        MaxPooling2D(pool_size=(3, 3), padding='same'),
        
        Conv2D(filters=64, kernel_size=8,  strides=(1, 1), padding='same'),
        LeakyReLU(),
        
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        
        Conv2D(filters=128, kernel_size=4,  strides=(1, 1), padding='same'),
        LeakyReLU(),
        
        GlobalAveragePooling2D(),
        
        Dense(units=64),
        LeakyReLU(),
        
        Dense(units=32),
        LeakyReLU(),
        
        Dense(units=32),
        LeakyReLU(),
        
        Dense(units=32),
        LeakyReLU(),
        
#        Dense(units=8),
#        LeakyReLU(),
        
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


model2 = build_model_1()

# We are now ready to fit the model
checkpoint_cb = ModelCheckpoint(
    filepath='models/best_model.keras',
    monitor='val_binary_accuracy',
    mode='max',
    save_best_only=True,
    verbose=1,
)

early_stop_cb = EarlyStopping(
    monitor='val_binary_accuracy',
    mode='max',
    patience=10,
    restore_best_weights=True,
    verbose=1,
)


history2 = model2.fit(x = X_train_1_1
                     ,y = y_train_1_1
                     ,batch_size = 512
                     ,epochs = 100
                     ,validation_split = .2
                     ,callbacks = [checkpoint_cb, early_stop_cb]
                     )


# Plotting Loss: train vs validation
plt.figure()
plt.plot(history2.history['loss'])
plt.plot(history2.history['val_loss'])
plt.legend(['train', 'validation'])
plt.title('Optimizer: Adam lr=0.001, BinaryCrossentropy')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
plt.clf()

# Plotting Accuracy and F1: train vs validation
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history2.history['binary_accuracy'])
axes[0].plot(history2.history['val_binary_accuracy'])
axes[0].set_title('Binary Accuracy')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Accuracy')
axes[0].legend(['train', 'validation'])

axes[1].plot(history2.history['f1_score'])
axes[1].plot(history2.history['val_f1_score'])
axes[1].set_title('F1 Score')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1')
axes[1].legend(['train', 'validation'])

plt.suptitle('Optimizer: Adam lr=0.001, BinaryCrossentropy')
plt.tight_layout()
plt.show()
plt.clf()

pprint.pprint(history2.history)

test_accuracy = model2.evaluate(X_test_1_1, y_test_1_1, batch_size = 512 )[1]

print('Test set accuracy: {}%'.format(test_accuracy * 100))
# Test set accuracy: 99.8342514038086%
#
# let's fucking GOOOOOOO !!!!!!
#
#

model2.save("Best_Model_99p9_Percent.keras")





