# -*- coding: utf-8 -*-
"""
Created on Sun Apr 19 17:46:56 2026

@author: morai

Testing different architectures.

Due to the way CUDA handels memory, after each run the Console needs to be 
restarted. Knowing that, importing the datasets each time will be a more
efficient than creating them from scratch like it is done in 
Main_Binary_Image_Classification.py


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

# I will be running the model on my graphic card with CUDA
import tensorflow as tf
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)










'''
-------------  Importing Data  -----------
'''

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')


# Sanity check
for i in range(20):
    Image_Display(X_train[i, :, :, :]\
                  , title= 'image #' + str(i) + ' cracked: ' + str(y_train[i]))
for i in range(20):
    Image_Display(X_test[i, :, :, :]\
                  , title= 'image #' + str(i) + ' cracked: ' + str(y_test[i]))


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


visualize_kernel(X_test[1, :, :], kernel_size=32, top_left=(100, 100))
Image_Display(X_test[1, :, :])


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





















