# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:09:54 2026

@author: morai

Current situation: We have a model that reaches 79% acc on the test set before
overfiting.

I think that the limitation is on the data and not the model. Meaning that model
learned everything it could out of the images. I think some cracks are very 
small and the current images processing steps are losing them in noise
and the model is not able to see them anymore.

Lets create different data preparation pipelines and feed the images into the 
model. We are going to retrain it. What we are keeping is the model 
architecture.



Goal: Create different image pre processing pipelines and compare each one of
them on the same model architecture


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

#for i in range(len(decks_c)):
#    Image_Display(decks_c[i, :, :, :], title = 'Images Cracked ' + str(i))

    
    
Image_Display(decks_c_all[1000, :, :, :], title = "This image")
# images 2, 4, 5, 6, 8, 9, 1000 are quite interesting

best_cracked = np.array([decks_c_all[i, :, :, :] for i in [2, 4, 5, 6, 8, 9, 1000]])
#best_cracked = np.concatenate([best_cracked, decks_c_all[[1000]]], axis = 0)

# Get the solid / non-cracked images
list_Decks_Solid = glob.glob('archive (2)/Decks/Non-cracked/*.jpg')

for i in range(len(best_cracked)):
    Image_Display(best_cracked[i, :, :, :], title = 'best_cracked ' + str(i))

decks_s_all = np.array([np.array(Image.open(fname)) \
                    for fname in list_Decks_Solid[:]])

# Display some of the solid/non-cracked images
#[Image_Display(decks_s_all[im, :, :, :], title= "Some Solid Images")\
# for im in range(0, 10000, 3000)]

    
# cleaning
del list_Decks_Solid, decks_c, list_Decks_Cracked



'''
-------------------------------------------------------------------------------
Image manipulation ------------------------------------------------------------
-------------------------------------------------------------------------------

Let's rework our image binarysation process as it erases some of faint crack
'''




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


#Small scale test

best_cracked_manip_1_0 = np.array([image_manipulation_take_1_0(best_cracked[i, :, :, :])\
                    for i in range(len(best_cracked))])
    
best_cracked_manip_1_1 = np.array([image_manipulation_take_1_1(best_cracked[i, :, :, :])\
                    for i in range(len(best_cracked))])

for i in range(len(best_cracked_manip_1_0)):
    Image_Display(image = best_cracked_manip_1_0[i, :, :],\
                  image2 = best_cracked_manip_1_1[i, :, :] ,\
                  title = 'best_cracked_manip_1_0 ' + str(i),\
                  title2= 'best_cracked_manip_1_1 ' + str(i))


        

'''
-------------------------------------------------------------------------------
Data Preparation --------------------------------------------------------------
-------------------------------------------------------------------------------
We are goin to create 4 data sets:
    im_c_1_0
    im_s_1_0
    im_c_1_1
    im_s_1_1
    

'''


im_c_1_0_temp = np.array([image_manipulation_take_1_0(decks_c_all[i, :, :, :])\
            for i in range(len(decks_c_all))])

im_c_1_0 = np.concatenate([image_augmentation_1(im_c_1_0_temp[i, :, :]) \
            for i in range(len(im_c_1_0_temp))], axis = 0)

im_s_1_0 = np.array([image_manipulation_take_1_0(decks_s_all[i, :, :, :])\
            for i in range(len(decks_s_all))])




im_c_1_1_temp = np.array([image_manipulation_take_1_1(decks_c_all[i, :, :, :])\
            for i in range(len(decks_c_all))])

im_c_1_1 = np.concatenate([image_augmentation_1(im_c_1_0_temp[i, :, :]) \
            for i in range(len(im_c_1_0_temp))], axis = 0)

im_s_1_1 = np.array([image_manipulation_take_1_1(decks_s_all[i, :, :, :])\
            for i in range(len(decks_s_all))])
    
del im_c_1_0_temp, im_c_1_1_temp

'''
-------------------------------------------------------------------------------
Data Exportation --------------------------------------------------------------
-------------------------------------------------------------------------------
We are going to export 4 data sets:
    im_c_1_0
    im_s_1_0
    im_c_1_1
    im_s_1_1    

'''

np.save("datasets/im_c_1_0.npy", im_c_1_0 )
np.save("datasets/im_s_1_0.npy", im_s_1_0 )

np.save("datasets/im_c_1_1.npy", im_c_1_1 )
np.save("datasets/im_s_1_1.npy", im_s_1_1 )


















