# -*- coding: utf-8 -*-
"""
@author: morai

Current situation: We have a model that reaches 79% acc on the test set before
overfitting. The limitation seems to be in the data, not the model — some cracks
are very small and the current image processing steps are losing them in noise.

Goal: Create different image preprocessing pipelines and compare each one
visually before feeding them into the model.

Pipelines produced:
    im_c_1_0 / im_s_1_0  — global Otsu threshold (pipeline 1.0)
    im_c_1_1 / im_s_1_1  — local threshold (pipeline 1.1)

These four .npy files are the input to experiments 04 and 05.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# --- Path setup -----------------------------------------------------------
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = r'C:\Users\morai\Python_Project\Binary Classification Defect Detection\Computer_Vision_Cracked_Concrete_Detection-main\Computer_Vision_Cracked_Concrete_Detection-main'

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# --- Shared utilities ------------------------------------------------------
from src.utils import Image_Display
from src.preprocessing import (
    image_manipulation_take_1_0,
    image_manipulation_take_1_1,
    image_augmentation_1,
)

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

list_Decks_Cracked = glob.glob('archive (2)/Decks/Cracked/*.jpg')

decks_c = np.array([np.array(Image.open(fname)) for fname in list_Decks_Cracked[:10]])
decks_c_all = np.array([np.array(Image.open(fname)) for fname in list_Decks_Cracked[:]])

Image_Display(decks_c_all[1000, :, :, :], title='This image')
# images 2, 4, 5, 6, 8, 9, 1000 are quite interesting

best_cracked = np.array([decks_c_all[i, :, :, :] for i in [2, 4, 5, 6, 8, 9, 1000]])

list_Decks_Solid = glob.glob('archive (2)/Decks/Non-cracked/*.jpg')

for i in range(len(best_cracked)):
    Image_Display(best_cracked[i, :, :, :], title='best_cracked ' + str(i))

decks_s_all = np.array([np.array(Image.open(fname)) for fname in list_Decks_Solid[:]])

del list_Decks_Solid, decks_c, list_Decks_Cracked


'''
-------------------------------------------------------------------------------
Image manipulation — side-by-side comparison ----------------------------------
-------------------------------------------------------------------------------
'''

best_cracked_manip_1_0 = np.array([image_manipulation_take_1_0(best_cracked[i, :, :, :])
                    for i in range(len(best_cracked))])

best_cracked_manip_1_1 = np.array([image_manipulation_take_1_1(best_cracked[i, :, :, :])
                    for i in range(len(best_cracked))])

for i in range(len(best_cracked_manip_1_0)):
    Image_Display(image=best_cracked_manip_1_0[i, :, :],
                  image2=best_cracked_manip_1_1[i, :, :],
                  title='best_cracked_manip_1_0 ' + str(i),
                  title2='best_cracked_manip_1_1 ' + str(i))


'''
-------------------------------------------------------------------------------
Data Preparation — build all 4 datasets ---------------------------------------
-------------------------------------------------------------------------------

    im_c_1_0  cracked images, pipeline 1.0, with 5x augmentation
    im_s_1_0  solid images, pipeline 1.0, no augmentation
    im_c_1_1  cracked images, pipeline 1.1, with 5x augmentation
    im_s_1_1  solid images, pipeline 1.1, no augmentation
'''

im_c_1_0_temp = np.array([image_manipulation_take_1_0(decks_c_all[i, :, :, :])
                           for i in range(len(decks_c_all))])

im_c_1_0 = np.concatenate([image_augmentation_1(im_c_1_0_temp[i, :, :])
                            for i in range(len(im_c_1_0_temp))], axis=0)

im_s_1_0 = np.array([image_manipulation_take_1_0(decks_s_all[i, :, :, :])
                      for i in range(len(decks_s_all))])


im_c_1_1_temp = np.array([image_manipulation_take_1_1(decks_c_all[i, :, :, :])
                           for i in range(len(decks_c_all))])

im_c_1_1 = np.concatenate([image_augmentation_1(im_c_1_0_temp[i, :, :])
                            for i in range(len(im_c_1_0_temp))], axis=0)

im_s_1_1 = np.array([image_manipulation_take_1_1(decks_s_all[i, :, :, :])
                      for i in range(len(decks_s_all))])

del im_c_1_0_temp, im_c_1_1_temp


'''
-------------------------------------------------------------------------------
Data Exportation --------------------------------------------------------------
-------------------------------------------------------------------------------
'''

np.save('datasets/im_c_1_0.npy', im_c_1_0)
np.save('datasets/im_s_1_0.npy', im_s_1_0)

np.save('datasets/im_c_1_1.npy', im_c_1_1)
np.save('datasets/im_s_1_1.npy', im_s_1_1)
