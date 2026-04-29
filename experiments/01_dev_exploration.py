# -*- coding: utf-8 -*-
"""
@author: morai

Development version — refining the initial pipeline.

Key additions over 00_initial_exploration.py:
    - Split preprocessing into two variants (1_0 global vs 1_1 local threshold)
      to systematically compare their effect on model accuracy
    - Added image_augmentation_1 as a standalone augmentation step decoupled
      from preprocessing, making it composable
    - Extended Image_Display to show up to 3 images side-by-side for faster
      visual comparison of preprocessing variants
    - Checked low-contrast flags on best_cracked sample set

The hypothesis behind the split: global Otsu thresholding (1_0) may be erasing
faint cracks. Local thresholding (1_1) adapts to local intensity and may
preserve them.
"""

import os
import sys
import glob
import pprint
import numpy as np
import matplotlib.pyplot as plt
import skimage as ski
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

# --- Path setup -----------------------------------------------------------
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = r'C:\Users\morai\Python_Project\Binary Classification Defect Detection\Computer_Vision_Cracked_Concrete_Detection-main\Computer_Vision_Cracked_Concrete_Detection-main'

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# --- Shared utilities ------------------------------------------------------
from src.utils import Image_Display, var_memory_report, visualize_kernel
from src.preprocessing import (
    image_manipulation_take_1_0,
    image_manipulation_take_1_1,
    image_manipulation_take_2,
    image_augmentation_1,
)
from src.model import F1ScoreBinary, FocalLoss, build_model_v1, build_model_v2

# --- GPU setup -------------------------------------------------------------
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

for i in range(len(decks_c)):
    Image_Display(decks_c[i, :, :, :], title='Images Cracked ' + str(i))

Image_Display(decks_c_all[1000, :, :, :])
# images 2, 4, 5, 6, 8, 9, 1000 are quite interesting

best_cracked = np.array([decks_c[i, :, :, :] for i in [2, 4, 5, 6, 8, 9]])

list_Decks_Solid = glob.glob('archive (2)/Decks/Non-cracked/*.jpg')
decks_s_all = np.array([np.array(Image.open(fname)) for fname in list_Decks_Solid[:]])

[Image_Display(decks_s_all[im, :, :, :], title='Some Solid Images') for im in range(0, 10000, 3000)]

del list_Decks_Solid


'''
-------------------------------------------------------------------------------
Image manipulation ------------------------------------------------------------
-------------------------------------------------------------------------------

Let's rework our image binarization process — global Otsu may be losing faint cracks.
'''

# Check for low-contrast images in our hand-picked set
for i in best_cracked:
    print(ski.exposure.is_low_contrast(i))


#------------------------------------------------------------------------------
# Small-scale comparison: 1_0 (global) vs 1_1 (local) on best_cracked

best_cracked_manip_1_0 = np.array([image_manipulation_take_1_0(best_cracked[i, :, :, :])
                    for i in range(len(best_cracked))])

best_cracked_manip_1_1 = np.array([image_manipulation_take_1_1(best_cracked[i, :, :, :])
                    for i in range(len(best_cracked))])

for i in range(len(best_cracked_manip_1_0)):
    Image_Display(best_cracked[i, :, :],  best_cracked_manip_1_0[i, :, :],
                  best_cracked_manip_1_1[i, :, :],
                  title='raw image #' + str(i),
                  title2='image_manipulation_take_1_0 #' + str(i),
                  title3='image_manipulation_take_1_1')
# Opinion: the cracks are not more obvious after 1_1 on this sample set.
# Conclusion: will compare both pipelines systematically on the full dataset.


#------------------------------------------------------------------------------
# Apply Take 2 to the full cracked dataset (5x augmentation)

images_t2_cracked = np.concatenate([image_manipulation_take_2(im) for im in decks_c_all])

# Solid images use Take 1 (no augmentation)
images_t1_solid = np.stack([image_manipulation_take_1_0(im) for im in decks_s_all])


'''
-------------------------------------------------------------------------------
Data Preparation --------------------------------------------------------------
-------------------------------------------------------------------------------
'''

cracked = images_t2_cracked[:, :, :, np.newaxis]
solid   = images_t1_solid[:, :, :, np.newaxis]

X = np.concatenate([cracked, solid], axis=0)

y = np.zeros(shape=(X.shape[0],))
y[:len(cracked)] = 1

del solid, cracked

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=3, shuffle=True)

# Sanity check
for i in range(20):
    Image_Display(X_train[i, :, :, :], title='image #' + str(i) + ' cracked: ' + str(y_train[i]))
for i in range(20):
    Image_Display(X_test[i, :, :, :], title='image #' + str(i) + ' cracked: ' + str(y_test[i]))

var_memory_report()

del X, y, images_t2_cracked, images_t1_solid

var_memory_report()


'''
-------------  Exporting Data  -----------------------------------------------
'''

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)
np.save('X_train.npy', X_train)
np.save('y_train.npy', y_train)


'''
-------------------------------------------------------------------------------
Model Building ----------------------------------------------------------------
-------------------------------------------------------------------------------
'''

visualize_kernel(decks_c[1, :, :], kernel_size=32, top_left=(100, 100))
Image_Display(decks_c[1, :, :])

model1 = build_model_v2()

model1.evaluate(X_test, y_test)

split = int(len(X_train) * 0.8)
X_tr, X_val = X_train[:split], X_train[split:]
y_tr, y_val = y_train[:split], y_train[split:]

train_ds = tf.data.Dataset.from_tensor_slices((X_tr, y_tr)).batch(32)
val_ds   = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)

history = model1.fit(train_ds, validation_data=val_ds, epochs=8)

history = model1.fit(x=X_train, y=y_train, batch_size=32, epochs=30, validation_split=.2)

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['binary_accuracy'])
plt.legend(['loss', 'binary_accuracy'])
plt.title('SGD 0.001, pooling 2x2, LeakyReLU()')
plt.show()
plt.clf()

pprint.pprint(history.history)

test_accuracy = model1.evaluate(X_test, y_test, batch_size=512)[1]
print('Test set accuracy: {}%'.format(test_accuracy * 100))
