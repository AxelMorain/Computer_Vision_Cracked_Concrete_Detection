# -*- coding: utf-8 -*-
"""
@author: Axel

Project:
Implement Convolution layers in a deep neural network to spot cracks in
images of concrete decks.

Dataset:
The dataset was acquired from Kaggle. In this project we are only using the
deck images.
https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images/code

Notes / overview:
    Model1 is looking good! The accuracy plateaus after 5ish epochs, which is
    quite fast. The peak accuracy keeps increasing as we feed it more data.
    This is a sign of a healthy model. With the data from
    Image_Manipulation_Take_1 we can only achieve an accuracy of around 80%.
    Let's go back and re-work an image manipulation workflow.

    Image_Manipulation_take_2 utilized data augmentation techniques allowing
    us to multiply the cracked images (which were sparse) 5x. This led to a
    massive accuracy improvement while still using the same model as before.
    An accuracy of 100% was achieved! Yay!!
"""

import os
import sys
import glob
import pprint
import numpy as np
import matplotlib.pyplot as plt
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
    image_manipulation_take_1_0 as image_manipulation_take_1,
    image_manipulation_take_2,
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

# Start with the first 10 images of cracked decks
decks_c = np.array([np.array(Image.open(fname)) for fname in list_Decks_Cracked[:10]])
decks_c_all = np.array([np.array(Image.open(fname)) for fname in list_Decks_Cracked[:]])

# Display the first 10 images
for i in range(len(decks_c)):
    Image_Display(decks_c[i, :, :, :], title='Images Cracked ' + str(i))

Image_Display(decks_c_all[1000, :, :, :])
# images 2, 4, 5, 6, 8, 9, 1000 are quite interesting

best_cracked = [decks_c[i, :, :, :] for i in [2, 4, 5, 6, 8, 9]]

# Get the solid / non-cracked images
list_Decks_Solid = glob.glob('archive (2)/Decks/Non-cracked/*.jpg')
decks_s_all = np.array([np.array(Image.open(fname)) for fname in list_Decks_Solid[:]])

# Display some of the solid/non-cracked images
[Image_Display(decks_s_all[im, :, :, :], title='Some Solid Images') for im in range(0, 10000, 3000)]

del list_Decks_Solid


'''
-------------------------------------------------------------------------------
Image manipulation ------------------------------------------------------------
-------------------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
# Take 1: Basic preprocessing — global Otsu threshold

"""
# Test run on the 10 pictures
images_1 = np.stack([image_manipulation_take_1(im) for im in decks_c])

for i in range(len(images_1)):
    Image_Display(images_1[i, :, :], title=f'Images After Manipulation id: {i}')

# It is all very ugly, but maybe the computer will have an easier time
# understanding it than me.

# Apply Take_1 to all the pictures — first cracked, then uncracked
images_t1_cracked = np.stack([image_manipulation_take_1(im) for im in decks_c_all])
# That took less than a minute to run! Pretty quick!

images_t1_solid = np.stack([image_manipulation_take_1(im) for im in decks_s_all])

del i, images_1
"""

#------------------------------------------------------------------------------
# Take 2: Preprocessing + 5x data augmentation (original + 4 mirror flips)
#
# We are only going to augment the cracked images. If our predictions are
# still bad and we need more data, we will augment the solid images as well.
# Takes a few minutes to run.

images_t2_cracked = np.concatenate([image_manipulation_take_2(im) for im in decks_c_all])

# Solid images: same preprocessing as Take 1, no augmentation
images_t1_solid = np.stack([image_manipulation_take_1(im) for im in decks_s_all])


'''
-------------------------------------------------------------------------------
Data Preparation --------------------------------------------------------------
-------------------------------------------------------------------------------
'''

#------------------------------------------------------------------------------
# Create X and y for Take 1

"""
N_SOLID = 11595  # cap solid samples to limit memory and reduce class imbalance

cracked = images_t1_cracked[:, :, :, np.newaxis]
solid   = images_t1_solid[:N_SOLID, :, :, np.newaxis]

X = np.concatenate([cracked, solid], axis=0)
y = np.concatenate([np.ones(cracked.shape[0]), np.zeros(solid.shape[0])])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=3, shuffle=True)

for i in range(16):
    Image_Display(X_train[i, :, :], title='image #' + str(i) + ' cracked: ' + str(y_train[i]))
"""

#------------------------------------------------------------------------------
# Create X and y for Take 2

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


# Memory check before training
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
Model Building — Take 1 -------------------------------------------------------
-------------------------------------------------------------------------------
'''

# Visualize kernel footprint on a sample image before deciding kernel sizes
visualize_kernel(decks_c[1, :, :], kernel_size=32, top_left=(100, 100))
Image_Display(decks_c[1, :, :])

model1 = build_model_v2()

# Dry run to verify output shape and overall health/speed of the model
model1.evaluate(X_test, y_test)

# Prepare datasets in batches for tf
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


'''
Okay, so this is not good, there are a lot of red flags going on.
- val_binary_accuracy is completely stagnant from the first epoch to the last
- binary accuracy is completely stagnant from the 2nd epoch onwards
- the losses are going down but not by much...

Overall it does not look like the model is learning much.

Hypothesis: strong class imbalance — the model is being lazy and classifying
every image as solid. We have 10x more solid images than cracked ones.

Code update:
    - adding F1score as a metric
    - Change the loss function to FocalLoss, designed for class imbalance

Outcome:
    - f1_score of 0 on the test set. Proves our hypothesis — the model
      predicts every image as solid.

Changes needed:
    - Data augmentation to have more cracked concrete images
'''


'''
-------------------------------------------------------------------------------
Model Building — Take 2 (with augmented data) ---------------------------------
-------------------------------------------------------------------------------
'''

model1 = build_model_v1()

model1.evaluate(X_test, y_test)

history = model1.fit(x=X_train, y=y_train, batch_size=1024, epochs=100, validation_split=.2)

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

test_accuracy = model1.evaluate(X_test, y_test, batch_size=512)[1]
print('Test set accuracy: {}%'.format(test_accuracy * 100))
