# -*- coding: utf-8 -*-
"""
@author: morai

Goal: Test different model architectures on the same dataset (X_train/X_test
exported by 00_initial_exploration.py) without reprocessing images each run.

Due to the way CUDA handles memory, the console must be restarted between runs.
Importing pre-saved .npy files is far more efficient than regenerating datasets
from scratch.

Best architecture found here (build_model_v3):
    Conv(32, 16×16, stride=3) → MaxPool(3×3)
    → Conv(64, 8×8) → MaxPool(2×2)
    → Conv(128, 4×4) → GlobalAveragePooling
    → Dense(64) → Dense(32) → Dense(32) → Dense(32)
    → Dense(1, sigmoid)

Adam lr=0.001 + BinaryCrossentropy
Used ModelCheckpoint + EarlyStopping (patience=10)
"""

import os
import sys
import pprint
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# --- Path setup -----------------------------------------------------------
try:
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    PROJECT_ROOT = r'C:\Users\morai\Python_Project\Binary Classification Defect Detection\Computer_Vision_Cracked_Concrete_Detection-main\Computer_Vision_Cracked_Concrete_Detection-main'

sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# --- Shared utilities ------------------------------------------------------
from src.utils import Image_Display, visualize_kernel
from src.model import F1ScoreBinary, FocalLoss, build_model_v1, build_model_v2, build_model_v3

# --- GPU setup -------------------------------------------------------------
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


'''
-------------  Importing Data  -----------------------------------------------
'''

X_test  = np.load('X_test.npy')
y_test  = np.load('y_test.npy')
X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')

"""
# Sanity check
for i in range(20):
    Image_Display(X_train[i, :, :, :], title='image #' + str(i) + ' cracked: ' + str(y_train[i]))
for i in range(20):
    Image_Display(X_test[i, :, :, :], title='image #' + str(i) + ' cracked: ' + str(y_test[i]))
"""


'''
-------------------------------------------------------------------------------
Model Building — architecture search ------------------------------------------
-------------------------------------------------------------------------------
'''

# Visualize kernel footprint to guide kernel size choices
visualize_kernel(X_test[1, :, :], kernel_size=32, top_left=(100, 100))
Image_Display(X_test[1, :, :])

# Best architecture so far
model1 = build_model_v3()

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

history = model1.fit(
    x=X_train,
    y=y_train,
    batch_size=512,
    epochs=100,
    validation_split=.2,
    callbacks=[checkpoint_cb, early_stop_cb],
)


# Plotting Loss: train vs validation
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train', 'validation'])
plt.title('Loss — Adam lr=0.001, BinaryCrossentropy')
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

plt.suptitle('Adam lr=0.001 — 3×Conv + 4×Dense')
plt.tight_layout()
plt.show()
plt.clf()

pprint.pprint(history.history)

test_accuracy = model1.evaluate(X_test, y_test, batch_size=512)[1]
print('Test set accuracy: {}%'.format(test_accuracy * 100))
