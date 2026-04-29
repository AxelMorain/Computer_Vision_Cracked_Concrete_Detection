# -*- coding: utf-8 -*-
"""
@author: morai

Context: 02_preprocessing_pipelines.py generated 4 datasets. Now we evaluate
pipeline 1.0 (global Otsu threshold) against the best model architecture.

Goal: Train build_model_v3 on im_c_1_0 / im_s_1_0 and measure test accuracy.
To ensure fair comparison with experiment 05, the same train/test ratio and
random seed are used.

Result: Test set accuracy ~80% — pipeline 1.0 is the bottleneck.
→ See 05_pipeline_1_1_evaluation.py for the breakthrough.
"""

import os
import sys
import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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
from src.model import F1ScoreBinary, FocalLoss, build_model_v3

# --- GPU setup -------------------------------------------------------------
tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


'''
-------------------------------------------------------------------------------
Run 1 — im_c_1_0 and im_s_1_0 -----------------------------------------------
-------------------------------------------------------------------------------
'''

#
# Importing Data
#

im_c_1_0 = np.load('datasets/im_c_1_0.npy')
im_s_1_0 = np.load('datasets/im_s_1_0.npy')

#
# Splitting Data
#

X_1_0 = np.concatenate([im_c_1_0, im_s_1_0], axis=0)

y_1_0 = np.zeros(shape=(X_1_0.shape[0],))
y_1_0[:len(im_c_1_0)] = 1

del im_c_1_0, im_s_1_0

X_train_1_0, X_test_1_0, y_train_1_0, y_test_1_0 = train_test_split(
    X_1_0, y_1_0, test_size=.25, random_state=3, shuffle=True
)

#
# Run the model
#

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
    x=X_train_1_0,
    y=y_train_1_0,
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

plt.suptitle('Pipeline 1.0 (global Otsu) — Adam lr=0.001')
plt.tight_layout()
plt.show()
plt.clf()

pprint.pprint(history.history)

test_accuracy = model1.evaluate(X_test_1_0, y_test_1_0, batch_size=512)[1]
print('Test set accuracy: {}%'.format(test_accuracy * 100))
# output: Test set accuracy: 80.07%
