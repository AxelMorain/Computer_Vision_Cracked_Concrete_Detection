# -*- coding: utf-8 -*-
"""
@author: morai

Context: 04_pipeline_1_0_evaluation.py showed that pipeline 1.0 (global Otsu)
plateaus at ~80% accuracy. Here we test pipeline 1.1 (local threshold) on the
exact same model architecture and hyperparameters.

Goal: Determine whether switching to local thresholding meaningfully improves
the model's ability to detect faint cracks.

RESULT: 99.9% binary accuracy and F1-score on the test set after ~25 epochs!
        The bottleneck was entirely in the preprocessing — the model architecture
        from experiment 03 was already powerful enough.

The saved model is: models/Best_Model_99p9_Percent.keras
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
Run 2 — im_c_1_1 and im_s_1_1 -----------------------------------------------
-------------------------------------------------------------------------------
'''

#
# Importing Data
#

im_c_1_1 = np.load('datasets/im_c_1_1.npy')
im_s_1_1 = np.load('datasets/im_s_1_1.npy')

#
# Splitting Data — same ratio and seed as experiment 04 for a fair comparison
#

X_1_1 = np.concatenate([im_c_1_1, im_s_1_1], axis=0)

y_1_1 = np.zeros(shape=(X_1_1.shape[0],))
y_1_1[:len(im_c_1_1)] = 1

del im_c_1_1, im_s_1_1

X_train_1_1, X_test_1_1, y_train_1_1, y_test_1_1 = train_test_split(
    X_1_1, y_1_1, test_size=.25, random_state=3, shuffle=True
)

#
# Run the model — identical architecture and hyperparameters as experiment 04
#

model2 = build_model_v3()

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

history2 = model2.fit(
    x=X_train_1_1,
    y=y_train_1_1,
    batch_size=512,
    epochs=100,
    validation_split=.2,
    callbacks=[checkpoint_cb, early_stop_cb],
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

plt.suptitle('Pipeline 1.1 (local threshold) — Adam lr=0.001')
plt.tight_layout()
plt.show()
plt.clf()

pprint.pprint(history2.history)

test_accuracy = model2.evaluate(X_test_1_1, y_test_1_1, batch_size=512)[1]
print('Test set accuracy: {}%'.format(test_accuracy * 100))
# Test set accuracy: 99.83%  →  let's fucking GOOOOOOO !!!!!!

model2.save('models/Best_Model_99p9_Percent.keras')
