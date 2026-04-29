# -*- coding: utf-8 -*-
"""
Concrete Crack Detection — full training pipeline.

Reproduces the 99.9% binary accuracy result end-to-end:
    1. Load raw deck images from archive (2)/Decks/
    2. Preprocess with pipeline 1.1 (local threshold + 5x augmentation on cracked)
    3. Train/test split  (75% / 25%, seed=3)
    4. Train build_model_v3 with early stopping
    5. Evaluate on the held-out test set
    6. Save the trained model to models/

Dataset:
    Kaggle — Structural Defects Network: Concrete Crack Images
    https://www.kaggle.com/datasets/aniruddhsharma/structural-defects-network-concrete-crack-images
    Only the Decks subset is used.

Usage:
    conda activate crack-detection
    python main.py
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

# --- Path setup -----------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.preprocessing import image_manipulation_take_1_1, image_augmentation_1
from src.model import build_model_v3

# --- GPU setup -------------------------------------------------------------
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f'GPU detected: {gpus[0].name}')
else:
    print('No GPU detected — running on CPU.')


'''
-------------------------------------------------------------------------------
1. Load raw images
-------------------------------------------------------------------------------
'''

print('\nLoading images...')

cracked_paths = glob.glob('archive (2)/Decks/Cracked/*.jpg')
solid_paths   = glob.glob('archive (2)/Decks/Non-cracked/*.jpg')

decks_c_all = np.array([np.array(Image.open(p)) for p in cracked_paths])
decks_s_all = np.array([np.array(Image.open(p)) for p in solid_paths])

print(f'  Cracked images loaded : {len(decks_c_all)}')
print(f'  Solid images loaded   : {len(decks_s_all)}')


'''
-------------------------------------------------------------------------------
2. Preprocess
   - Cracked: pipeline 1.1 (local threshold) + 5x flip augmentation
   - Solid:   pipeline 1.1 (local threshold), no augmentation
-------------------------------------------------------------------------------
'''

print('\nPreprocessing...')

im_c_temp = np.array([image_manipulation_take_1_1(img) for img in decks_c_all])
im_c = np.concatenate([image_augmentation_1(im_c_temp[i]) for i in range(len(im_c_temp))], axis=0)

im_s = np.array([image_manipulation_take_1_1(img) for img in decks_s_all])

del decks_c_all, decks_s_all, im_c_temp

print(f'  Cracked after augmentation : {len(im_c)}')
print(f'  Solid                      : {len(im_s)}')


'''
-------------------------------------------------------------------------------
3. Build X, y and split
-------------------------------------------------------------------------------
'''

X = np.concatenate([im_c[:, :, :, np.newaxis], im_s[:, :, :, np.newaxis]], axis=0)

y = np.zeros(X.shape[0])
y[:len(im_c)] = 1

del im_c, im_s

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3, shuffle=True
)

del X, y

print(f'\nDataset split:')
print(f'  Train : {len(X_train)} samples')
print(f'  Test  : {len(X_test)} samples')


'''
-------------------------------------------------------------------------------
4. Train
-------------------------------------------------------------------------------
'''

print('\nBuilding model...')
model = build_model_v3()

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

print('\nTraining...')
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=512,
    epochs=100,
    validation_split=0.2,
    callbacks=[checkpoint_cb, early_stop_cb],
)


'''
-------------------------------------------------------------------------------
5. Evaluate
-------------------------------------------------------------------------------
'''

print('\nEvaluating on test set...')
_, test_accuracy, test_f1 = model.evaluate(X_test, y_test, batch_size=512, verbose=0)

print(f'\n  Test binary accuracy : {test_accuracy * 100:.2f}%')
print(f'  Test F1 score        : {test_f1 * 100:.2f}%')


'''
-------------------------------------------------------------------------------
6. Plot training history
-------------------------------------------------------------------------------
'''

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'],           label='train')
axes[0].plot(history.history['val_loss'],        label='validation')
axes[0].set_title('Loss (BinaryCrossentropy)')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['binary_accuracy'],     label='train acc')
axes[1].plot(history.history['val_binary_accuracy'], label='val acc')
axes[1].plot(history.history['f1_score'],            label='train F1')
axes[1].plot(history.history['val_f1_score'],        label='val F1')
axes[1].set_title('Accuracy & F1 Score')
axes[1].set_xlabel('Epoch')
axes[1].legend()

plt.suptitle(f'Pipeline 1.1 — Final result: {test_accuracy*100:.2f}% accuracy')
plt.tight_layout()
plt.savefig('models/training_history.png', dpi=150)
plt.show()

print('\nDone. Model saved to models/best_model.keras')
