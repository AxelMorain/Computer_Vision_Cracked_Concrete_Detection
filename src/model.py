import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, Flatten, Dense, MaxPooling2D,
    LeakyReLU, GlobalAveragePooling2D,
)
from tensorflow.keras import Input
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.optimizers import Adam, SGD

try:
    from tensorflow.keras.metrics import F1Score
except ImportError:
    from tensorflow_addons.metrics import F1Score


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


def build_model_v1(
    input_shape: tuple = (256, 256, 1),
    n_filters: int = 10,
    learning_rate: float = 0.0001,
) -> Sequential:
    """Early CNN: MaxPool → Conv(16×16) → Conv(8×8) → Flatten → Dense → sigmoid.

    Uses FocalLoss + SGD. First attempt at addressing class imbalance.
    Achieved ~80% accuracy — bottleneck was data, not architecture.
    """
    model = Sequential([
        Input(shape=input_shape),
        MaxPooling2D(pool_size=(3, 3), padding='same'),
        Conv2D(filters=n_filters, kernel_size=16, strides=(1, 1), padding='same'),
        LeakyReLU(),
        Conv2D(filters=n_filters, kernel_size=8, strides=(1, 1), padding='same'),
        LeakyReLU(),
        Flatten(),
        Dense(units=n_filters),
        LeakyReLU(),
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=SGD(learning_rate=learning_rate),
        loss=FocalLoss(gamma=2.0, alpha=0.25),
        metrics=[BinaryAccuracy(), F1ScoreBinary(threshold=0.5)],
    )
    model.summary()
    return model


def build_model_v2(
    input_shape: tuple = (256, 256, 1),
    n_filters: int = 10,
    learning_rate: float = 0.001,
) -> Sequential:
    """Intermediate CNN: Conv(16×16) → MaxPool → Conv(8×8) → MaxPool → GAP → Dense × 2 → sigmoid.

    Switched to Adam + BinaryCrossentropy. GlobalAveragePooling replaces Flatten
    to massively reduce parameter count.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(filters=n_filters, kernel_size=16, padding='same'),
        LeakyReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=n_filters, kernel_size=8, padding='same'),
        LeakyReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        GlobalAveragePooling2D(),
        Dense(units=n_filters),
        LeakyReLU(),
        Dense(units=n_filters),
        LeakyReLU(),
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=[BinaryAccuracy(), F1ScoreBinary(threshold=0.5)],
    )
    model.summary()
    return model


def build_model_v3(
    input_shape: tuple = (256, 256, 1),
    n_filters: int = 10,
    learning_rate: float = 0.001,
) -> Sequential:
    """Best CNN: Conv(32,16,s=3) → MaxPool(3) → Conv(64,8) → MaxPool(2) → Conv(128,4) → GAP → Dense × 4 → sigmoid.

    Strided first conv aggressively downsamples 256×256 before the expensive
    Conv layers, making training feasible on GPU. Adam lr=0.001 + BinaryCrossentropy.
    Achieved 99.9% binary accuracy and F1-score on the test set.
    """
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(filters=32, kernel_size=16, strides=(3, 3), padding='same'),
        LeakyReLU(),
        MaxPooling2D(pool_size=(3, 3), padding='same'),
        Conv2D(filters=64, kernel_size=8, strides=(1, 1), padding='same'),
        LeakyReLU(),
        MaxPooling2D(pool_size=(2, 2), padding='same'),
        Conv2D(filters=128, kernel_size=4, strides=(1, 1), padding='same'),
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
        Dense(units=1, activation='sigmoid'),
    ])
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[BinaryAccuracy(), F1ScoreBinary(threshold=0.5)],
    )
    model.summary()
    return model
