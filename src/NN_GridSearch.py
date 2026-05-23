import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from tensorflow.keras import regularizers
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


LOSS_FUNCTION = "binary_crossentropy"
TARGET_AUC = 0.85

# ── Grid ──────────────────────────────────────────────────────────────────────
PARAM_GRID = {
    "learning_rate": [0.001, 0.0005],  # 0.001 fastest to 0.83
    "dropout_rate": [0.1, 0.3],  # 0.1 = fast start; 0.3 = better generalisation
    "l2_reg": [1e-6, 1e-4],  # 1e-6 fast; 1e-4 reduces overfitting
    "batch_size": [256],  # 256 = smoothest gradient
    "epochs": [200],  # EarlyStopping will cut this
    "hidden_units": [512],  # 512 >> 64/128 for this dataset
}


# ── Model factory ─────────────────────────────────────────────────────────────
def create_model(input_dim, hidden_units, dropout_rate=0.1, l2_reg=1e-6):
    """
    Residual neural network with tapering width and ReLU activations.
    Architecture: Input → Dense(units) → [Residual(units), Residual(units//2),
    Residual(units//4)] → Dense(1, sigmoid)
    """
    reg = regularizers.l2(l2_reg)
    inputs = tf.keras.Input(shape=(input_dim,))

    # — Entry block: project input to hidden_units width —————————————————————
    x = tf.keras.layers.Dense(hidden_units, kernel_regularizer=reg)(inputs)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)

    # — Tapering residual blocks —————————————————————————————————
    for size in [hidden_units, hidden_units // 2, hidden_units // 4]:
        shortcut = x
        x = tf.keras.layers.Dense(size, kernel_regularizer=reg)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)
        x = tf.keras.layers.Dense(size, kernel_regularizer=reg)(x)
        # x = tf.keras.layers.BatchNormalization()(x)
        if shortcut.shape[-1] != size:
            shortcut = tf.keras.layers.Dense(size, use_bias=False)(shortcut)
        x = tf.keras.layers.Add()([x, shortcut])
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.Dropout(dropout_rate)(x)

    # — Output ———————————————————————————————————————————————
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)
