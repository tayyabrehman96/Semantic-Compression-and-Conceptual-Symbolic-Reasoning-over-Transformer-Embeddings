"""1D-CNN on reshaped embedding vectors (sparse categorical labels)."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np


def build_and_compile_cnn_model(
    params_dict: dict[str, Any],
    current_input_dim: int,
    current_num_classes: int,
):
    import tensorflow as tf
    from tensorflow import keras

    keras.backend.clear_session()
    gc.collect()

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(current_input_dim,)))
    model.add(keras.layers.Reshape((current_input_dim, 1)))

    num_cnn_layers = params_dict.get("num_cnn_layers", 1)
    model.add(
        keras.layers.Conv1D(
            filters=params_dict.get("filters_0", 32),
            kernel_size=params_dict.get("kernel_size_0", 3),
            activation="relu",
        )
    )
    model.add(keras.layers.MaxPooling1D(pool_size=params_dict.get("pool_size_0", 2)))
    model.add(keras.layers.Dropout(rate=params_dict.get("dropout_rate", 0.2)))

    if num_cnn_layers >= 2:
        model.add(
            keras.layers.Conv1D(
                filters=params_dict.get("filters_1", 64),
                kernel_size=params_dict.get("kernel_size_1", 3),
                activation="relu",
            )
        )
        model.add(keras.layers.MaxPooling1D(pool_size=params_dict.get("pool_size_1", 2)))
        model.add(keras.layers.Dropout(rate=params_dict.get("dropout_rate", 0.2)))

    model.add(keras.layers.Flatten())

    num_dense_layers = params_dict.get("num_dense_layers", 1)
    model.add(keras.layers.Dense(units=params_dict.get("dense_units_0", 64), activation="relu"))
    model.add(keras.layers.Dropout(rate=params_dict.get("dropout_rate", 0.3)))

    if num_dense_layers >= 2:
        model.add(keras.layers.Dense(units=params_dict.get("dense_units_1", 32), activation="relu"))
        model.add(keras.layers.Dropout(rate=params_dict.get("dropout_rate", 0.3)))

    model.add(keras.layers.Dense(current_num_classes, activation="softmax"))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=params_dict.get("learning_rate", 1e-4)),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def default_cnn_configs() -> list[dict[str, Any]]:
    return [
        {
            "name": "cnn_model_1_simple",
            "num_cnn_layers": 1,
            "filters_0": 32,
            "kernel_size_0": 3,
            "pool_size_0": 2,
            "num_dense_layers": 1,
            "dense_units_0": 64,
            "dropout_rate": 0.3,
            "learning_rate": 1e-4,
        },
        {
            "name": "cnn_model_2_deeper_cnn",
            "num_cnn_layers": 2,
            "filters_0": 64,
            "kernel_size_0": 5,
            "pool_size_0": 2,
            "filters_1": 128,
            "kernel_size_1": 3,
            "pool_size_1": 2,
            "num_dense_layers": 1,
            "dense_units_0": 128,
            "dropout_rate": 0.4,
            "learning_rate": 1e-4,
        },
    ]


def train_cnn_config(
    params_dict: dict[str, Any],
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    num_classes: int,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    import tensorflow as tf

    tf.random.set_seed(seed)
    model = build_and_compile_cnn_model(params_dict, X_train.shape[1], num_classes)
    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
    )
    loss, acc = model.evaluate(X_val, y_val, verbose=0)
    return {
        "name": params_dict["name"],
        "val_loss": float(loss),
        "val_accuracy": float(acc),
        "epochs_ran": len(hist.history.get("loss", [])),
    }
