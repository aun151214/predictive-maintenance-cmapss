import os
import argparse
import joblib
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(dataset):
    """Load preprocessed dataset from pickle file"""
    data = joblib.load(os.path.join(DATA_DIR, f"{dataset}.pkl"))
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def build_model(model_name, input_shape):
    """Build different model architectures"""
    model = keras.Sequential()

    if model_name == "lstm":
        model.add(layers.Input(shape=input_shape))
        model.add(layers.LSTM(64, return_sequences=True))
        model.add(layers.LSTM(32))
        model.add(layers.Dense(1))

    elif model_name == "gru":
        model.add(layers.Input(shape=input_shape))
        model.add(layers.GRU(64, return_sequences=True))
        model.add(layers.GRU(32))
        model.add(layers.Dense(1))

    elif model_name == "transformer":
        inputs = keras.Input(shape=input_shape)
        x = layers.LayerNormalization()(inputs)
        x = layers.MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(1)(x)
        model = keras.Model(inputs, outputs)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.compile(optimizer="adam", loss="mse")
    return model


def train_model(model_name, dataset, epochs=100, batch_size=64):
    """Train a model and save it + training history"""
    print(f"Training {model_name} on {dataset} ...")

    X_train, y_train, X_test, y_test = load_data(dataset)
    model = build_model(model_name, input_shape=X_train.shape[1:])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )

    # save model
    model_path = os.path.join(MODELS_DIR, f"{model_name}_{dataset}.keras")
    model.save(model_path)
    print(f"✅ Saved model to {model_path}")

    # save training history
    history_df = pd.DataFrame(history.history)
    hist_path = os.path.join(RESULTS_DIR, f"history_{model_name}_{dataset}.csv")
    history_df.to_csv(hist_path, index=False)
    print(f"📊 Saved training history to {hist_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model type: lstm, gru, transformer")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset: FD001, FD002, FD003, FD004")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    train_model(args.model, args.dataset, args.epochs, args.batch_size)
