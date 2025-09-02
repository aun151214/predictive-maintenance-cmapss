import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow import keras

DATA_DIR = "data/processed"
MODELS_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(dataset):
    """Load preprocessed dataset from pickle file"""
    data = joblib.load(os.path.join(DATA_DIR, f"{dataset}.pkl"))
    return data["X_train"], data["y_train"], data["X_test"], data["y_test"]


def evaluate_model(model_name, dataset):
    """Evaluate trained model and save results"""
    print(f"Evaluating {model_name} on {dataset} ...")

    _, _, X_test, y_test = load_data(dataset)

    model_path = os.path.join(MODELS_DIR, f"{model_name}_{dataset}.keras")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found for {model_name} on {dataset}")

    model = keras.models.load_model(model_path)
    y_pred = model.predict(X_test).flatten()

    # compute metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 {model_name} on {dataset}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    # save results row
    results_path = os.path.join(RESULTS_DIR, "metrics.csv")
    row = pd.DataFrame([{
        "model": model_name,
        "dataset": dataset,
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }])

    if os.path.exists(results_path):
        row.to_csv(results_path, mode="a", header=False, index=False)
    else:
        row.to_csv(results_path, index=False)

    print(f"✅ Results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model type: lstm, gru, transformer")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset: FD001, FD002, FD003, FD004")
    args = parser.parse_args()

    evaluate_model(args.model, args.dataset)
