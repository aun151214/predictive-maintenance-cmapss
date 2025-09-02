import os
import joblib
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

def preprocess_dataset(fd_id):
    # Load raw training & test data
    train_file = os.path.join(RAW_DIR, f"train_{fd_id}.txt")
    test_file = os.path.join(RAW_DIR, f"test_{fd_id}.txt")
    rul_file = os.path.join(RAW_DIR, f"RUL_{fd_id}.txt")

    train_df = pd.read_csv(train_file, sep=" ", header=None)
    test_df = pd.read_csv(test_file, sep=" ", header=None)
    rul_df = pd.read_csv(rul_file, sep=" ", header=None)

    # Clean blank columns
    train_df = train_df.dropna(axis=1, how="all")
    test_df = test_df.dropna(axis=1, how="all")

    # Rename columns
    col_names = ["unit", "time"] + [f"op_setting_{i}" for i in range(3)] + [f"sensor_{i}" for i in range(21)]
    train_df.columns = col_names
    test_df.columns = col_names

    # Compute RUL for training set
    rul_train = train_df.groupby("unit")["time"].max().reset_index()
    rul_train.columns = ["unit", "max_time"]
    train_df = train_df.merge(rul_train, on="unit")
    train_df["RUL"] = train_df["max_time"] - train_df["time"]
    train_df = train_df.drop("max_time", axis=1)

    # Compute RUL for test set
    rul_values = rul_df[0].values
    rul_test = test_df.groupby("unit")["time"].max().reset_index()
    rul_test.columns = ["unit", "max_time"]
    rul_test["RUL"] = rul_values
    test_df = test_df.merge(rul_test, on="unit")
    test_df["RUL"] = test_df["max_time"] - test_df["time"] + test_df["RUL"]
    test_df = test_df.drop("max_time", axis=1)

    # Select features (drop id/time)
    features = [c for c in train_df.columns if c not in ["unit", "time", "RUL"]]

    # Windowing
    window_size = 30
    def create_windows(df):
        X, y = [], []
        for uid in df["unit"].unique():
            unit_df = df[df["unit"] == uid]
            values = unit_df[features].values
            rul_vals = unit_df["RUL"].values
            for i in range(len(values) - window_size + 1):
                X.append(values[i:i+window_size])
                y.append(rul_vals[i+window_size-1])
        return np.array(X), np.array(y)

    X_train, y_train = create_windows(train_df)
    X_test, y_test = create_windows(test_df)

    # Save dataset as .pkl
    out_file = os.path.join(PROCESSED_DIR, f"{fd_id}.pkl")
    joblib.dump({
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test
    }, out_file)

    print(f"✅ {fd_id}: X_train {X_train.shape}, y_train {y_train.shape}, X_test {X_test.shape}, y_test {y_test.shape}")
    print(f"💾 Saved to {out_file}")

if __name__ == "__main__":
    for fd in ["FD001", "FD002", "FD003", "FD004"]:
        print(f"\n🔹 Processing {fd} ...")
        preprocess_dataset(fd)
