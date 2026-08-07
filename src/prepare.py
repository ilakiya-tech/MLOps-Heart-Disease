import yaml
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_dataset():
    return pd.read_csv("data/heart_disease.csv")

DATA_DIR = Path("data")


def main():
    params = yaml.safe_load(open("params.yaml"))["prepare"]

    df = pd.read_csv("data/heart_disease.csv")

# Fill all numeric missing values with median
    df = df.fillna(df.median(numeric_only=True))

    train_df, test_df = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"],
        stratify=df["target"],
    )

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)

    print("===================================")
    print("Data preparation completed successfully.")
    print(f"Train Shape : {train_df.shape}")
    print(f"Test Shape  : {test_df.shape}")
    print("===================================")


if __name__ == "__main__":
    main()