import yaml
import json
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import mlflow


def main():

    mlflow.set_experiment("Heart Disease Prediction")

    params = yaml.safe_load(open("params.yaml"))["evaluate"]

    test_df = pd.read_csv("data/test.csv")

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    # Load the BEST model selected during training
    model = joblib.load("models/best_model.joblib")

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }

    with mlflow.start_run(run_name="Best Model Evaluation"):

        mlflow.log_metrics(metrics)

        with open("metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)

        mlflow.log_artifact("metrics.json")

    print("=" * 50)
    print("Evaluation Results")
    print("=" * 50)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print("=" * 50)

    if accuracy < params["min_accuracy"]:
        raise Exception(
            f"Accuracy {accuracy:.4f} is below threshold ({params['min_accuracy']})"
        )

    print("Best model passed quality gate.")


if __name__ == "__main__":
    main()