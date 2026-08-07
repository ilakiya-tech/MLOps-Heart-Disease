import yaml
import joblib
import pandas as pd
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

MODEL_DIR = Path("models")


def main():

    mlflow.set_experiment("Heart Disease Prediction")

    params = yaml.safe_load(open("params.yaml"))["train"]

    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    models = {

        "Logistic Regression": LogisticRegression(
            C=params["logistic_regression"]["C"],
            max_iter=params["logistic_regression"]["max_iter"],
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=params["random_forest"]["n_estimators"],
            max_depth=params["random_forest"]["max_depth"],
            random_state=params["random_forest"]["random_state"],
        ),

        "XGBoost": XGBClassifier(
            n_estimators=params["xgboost"]["n_estimators"],
            max_depth=params["xgboost"]["max_depth"],
            learning_rate=params["xgboost"]["learning_rate"],
            random_state=params["xgboost"]["random_state"],
            eval_metric="logloss",
        ),
    }

    MODEL_DIR.mkdir(exist_ok=True)

    best_accuracy = 0
    best_model = None
    best_model_name = ""

    print("=" * 50)
    print("Training Models")
    print("=" * 50)

    with mlflow.start_run(run_name="Model Comparison"):

        for model_name, model in models.items():

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)

            print(f"{model_name} Accuracy : {accuracy:.4f}")

            mlflow.log_metric(
                f"{model_name.lower().replace(' ', '_')}_accuracy",
                accuracy,
            )

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = model_name

        # Save best model
        joblib.dump(best_model, MODEL_DIR / "best_model.joblib")

        # Log best model information
        mlflow.log_param("best_model", best_model_name)
        mlflow.log_metric("best_accuracy", best_accuracy)

        print("\n" + "=" * 50)
        print("Best Model")
        print("=" * 50)
        print(f"Model    : {best_model_name}")
        print(f"Accuracy : {best_accuracy:.4f}")
        print("=" * 50)

        # Log only the best model
        if best_model_name == "XGBoost":
            mlflow.xgboost.log_model(
                xgb_model=best_model,
                artifact_path="best_model"
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model"
            )


if __name__ == "__main__":
    main()