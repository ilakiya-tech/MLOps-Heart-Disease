import subprocess
import sys
from pathlib import Path
import pandas as pd
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_exists():
    """Check dataset exists."""

    assert (PROJECT_ROOT / "data" / "heart_disease.csv").exists()


def test_prepare_stage():

    result = subprocess.run(
        [sys.executable, "src/prepare.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    assert (PROJECT_ROOT / "data" / "train.csv").exists()
    assert (PROJECT_ROOT / "data" / "test.csv").exists()


def test_train_stage():

    result = subprocess.run(
        [sys.executable, "src/train.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    assert (PROJECT_ROOT / "models" / "best_model.joblib").exists()


def test_best_model_can_predict():

    model = joblib.load(PROJECT_ROOT / "models" / "best_model.joblib")

    test_df = pd.read_csv(PROJECT_ROOT / "data" / "test.csv")

    X = test_df.drop(columns=["target"])

    prediction = model.predict(X.head(1))

    assert prediction is not None

    assert len(prediction) == 1


def test_evaluation_stage():

    result = subprocess.run(
        [sys.executable, "src/evaluate.py"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0

    assert (PROJECT_ROOT / "metrics.json").exists()