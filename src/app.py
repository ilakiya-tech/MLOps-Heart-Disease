from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(
    title="Heart Disease Prediction API",
    version="1.0",
    description="Predict Heart Disease using the Best Machine Learning Model",
)

# Load the best model selected during training
model = joblib.load("models/best_model.joblib")


class HeartData(BaseModel):
    age: float
    sex: float
    cp: float
    trestbps: float
    chol: float
    fbs: float
    restecg: float
    thalach: float
    exang: float
    oldpeak: float
    slope: float
    ca: float
    thal: float


@app.get("/")
def home():
    return {
        "message": "Heart Disease Prediction API is Running 🚀",
        "model": "Best Selected Model",
    }


@app.post("/predict")
def predict(data: HeartData):

    df = pd.DataFrame([data.dict()])

    prediction = model.predict(df)[0]

    return {
        "prediction": int(prediction)
    }