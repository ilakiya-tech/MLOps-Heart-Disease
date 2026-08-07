import joblib
import pandas as pd

# Load the best model selected during training
model = joblib.load("models/best_model.joblib")

print("=" * 50)
print("Heart Disease Prediction")
print("=" * 50)
print("Using Best Selected Model")
print("=" * 50)

sample = {
    "age": 63,
    "sex": 1,
    "cp": 3,
    "trestbps": 145,
    "chol": 233,
    "fbs": 1,
    "restecg": 0,
    "thalach": 150,
    "exang": 0,
    "oldpeak": 2.3,
    "slope": 0,
    "ca": 0,
    "thal": 1,
}

df = pd.DataFrame([sample])

prediction = model.predict(df)[0]

print(f"Prediction : {prediction}")

if prediction == 1:
    print("Heart Disease Detected")
else:
    print("No Heart Disease Detected")

print("=" * 50)