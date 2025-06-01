from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Breast Cancer Prediction API")

model = joblib.load("../ml/model.pkl")

class CancerInput(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(input: CancerInput):
    data = np.array(input.features).reshape(1, -1)
    prediction = model.predict(data)
    result = "Malignant" if prediction[0] == 0 else "Benign"
    return {"prediction": result}
