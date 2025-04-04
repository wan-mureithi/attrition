# app/main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="IBM Attrition Prediction API")


def model_load():
    try:
        model, metadata = joblib.load("model/attrition_model_metadata.pkl")
        return model, metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail="Error loading model")


def predict_model(model, data):
    try:
        prediction = model.predict(data)
        probability = model.predict_proba(data)[:, 1]
        return prediction, probability
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class AttritionInput(BaseModel):
    Age: int
    MonthlyIncome: float
    OverTime: int
    JobLevel: int
    TotalWorkingYears: float
    YearsAtCompany: float
    BusinessTravel_NonTravel: int
    BusinessTravel_Travel_Rarely: int
    Department_Human_Resources: int
    Department_Sales: int
    # Add more fields if needed


@app.post("/predict")
def predict(data: AttritionInput):
    model, feature_names = model_load()
    df = pd.DataFrame([data.dict()])[feature_names]
    pred, proba = predict_model(model, df)
    return {"prediction": int(pred[0]), "probability": round(float(proba[0]), 4)}
