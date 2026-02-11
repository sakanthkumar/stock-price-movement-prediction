# src/app/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

app = FastAPI()
model = tf.keras.models.load_model('models/lstm_model.h5')

class PortfolioItem(BaseModel):
    ticker: str
    quantity: int
    buy_price: float

@app.post("/predict-risk/")
async def predict_risk(portfolio: list[PortfolioItem]):
    # Placeholder: aggregate features, run model.predict, return risk score
    # For demo, just return dummy risk
    return {"risk_score": 0.42}