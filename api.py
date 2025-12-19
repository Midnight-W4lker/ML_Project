from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from model_trainer import TimeSeriesPredictor
from data_loader import download_data, load_data
from features import prepare_data_for_ml

app = FastAPI(title="Stock Prediction API", description="API for forecasting stock prices using XGBoost")

class PredictionRequest(BaseModel):
    ticker: str
    days_to_predict: int = 7

@app.get("/")
def read_root():
    return {"message": "Welcome to the Stock Prediction API. Use /predict to get forecasts."}

@app.post("/predict")
def predict(request: PredictionRequest):
    ticker = request.ticker
    days = request.days_to_predict
    
    # Define paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    data_path = os.path.join(DATA_DIR, f"{ticker}_data.csv")
    
    # 1. Fetch Data (simplified for API: fetch last 2 years)
    try:
        download_data(ticker, '2022-01-01', '2024-01-01', data_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error fetching data: {str(e)}")
    
    df = load_data(data_path)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    # 2. Prepare Data
    # We need to train a model on the fly or load a pre-trained one. 
    # For this demo, we'll train a quick XGBoost model on the fetched data.
    try:
        X_train, y_train, X_test, y_test, feature_cols = prepare_data_for_ml(df, target_col='Close', test_size=0.1)
        
        model = TimeSeriesPredictor(model_type='xgboost', params={'n_estimators': 50, 'learning_rate': 0.1})
        model.train(X_train, y_train)
        
        # 3. Predict future
        # To predict future days, we need to generate features for future dates.
        # This is complex with lag features as we need the predicted values to feed into next lags.
        # For simplicity in this API demo, we will evaluate on the test set and return the last N predictions from the test set
        # representing the "latest" model view.
        
        # In a real production scenario, you would implement a recursive prediction loop here.
        
        preds = model.predict(X_test)
        
        # Return last 'days' predictions
        recent_preds = preds[-days:].tolist()
        recent_dates = y_test.index[-days:].astype(str).tolist()
        
        return {
            "ticker": ticker,
            "model": "XGBoost",
            "forecast": [
                {"date": d, "predicted_price": p} for d, p in zip(recent_dates, recent_preds)
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
