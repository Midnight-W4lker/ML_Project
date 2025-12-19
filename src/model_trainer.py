import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import os

# Try importing Prophet
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

class TimeSeriesPredictor:
    def __init__(self, model_type='arima', params=None):
        self.model_type = model_type.lower()
        self.params = params if params else {}
        self.model = None
        
    def train(self, X_train, y_train):
        if self.model_type == 'arima':
            # ARIMA only needs y_train (univariate)
            order = self.params.get('order', (5,1,0))
            self.model = ARIMA(y_train, order=order)
            self.model_fit = self.model.fit()
            
        elif self.model_type == 'xgboost':
            self.model = XGBRegressor(**self.params)
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(**self.params)
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'svr':
            self.model = SVR(**self.params)
            self.model.fit(X_train, y_train)
            
        elif self.model_type == 'prophet':
            if not PROPHET_AVAILABLE:
                raise ImportError("Prophet is not installed.")
            # Prophet requires 'ds' and 'y' columns
            df = pd.DataFrame({'ds': y_train.index, 'y': y_train.values})
            self.model = Prophet(**self.params)
            self.model.fit(df)
            
    def predict(self, X_test, start=None, end=None):
        if self.model_type == 'arima':
            if start is None or end is None:
                steps = len(X_test) if X_test is not None else 10
                forecast = self.model_fit.forecast(steps=steps)
            else:
                forecast = self.model_fit.predict(start=start, end=end)
            return forecast
            
        elif self.model_type in ['xgboost', 'random_forest', 'svr']:
            return self.model.predict(X_test)
            
        elif self.model_type == 'prophet':
            # Prophet needs a future dataframe
            # If X_test is provided, we use its index
            if X_test is not None:
                future = pd.DataFrame({'ds': X_test.index})
            else:
                # Default to 30 days if no X_test
                future = self.model.make_future_dataframe(periods=30)
            
            forecast = self.model.predict(future)
            return forecast['yhat'].values

    def evaluate(self, y_true, y_pred):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape}

    def save(self, path):
        if self.model_type == 'arima':
            self.model_fit.save(path)
        else:
            joblib.dump(self.model, path)

    def load(self, path):
        if self.model_type == 'arima':
            from statsmodels.tsa.arima.model import ARIMAResults
            self.model_fit = ARIMAResults.load(path)
        else:
            self.model = joblib.load(path)
