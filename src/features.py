import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_lag_features(df, target_col='Close', lags=[1, 2, 3, 5, 7, 14, 30]):
    """
    Creates lag features for time series forecasting.
    """
    df_featured = df.copy()
    for lag in lags:
        df_featured[f'lag_{lag}'] = df_featured[target_col].shift(lag)
    return df_featured

def create_rolling_features(df, target_col='Close', windows=[7, 14, 30]):
    """
    Creates rolling mean and standard deviation features.
    """
    df_featured = df.copy()
    for window in windows:
        df_featured[f'rolling_mean_{window}'] = df_featured[target_col].rolling(window=window).mean()
        df_featured[f'rolling_std_{window}'] = df_featured[target_col].rolling(window=window).std()
    return df_featured

def create_date_features(df):
    """
    Extracts date-based features.
    Assumes the index is a DatetimeIndex.
    """
    df_featured = df.copy()
    df_featured['dayofweek'] = df_featured.index.dayofweek
    df_featured['quarter'] = df_featured.index.quarter
    df_featured['month'] = df_featured.index.month
    df_featured['year'] = df_featured.index.year
    df_featured['dayofyear'] = df_featured.index.dayofyear
    return df_featured

def prepare_data_for_ml(df, target_col='Close', test_size=0.2, scale=False):
    """
    Prepares data for ML models by creating features, handling NaNs, and splitting.
    """
    # Feature Engineering
    df = create_lag_features(df, target_col)
    df = create_rolling_features(df, target_col)
    df = create_date_features(df)
    
    # Drop NaN values created by lags/rolling
    df.dropna(inplace=True)
    
    # Define features and target
    # Exclude target and original price columns to avoid leakage (if they exist)
    exclude_cols = [target_col, 'Date', 'Adj Close', 'Open', 'High', 'Low', 'Volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Split into train and test (Chronological split)
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx]
    test = df.iloc[split_idx:]
    
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    
    scaler = None
    if scale:
        scaler = MinMaxScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=feature_cols, index=X_test.index)
        
    return X_train, y_train, X_test, y_test, feature_cols, scaler
