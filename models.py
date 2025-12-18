# models.py
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import tensorflow as tf
from tensorflow.keras import layers, models
from utils import save_sklearn_model, save_tf_model
import warnings

warnings.filterwarnings("ignore")

def metrics(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = (np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9)))) * 100
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}

def train_linear(X: pd.DataFrame, y: pd.Series):
    model = LinearRegression()
    model.fit(X, y)
    preds = model.predict(X)
    m = metrics(y, preds)
    save_sklearn_model(model, "linear_regression")
    return model, m

def train_random_forest(X: pd.DataFrame, y: pd.Series):
    rf = RandomForestRegressor(random_state=42)
    params = {"n_estimators":[50,100],"max_depth":[5,10,None]}
    gs = GridSearchCV(rf, params, cv=2, scoring='neg_mean_squared_error', n_jobs=1)
    gs.fit(X, y)
    best = gs.best_estimator_
    preds = best.predict(X)
    m = metrics(y, preds)
    save_sklearn_model(best, "random_forest")
    return best, m

# LSTM utilities
def build_lstm(input_shape=(30,1), units=64):
    m = models.Sequential()
    m.add(layers.Input(shape=input_shape))
    m.add(layers.LSTM(units))
    m.add(layers.Dropout(0.2))
    m.add(layers.Dense(32, activation='relu'))
    m.add(layers.Dense(1))
    m.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return m

def prepare_lstm(series: pd.Series, lookback=30):
    arr = series.values
    X, y = [], []
    for i in range(len(arr)-lookback):
        X.append(arr[i:i+lookback])
        y.append(arr[i+lookback])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y)
    return X, y

def train_lstm(series: pd.Series, lookback=30, epochs=5, batch_size=16):
    X, y = prepare_lstm(series.dropna(), lookback)
    if len(X) < 10:
        raise ValueError("Not enough data for LSTM")
    model = build_lstm((lookback,1), units=64)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    save_tf_model(model, "lstm_model")
    preds = model.predict(X).flatten()
    return model, metrics(y, preds), history.history

# Prophet - optional (may not be installed)
def train_prophet(df: pd.DataFrame, periods: int = 7):
    try:
        from prophet import Prophet
    except Exception as e:
        raise ImportError("Prophet not installed")
    pdf = df[['date','close']].rename(columns={'date':'ds','close':'y'}).dropna()
    m = Prophet()
    m.fit(pdf)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    save_sklearn_model(m, "prophet_model")
    return m, forecast

def automl_select(models_dict, X, y):
    best_name, best_model, best_metrics = None, None, None
    best_score = float('inf')
    for name, trainer in models_dict.items():
        try:
            model, m = trainer(X, y)
            score = m.get('rmse', float('inf'))
            if score < best_score:
                best_score = score
                best_name, best_model, best_metrics = name, model, m
        except Exception:
            continue
    return best_name, best_model, best_metrics
