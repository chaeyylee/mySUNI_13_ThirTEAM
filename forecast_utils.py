# 📁 forecast_utils.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge

def make_time_index(n):
    return np.arange(n, dtype=float).reshape(-1, 1)

def next_dates(last_date, horizon):
    return pd.date_range(last_date + pd.Timedelta(days=1), periods=horizon, freq="D")

def poly_forecast(series: pd.Series, degree=2, n_train=45, horizon=5, ridge_alpha=None):
    y = series.values.astype(float)
    if len(y) < n_train:
        n_train = len(y)

    y_train = y[-n_train:]
    t_train = make_time_index(n_train)
    t_future = np.arange(n_train, n_train + horizon).reshape(-1, 1)

    model = Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
        ("reg", Ridge(alpha=ridge_alpha) if ridge_alpha else LinearRegression())
    ])
    model.fit(t_train, y_train)

    y_pred = model.predict(t_future)
    dof = max(1, len(y_train) - (degree + 1))
    sigma = float(np.sqrt(np.sum((y_train - model.predict(t_train)) ** 2) / dof))
    band = 1.96 * sigma

    dates_future = next_dates(series.index[-1], horizon)
    out = pd.DataFrame({
        "date": dates_future,
        "forecast": y_pred,
        "lower_approx": y_pred - band,
        "upper_approx": y_pred + band
    })
    return out
