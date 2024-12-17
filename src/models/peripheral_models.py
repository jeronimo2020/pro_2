import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import IsolationForest
from statsmodels.tsa.arima.model import ARIMA
from .base_model import BaseModel

class RegresionPolinomica(BaseModel):
    def __init__(self):
        super().__init__("RegresionPolinomica")
        self.poly = PolynomialFeatures(degree=2)
        self.model = LinearRegression()

    def train(self, data: pd.DataFrame):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        X_poly = self.poly.fit_transform(X)
        self.model.fit(X_poly, y)
        return self.model.score(X_poly, y)

    def predict(self, data: pd.DataFrame):
        X_poly = self.poly.transform(data)
        return self.model.predict(X_poly)

    def get_params(self):
        return {
            'degree': self.poly.degree,
            'model_params': self.model.get_params()
        }

class RegresionLineal(BaseModel):
    def __init__(self):
        super().__init__("RegresionLineal")
        self.model = LinearRegression()

    def train(self, data: pd.DataFrame):
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        self.model.fit(X, y)
        return self.model.score(X, y)

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    def get_params(self):
        return self.model.get_params()

class SeriesTemporales(BaseModel):
    def __init__(self):
        super().__init__("SeriesTemporales")
        self.model = None
        self.model_fit = None

    def train(self, data: pd.Series):
        self.model = ARIMA(data, order=(1,1,1))
        self.model_fit = self.model.fit()
        return self.model_fit.aic

    def predict(self, steps: int = 5):
        return self.model_fit.forecast(steps=steps)

    def get_params(self):
        return {
            'order': self.model.order if self.model else None
        }

class DeteccionAnomalias(BaseModel):
    def __init__(self):
        super().__init__("DeteccionAnomalias")
        self.model = IsolationForest(contamination=0.1)

    def train(self, data: pd.DataFrame):
        self.model.fit(data)
        scores = self.model.score_samples(data)
        return np.mean(scores)

    def predict(self, data: pd.DataFrame):
        return self.model.predict(data)

    def get_params(self):
        return self.model.get_params()