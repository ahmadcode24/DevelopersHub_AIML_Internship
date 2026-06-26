"""
models.py
---------
Model training and evaluation utilities for regression tasks.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from typing import Dict


def train_linear_regression(X_train, y_train):
    """Train a standard Linear Regression model."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(X_train, y_train, alpha=1.0, random_state=42):
    """Train a Ridge (L2 regularized) Regression model."""
    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def train_gradient_boosting(X_train, y_train, random_state=42):
    """Train a Gradient Boosting Regressor with sensible defaults."""
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=random_state,
        subsample=0.8
    )
    model.fit(X_train, y_train)
    return model


def evaluate_regression_model(model, X_test, y_test, model_name: str) -> Dict:
    """
    Comprehensive regression model evaluation.

    Returns
    -------
    Dict
        Dictionary with MAE, RMSE, R², and predictions.
    """
    y_pred = model.predict(X_test)

    return {
        'model_name': model_name,
        'mae': mean_absolute_error(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'r2': r2_score(y_test, y_pred),
        'y_pred': y_pred
    }


def cross_validate_regression(model, X, y, cv_folds=5, random_state=42):
    """Perform k-fold cross-validation for regression."""
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    return scores
