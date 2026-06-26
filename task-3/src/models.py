"""
models.py
---------
Model training, evaluation, and feature importance extraction.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, roc_curve
from typing import Dict, Tuple


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series, 
                               random_state: int = 42) -> Tuple[LogisticRegression, np.ndarray]:
    """
    Train a Logistic Regression model with standard scaling.

    Returns
    -------
    Tuple[LogisticRegression, np.ndarray]
        Trained model and scaled training features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    return model, scaler, X_train_scaled


def train_decision_tree(X_train: pd.DataFrame, y_train: pd.Series,
                         random_state: int = 42) -> DecisionTreeClassifier:
    """Train a regularized Decision Tree classifier."""
    model = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
        class_weight='balanced'
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, model_name: str, scaler=None) -> Dict:
    """
    Comprehensive model evaluation.

    Returns
    -------
    Dict
        Dictionary containing all evaluation metrics.
    """
    if scaler is not None:
        X_test = scaler.transform(X_test)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return {
        'model_name': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_prob),
        'f1_score': f1_score(y_test, y_pred),
        'y_pred': y_pred,
        'y_prob': y_prob,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'fpr': roc_curve(y_test, y_prob)[0],
        'tpr': roc_curve(y_test, y_prob)[1]
    }


def cross_validate_model(model, X, y, cv_folds: int = 5, random_state: int = 42) -> np.ndarray:
    """Perform stratified k-fold cross-validation."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    return scores
