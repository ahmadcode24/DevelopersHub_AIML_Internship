"""
data_loader.py
--------------
Data ingestion, cleaning, and preprocessing for the UCI Heart Disease dataset.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from typing import Tuple


def load_heart_disease_data(url: str = None) -> pd.DataFrame:
    """
    Load the UCI Cleveland Heart Disease dataset.

    Parameters
    ----------
    url : str, optional
        Custom URL to the dataset. Defaults to UCI official source.

    Returns
    -------
    pd.DataFrame
        Raw dataset with 303 rows and 14 columns.
    """
    if url is None:
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data'

    columns = [
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
        'restecg', 'thalach', 'exang', 'oldpeak',
        'slope', 'ca', 'thal', 'target'
    ]

    df = pd.read_csv(url, names=columns, na_values='?')
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset: handle missing values, binarize target, fix dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for modeling.
    """
    df = df.copy()

    # Median imputation for missing values (ca, thal)
    imputer = SimpleImputer(strategy='median')
    df[['ca', 'thal']] = imputer.fit_transform(df[['ca', 'thal']])

    # Binarize target: 0 = no disease, 1-4 = disease present
    df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)

    # Convert categoricals to integers
    categorical_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
    df[categorical_cols] = df[categorical_cols].astype(int)

    return df


def get_feature_descriptions() -> dict:
    """Return a dictionary mapping feature names to clinical descriptions."""
    return {
        'age': 'Age in years',
        'sex': 'Sex (1 = male, 0 = female)',
        'cp': 'Chest pain type (1-4)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl',
        'restecg': 'Resting ECG results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise-induced angina (1 = yes)',
        'oldpeak': 'ST depression induced by exercise',
        'slope': 'Slope of peak exercise ST segment (1-3)',
        'ca': 'Number of major vessels colored by fluoroscopy (0-3)',
        'thal': 'Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible)',
        'target': 'Heart disease presence (1 = yes, 0 = no)'
    }
