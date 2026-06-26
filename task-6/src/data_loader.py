"""
data_loader.py
--------------
Data ingestion, feature engineering, and preprocessing for house price prediction.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def load_housing_data() -> pd.DataFrame:
    """
    Load the California Housing dataset (proxy for Kaggle house prices).

    Returns
    -------
    pd.DataFrame
        Dataset with 20,640 rows and 9 columns.
    """
    california = fetch_california_housing(as_frame=True)
    df = california.frame

    # Rename columns for clarity
    df = df.rename(columns={
        'MedInc': 'median_income',
        'HouseAge': 'house_age',
        'AveRooms': 'avg_rooms',
        'AveBedrms': 'avg_bedrooms',
        'Population': 'population',
        'AveOccup': 'avg_occupancy',
        'Latitude': 'latitude',
        'Longitude': 'longitude',
        'MedHouseVal': 'price'
    })

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for better model performance.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered features.
    """
    df = df.copy()

    # Ratio features
    df['rooms_per_bedroom'] = df['avg_rooms'] / df['avg_bedrooms']
    df['rooms_per_person'] = df['avg_rooms'] / df['avg_occupancy']
    df['income_per_room'] = df['median_income'] / df['avg_rooms']

    # Handle infinity values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median())

    return df


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, 
                 random_state: int = 42) -> Tuple:
    """
    Split data and apply scaling.

    Returns
    -------
    Tuple
        X_train, X_test, y_train, y_test, scaler
    """
    X = df.drop('price', axis=1)
    y = df['price']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
