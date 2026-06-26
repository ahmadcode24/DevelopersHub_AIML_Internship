"""
visualizations.py
-----------------
Publication-quality plotting functions for house price regression analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import r2_score

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 10

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_price_distribution(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Visualize target price distribution with mean and median lines."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    sns.histplot(df['price'], kde=True, bins=50, color='#2E86AB', ax=axes[0], edgecolor='white')
    axes[0].set_title('House Price Distribution', fontweight='bold')
    axes[0].set_xlabel('Median House Value ($100,000s)')
    axes[0].axvline(df['price'].mean(), color='#C73E1D', linestyle='--', linewidth=2, label=f'Mean: {df["price"].mean():.2f}')
    axes[0].axvline(df['price'].median(), color='#F18F01', linestyle='--', linewidth=2, label=f'Median: {df["price"].median():.2f}')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    sns.boxplot(y=df['price'], color='#2E86AB', ax=axes[1], width=0.3)
    axes[1].set_title('House Price Spread', fontweight='bold')
    axes[1].set_ylabel('Median House Value ($100,000s)')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle('Target Variable: House Price Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '01_price_distribution.png'), dpi=200, bbox_inches='tight', facecolor='white')
    return fig


def plot_predicted_vs_actual(y_test, predictions: dict, save: bool = True) -> plt.Figure:
    """Plot predicted vs actual prices for multiple models."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(6*len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for ax, (name, y_pred), color in zip(axes, predictions.items(), colors):
        ax.scatter(y_test, y_pred, alpha=0.4, s=20, color=color, edgecolors='none')
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, alpha=0.6, label='Perfect Prediction')
        ax.set_xlabel('Actual Price ($100,000s)')
        ax.set_ylabel('Predicted Price ($100,000s)')
        ax.set_title(f'{name}', fontweight='bold')
        ax.legend(fontsize=9, frameon=True)
        ax.grid(True, alpha=0.3)
        r2 = r2_score(y_test, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, fontsize=11, fontweight='bold',
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.suptitle('Predicted vs Actual House Prices', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '06_predicted_vs_actual.png'), dpi=200, bbox_inches='tight', facecolor='white')
    return fig


def plot_residuals(y_test, predictions: dict, save: bool = True) -> plt.Figure:
    """Plot residual analysis for multiple models."""
    fig, axes = plt.subplots(1, len(predictions), figsize=(6*len(predictions), 5))
    if len(predictions) == 1:
        axes = [axes]

    colors = ['#2E86AB', '#A23B72', '#F18F01']

    for ax, (name, y_pred), color in zip(axes, predictions.items(), colors):
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.4, s=20, color=color, edgecolors='none')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        ax.set_xlabel('Predicted Price ($100,000s)')
        ax.set_ylabel('Residuals ($100,000s)')
        ax.set_title(f'{name} — Residuals', fontweight='bold')
        ax.grid(True, alpha=0.3)
        mean_res = residuals.mean()
        std_res = residuals.std()
        ax.text(0.05, 0.95, f'Mean: {mean_res:.3f}\nStd: {std_res:.3f}', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.suptitle('Residual Analysis: Checking for Heteroscedasticity', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, '07_residuals.png'), dpi=200, bbox_inches='tight', facecolor='white')
    return fig
