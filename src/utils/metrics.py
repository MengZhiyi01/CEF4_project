"""
Evaluation metrics for model performance assessment.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Any
from scipy import stats


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE)."""
    return float(np.mean((y_true - y_pred) ** 2))  # type: ignore


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(calculate_mse(y_true, y_pred)))  # type: ignore


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error (MAE)."""
    return float(np.mean(np.abs(y_true - y_pred)))  # type: ignore


def calculate_ic(y_true: Union[np.ndarray, Any], 
                 y_pred: Union[np.ndarray, Any],
                 method: str = 'pearson') -> Tuple[float, float]:
    """
    Calculate Information Coefficient (IC).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        method: Correlation method ('pearson', 'spearman', 'rank_ic')
        
    Returns:
        ic: Information coefficient
        p_value: P-value
    """
    # Ensure inputs are 1D arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    if len(y_true) < 2:
        return np.nan, np.nan
    
    if method == 'pearson':
        ic, p_value = stats.pearsonr(y_true, y_pred)
    elif method == 'spearman':
        ic, p_value = stats.spearmanr(y_true, y_pred)
    elif method == 'rank_ic':
        # Rank IC: Convert to ranks then calculate correlation
        y_true_rank = stats.rankdata(y_true)
        y_pred_rank = stats.rankdata(y_pred)
        ic, p_value = stats.pearsonr(y_true_rank, y_pred_rank)
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return float(ic), float(p_value)  # type: ignore


def calculate_ic_series(df: pd.DataFrame,
                       true_col: str,
                       pred_col: str,
                       group_col: str = 'datetime',
                       method: str = 'pearson') -> Tuple[pd.DataFrame, dict]:
    """
    Calculate IC series grouped by time.
    
    Args:
        df: DataFrame with predictions and true values
        true_col: Column name for true values
        pred_col: Column name for predicted values
        group_col: Grouping column (usually datetime)
        method: IC calculation method
        
    Returns:
        DataFrame with IC values
    """
    ic_results = []
    
    for group_value, group_df in df.groupby(group_col):
        ic, p_value = calculate_ic(
            group_df[true_col].values,
            group_df[pred_col].values,
            method=method
        )
        
        ic_results.append({
            group_col: group_value,
            'ic': ic,
            'p_value': p_value,
            'count': len(group_df)
        })
    
    ic_df = pd.DataFrame(ic_results)
    
    # Calculate IC statistics
    ic_stats = {
        'mean_ic': ic_df['ic'].mean(),
        'std_ic': ic_df['ic'].std(),
        'ir': ic_df['ic'].mean() / (ic_df['ic'].std() + 1e-8),  # Information ratio
        'ic_positive_ratio': (ic_df['ic'] > 0).mean(),  # Ratio of positive IC
        'significant_ratio': (ic_df['p_value'] < 0.05).mean()  # Ratio of significant IC
    }
    
    return ic_df, ic_stats


def calculate_directional_accuracy(y_true: np.ndarray, 
                                 y_pred: np.ndarray,
                                 threshold: float = 0.0) -> float:
    """
    Calculate directional accuracy.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        threshold: Threshold for direction judgment
        
    Returns:
        Directional accuracy
    """
    # Calculate direction
    true_direction = np.sign(y_true - threshold)
    pred_direction = np.sign(y_pred - threshold)
    
    # Calculate accuracy
    correct = (true_direction == pred_direction)
    return np.mean(correct)


def calculate_sharpe_ratio(returns: np.ndarray,
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe ratio.
    
    Args:
        returns: Return series
        risk_free_rate: Risk-free rate (annualized)
        periods_per_year: Number of periods per year
        
    Returns:
        Sharpe ratio
    """
    returns = np.asarray(returns)
    
    # Calculate excess returns
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # Calculate Sharpe ratio
    if len(excess_returns) < 2:
        return np.nan
        
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns, ddof=1)
    
    if std_excess == 0:
        return np.nan
        
    sharpe = mean_excess / std_excess * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_max_drawdown(returns: np.ndarray) -> Tuple[float, int, int]:
    """
    Calculate maximum drawdown.
    
    Args:
        returns: Return series
        
    Returns:
        max_drawdown: Maximum drawdown
        peak_idx: Peak index
        trough_idx: Trough index
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(cum_returns)
    
    # Calculate drawdown
    drawdown = (cum_returns - running_max) / running_max
    
    # Find maximum drawdown
    max_drawdown = np.min(drawdown)
    trough_idx = np.argmin(drawdown)
    
    # Find peak index
    peak_idx = np.argmax(cum_returns[:trough_idx+1])
    
    return max_drawdown, peak_idx, trough_idx  # type: ignore 