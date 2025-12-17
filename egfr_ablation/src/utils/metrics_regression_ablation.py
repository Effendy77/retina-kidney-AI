import numpy as np


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error.
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
    
    Returns:
        RMSE (float). Returns NaN if arrays empty or contain NaN/inf.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        return np.nan
    
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """
    Mean Absolute Error.
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
    
    Returns:
        MAE (float). Returns NaN if arrays empty or contain NaN/inf.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        return np.nan
    
    return np.mean(np.abs(y_true - y_pred))


def r2_score(y_true, y_pred):
    """
    Coefficient of Determination (R²).
    
    Args:
        y_true: ground truth values
        y_pred: predicted values
    
    Returns:
        R² (float, typically in range [-inf, 1]). 
        Returns 0.0 if ss_tot == 0 or arrays empty.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        return 0.0

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


def mape(y_true, y_pred, eps=1e-8):
    """
    Mean Absolute Percentage Error.
    
    Args:
        y_true: ground truth values (must be non-zero or close to it)
        y_pred: predicted values
        eps: small epsilon to avoid division by zero (default 1e-8)
    
    Returns:
        MAPE (float, in percent). 
        Returns NaN if arrays empty or contain NaN/inf.
        
    Note:
        MAPE can be misleading when y_true values are close to zero.
        Consider using a custom threshold or alternative metric.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    if not (np.isfinite(y_true).all() and np.isfinite(y_pred).all()):
        return np.nan
    
    # Warn if many y_true values are near zero
    near_zero_count = np.sum(np.abs(y_true) < eps * 10)
    if near_zero_count > len(y_true) * 0.1:
        print(f"[WARN] MAPE: {near_zero_count}/{len(y_true)} y_true values < {eps*10}. "
              "MAPE may be unreliable.")

    return np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100
