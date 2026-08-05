"""Logical Error Rate (LER) fitting utilities.

This module provides utilities for computing LER from error rate data
using log-linear fitting as described in the paper.
"""

from typing import List, Optional, Tuple

import numpy as np


# Minimum round thresholds for different datasets
LER_FIT_MIN_ROUND_WILLOW = 10
LER_FIT_MIN_ROUND_SYCAMORE = 3
LER_FIT_MIN_ROUND_SI1000 = 3


def ler_fit(
    rounds: np.ndarray,
    error_rates: np.ndarray,
    min_round: Optional[int] = None,
) -> Tuple[float, float, float, float]:
    """Compute Logical Error Rate (LER) via log-linear regression.
    
    Fits log(fidelity) = intercept + slope * rounds, then computes
    epsilon = (1 - exp(slope)) / 2.
    
    Args:
        rounds: Array of round counts.
        error_rates: Array of corresponding error rates.
        min_round: Optional minimum round for filtering (default: no filtering).
        
    Returns:
        Tuple of (epsilon, r_squared, slope, intercept):
            - epsilon: The computed LER, clipped to [0, 0.5].
            - r_squared: R² of the linear fit (0.0 if < 2 points).
            - slope: Slope of log-fidelity fit.
            - intercept: Intercept of log-fidelity fit.
    """
    rounds = rounds.astype(np.float64)
    error_rates = error_rates.astype(np.float64)

    # Handle empty input
    if len(rounds) == 0:
        return 0.0, 0.0, 0.0, 0.0

    # Apply minimum round filter if specified
    if min_round is not None:
        mask = rounds >= min_round
        if mask.any():
            rounds = rounds[mask]
            error_rates = error_rates[mask]

    # Compute log-fidelity
    fidelity = 1 - 2 * error_rates
    fidelity = np.clip(fidelity, 1e-10, 1.0)
    log_fidelity = np.log(fidelity)

    # Handle single point case
    if len(rounds) < 2:
        epsilon = float(np.clip(error_rates.mean(), 0.0, 0.5))
        return epsilon, 0.0, 0.0, float(log_fidelity[0]) if len(log_fidelity) > 0 else 0.0

    # Linear regression on log-fidelity
    slope, intercept = np.polyfit(rounds, log_fidelity, deg=1)
    r = float(np.corrcoef(rounds, log_fidelity)[0, 1])
    r_squared = r * r

    # Compute epsilon from slope
    epsilon = float(np.clip((1 - np.exp(slope)) / 2, 0.0, 0.5))

    return epsilon, r_squared, float(slope), float(intercept)


def ler_fit_with_std(
    rounds: np.ndarray,
    error_rates: np.ndarray,
    min_round: int = 10,
) -> Tuple[float, float, float, float]:
    """Compute LER with standard error on intercept.
    
    Extended version of ler_fit that also computes the standard error
    of the intercept estimate.
    
    Args:
        rounds: Array of round counts.
        error_rates: Array of corresponding error rates.
        min_round: Minimum round for filtering (default: 10).
        
    Returns:
        Tuple of (epsilon, r_squared, intercept, std_intercept):
            - epsilon: The computed LER, clipped to [0, 0.5].
            - r_squared: R² of the linear fit.
            - intercept: Intercept of log-fidelity fit.
            - std_intercept: Standard error of intercept estimate.
    """
    rounds = rounds.astype(np.float64)
    error_rates = error_rates.astype(np.float64)

    if len(rounds) == 0:
        return 0.0, 0.0, 0.0, 0.0

    mask = rounds >= min_round
    if mask.any():
        rounds = rounds[mask]
        error_rates = error_rates[mask]

    fidelity = 1 - 2 * error_rates
    fidelity = np.clip(fidelity, 1e-10, 1.0)
    log_fidelity = np.log(fidelity)

    if len(rounds) < 2:
        epsilon = float(np.clip(error_rates.mean(), 0.0, 0.5))
        return epsilon, 0.0, float(log_fidelity[0]) if len(log_fidelity) > 0 else 0.0, 0.0

    slope, intercept = np.polyfit(rounds, log_fidelity, deg=1)
    r = float(np.corrcoef(rounds, log_fidelity)[0, 1])
    r_squared = r * r
    epsilon = float(np.clip((1 - np.exp(slope)) / 2, 0.0, 0.5))

    # Compute standard error of intercept
    n = len(rounds)
    residuals = log_fidelity - (intercept + slope * rounds)
    s = float(np.sqrt(np.sum(residuals**2) / max(1, n - 2)))
    x_mean = float(np.mean(rounds))
    denom = float(np.sum((rounds - x_mean) ** 2))
    std_intercept = float(s * np.sqrt(1 / n + (x_mean**2 / denom if denom > 0 else 0.0)))

    return epsilon, r_squared, float(intercept), std_intercept


def filter_for_ler_fit(
    rounds: List[int],
    error_rates: List[float],
    min_round: int,
    max_error_rate: float,
) -> Tuple[List[int], List[float]]:
    """Filter rounds and error rates for LER fitting.
    
    Args:
        rounds: List of round counts.
        error_rates: List of corresponding error rates.
        min_round: Minimum round to include.
        max_error_rate: Maximum error rate to include.
        
    Returns:
        Tuple of (filtered_rounds, filtered_error_rates).
    """
    fit_rounds = []
    fit_error_rates = []
    for r, err in zip(rounds, error_rates):
        if r >= min_round and err <= max_error_rate:
            fit_rounds.append(r)
            fit_error_rates.append(err)
    return fit_rounds, fit_error_rates
