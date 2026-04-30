"""Black–Scholes call helpers aligned with data/SPY/build_final_training_set.py."""

import numpy as np
from scipy.stats import norm

RISK_FREE_DEFAULT = 0.03


def black_scholes_call(S: float, K: float, T: float, r: float, sigma: float):
    """Return (call_price, delta, gamma, theta, vega) or None if inputs invalid."""
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return None

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    theta = (
        -S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
        - r * K * np.exp(-r * T) * norm.cdf(d2)
    )
    vega = S * norm.pdf(d1) * np.sqrt(T)

    return float(call_price), float(delta), float(gamma), float(theta), float(vega)


def apply_training_style_option_row(row: dict, r: float = RISK_FREE_DEFAULT) -> bool:
    """
    Overwrite call_price, delta, gamma, theta, vega and portfolio_* in row
    using the same scaling as build_final_training_set (portfolio = per-share * 100).

    Returns True if BS was applied, False if inputs were invalid (row unchanged).
    """
    S = float(row["spot_today"])
    K = float(row["strike"])
    T = float(row["T"])
    sigma = float(row["realized_vol_20d"])
    out = black_scholes_call(S, K, T, r, sigma)
    if out is None:
        return False

    call_price, delta, gamma, theta, vega = out
    row["call_price"] = call_price
    row["delta"] = delta
    row["gamma"] = gamma
    row["theta"] = theta
    row["vega"] = vega
    row["portfolio_delta"] = delta * 100.0
    row["portfolio_gamma"] = gamma * 100.0
    row["portfolio_theta"] = theta * 100.0
    row["portfolio_vega"] = vega * 100.0
    row["sigma_next"] = min(sigma * 1.02 + 0.001, 2.5)
    return True
