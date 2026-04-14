"""Reward Shaping Module for Portfolio RL.

This module provides shared reward shaping constants and functions
used by both the trading environment and the API layer.

The reward function is designed to encourage:
- High risk-adjusted returns (Sharpe ratio)
- Low transaction costs
- Diversified portfolios
- Stable allocation changes (low turnover)

Constants defined here are imported by:
- environments/portfolio_env.py (for reward calculation)
- api/main.py (for risk metric calculation)
"""

import math
from typing import List, Optional, Tuple

# ============================================================================
# Shared Constants
# ============================================================================

# Market parameters
RISK_FREE_RATE = 0.02  # Annual risk-free rate (2%)
TRADING_DAYS_PER_YEAR = 252

# Transaction costs
TRANSACTION_COST_RATE = 0.001  # 0.1% per trade
TURNOVER_PENALTY_FACTOR = 0.5  # Penalty multiplier for turnover

# Portfolio constraints
MAX_POSITION_WEIGHT = 0.4  # Maximum weight in single asset (40%)
MIN_POSITION_WEIGHT = 0.0  # Minimum weight (no short selling)
TARGET_TURNOVER = 0.1  # Target maximum turnover per rebalance (10%)

# Reward shaping targets
SHARPE_RATIO_TARGET = 1.5  # Target annual Sharpe ratio
VOLATILITY_TARGET = 0.15  # Target annual volatility (15%)
DIVERSIFICATION_TARGET = 0.2  # Target Herfindahl index (lower = more diversified)

# Reward weights
REWARD_RETURN_WEIGHT = 1.0
REWARD_RISK_WEIGHT = 0.5
REWARD_TRANSACTION_COST_WEIGHT = 0.3
REWARD_TURNOVER_WEIGHT = 0.2
REWARD_CONSTRAINT_WEIGHT = 0.5


# ============================================================================
# Reward Calculation Functions
# ============================================================================


def calculate_sharpe_ratio(
    returns: List[float],
    volatilities: List[float],
    weights: List[float],
    risk_free_rate: float = RISK_FREE_RATE,
) -> float:
    """Calculate portfolio Sharpe ratio.

    Args:
        returns: Asset returns
        volatilities: Asset volatilities
        weights: Portfolio weights
        risk_free_rate: Annual risk-free rate

    Returns:
        Annualized Sharpe ratio
    """
    if not weights or not returns or not volatilities:
        return 0.0

    # Portfolio return (weighted sum)
    port_return = sum(w * r for w, r in zip(weights, returns))

    # Portfolio volatility (simplified: ignoring correlations)
    port_variance = sum((w * v) ** 2 for w, v in zip(weights, volatilities))
    port_volatility = math.sqrt(port_variance)

    if port_volatility == 0:
        return 0.0

    # Annualize
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR
    annual_return = port_return * TRADING_DAYS_PER_YEAR
    annual_vol = port_volatility * math.sqrt(TRADING_DAYS_PER_YEAR)

    sharpe = (annual_return - risk_free_rate) / annual_vol
    return sharpe


def calculate_transaction_cost(
    current_weights: List[float],
    new_weights: List[float],
    transaction_cost_rate: float = TRANSACTION_COST_RATE,
) -> float:
    """Calculate transaction cost from rebalancing.

    Args:
        current_weights: Current portfolio weights
        new_weights: New target weights
        transaction_cost_rate: Cost rate per unit traded

    Returns:
        Total transaction cost as fraction of portfolio
    """
    if len(current_weights) != len(new_weights):
        raise ValueError("Weight lists must have same length")

    # Turnover is half the sum of absolute differences
    turnover = sum(abs(nw - cw) for nw, cw in zip(new_weights, current_weights)) / 2.0

    # Transaction cost
    cost = turnover * transaction_cost_rate
    return cost


def calculate_turnover(
    current_weights: List[float],
    new_weights: List[float],
) -> float:
    """Calculate portfolio turnover.

    Args:
        current_weights: Current portfolio weights
        new_weights: New target weights

    Returns:
        Turnover as fraction of portfolio
    """
    if len(current_weights) != len(new_weights):
        raise ValueError("Weight lists must have same length")

    turnover = sum(abs(nw - cw) for nw, cw in zip(new_weights, current_weights)) / 2.0
    return turnover


def calculate_concentration(weights: List[float]) -> float:
    """Calculate portfolio concentration (Herfindahl index).

    Args:
        weights: Portfolio weights

    Returns:
        Herfindahl index (sum of squared weights)
        Lower values indicate better diversification
    """
    return sum(w**2 for w in weights)


def calculate_constraint_penalty(
    weights: List[float],
    max_weight: float = MAX_POSITION_WEIGHT,
    min_weight: float = MIN_POSITION_WEIGHT,
) -> float:
    """Calculate penalty for constraint violations.

    Args:
        weights: Portfolio weights
        max_weight: Maximum allowed weight
        min_weight: Minimum allowed weight

    Returns:
        Penalty value (higher = more violations)
    """
    penalty = 0.0

    for w in weights:
        if w > max_weight:
            penalty += (w - max_weight) ** 2
        if w < min_weight:
            penalty += (min_weight - w) ** 2

    # Penalty for weights not summing to ~1
    weight_sum = sum(weights)
    if abs(weight_sum - 1.0) > 0.01:
        penalty += (weight_sum - 1.0) ** 2

    return penalty


def calculate_reward(
    returns: List[float],
    volatilities: List[float],
    current_weights: List[float],
    new_weights: List[float],
    prev_weights: Optional[List[float]] = None,
    # Override defaults if needed
    risk_free_rate: float = RISK_FREE_RATE,
    transaction_cost_rate: float = TRANSACTION_COST_RATE,
    sharpe_target: float = SHARPE_RATIO_TARGET,
    turnover_target: float = TARGET_TURNOVER,
    diversification_target: float = DIVERSIFICATION_TARGET,
) -> Tuple[float, dict]:
    """Calculate comprehensive reward with shaping.

    This is the main reward function used by the RL environment.
    It combines multiple objectives:
    - Risk-adjusted returns
    - Low transaction costs
    - Low turnover
    - Diversification
    - Constraint satisfaction

    Args:
        returns: Asset returns
        volatilities: Asset volatilities
        current_weights: Current portfolio weights
        new_weights: New target weights
        prev_weights: Previous weights (optional, for turnover calc)
        risk_free_rate: Risk-free rate override
        transaction_cost_rate: Transaction cost override
        sharpe_target: Target Sharpe ratio
        turnover_target: Target turnover
        diversification_target: Target diversification

    Returns:
        Tuple of (total_reward, components_dict)
        where components_dict contains individual reward terms
    """
    # 1. Risk-adjusted return (Sharpe ratio based)
    sharpe = calculate_sharpe_ratio(returns, volatilities, new_weights, risk_free_rate)
    # Normalize: reward is positive if Sharpe > 0, scaled by target
    return_reward = REWARD_RETURN_WEIGHT * (sharpe / sharpe_target)

    # 2. Risk penalty (deviation from target volatility)
    port_vol = math.sqrt(sum((w * v) ** 2 for w, v in zip(new_weights, volatilities)))
    annual_vol = port_vol * math.sqrt(TRADING_DAYS_PER_YEAR)
    vol_deviation = abs(annual_vol - VOLATILITY_TARGET) / VOLATILITY_TARGET
    risk_penalty = -REWARD_RISK_WEIGHT * vol_deviation

    # 3. Transaction cost
    tx_cost = calculate_transaction_cost(
        current_weights, new_weights, transaction_cost_rate
    )
    cost_penalty = -REWARD_TRANSACTION_COST_WEIGHT * tx_cost * 100  # Scale up

    # 4. Turnover penalty
    if prev_weights is not None:
        turnover = calculate_turnover(prev_weights, new_weights)
    else:
        turnover = calculate_turnover(current_weights, new_weights)

    if turnover > turnover_target:
        turnover_penalty = -REWARD_TURNOVER_WEIGHT * (turnover - turnover_target)
    else:
        turnover_penalty = 0.0  # No penalty if below target

    # 5. Diversification reward
    concentration = calculate_concentration(new_weights)
    # Lower concentration = better diversification
    if concentration < diversification_target:
        div_reward = REWARD_CONSTRAINT_WEIGHT * (diversification_target - concentration)
    else:
        div_reward = -REWARD_CONSTRAINT_WEIGHT * (concentration - diversification_target)

    # 6. Constraint violation penalty
    constraint_penalty = -calculate_constraint_penalty(
        new_weights, MAX_POSITION_WEIGHT, MIN_POSITION_WEIGHT
    )

    # Total reward
    total_reward = (
        return_reward + risk_penalty + cost_penalty + turnover_penalty + div_reward + constraint_penalty
    )

    components = {
        "return_reward": return_reward,
        "risk_penalty": risk_penalty,
        "cost_penalty": cost_penalty,
        "turnover_penalty": turnover_penalty,
        "diversification_reward": div_reward,
        "constraint_penalty": constraint_penalty,
        "sharpe_ratio": sharpe,
        "portfolio_volatility": annual_vol,
        "transaction_cost": tx_cost,
        "turnover": turnover,
        "concentration": concentration,
    }

    return total_reward, components


def normalize_weights(
    raw_weights: List[float],
    min_weight: float = MIN_POSITION_WEIGHT,
    max_weight: float = MAX_POSITION_WEIGHT,
) -> List[float]:
    """Normalize weights to valid probability distribution.

    Applies clipping and renormalization to ensure:
    - All weights are within [min_weight, max_weight]
    - Weights sum to 1.0

    Args:
        raw_weights: Raw weights from model
        min_weight: Minimum allowed weight
        max_weight: Maximum allowed weight

    Returns:
        Normalized weights
    """
    # Clip to valid range
    clipped = [max(min_weight, min(max_weight, w)) for w in raw_weights]

    # Ensure sum is 1.0
    total = sum(clipped)
    if total == 0:
        # Fallback to equal weights
        n = len(clipped)
        return [1.0 / n] * n

    normalized = [w / total for w in clipped]

    # Re-clip after normalization (may be needed due to rounding)
    final = [max(min_weight, min(max_weight, w)) for w in normalized]

    # Final renormalization
    final_total = sum(final)
    if final_total != 1.0 and final_total > 0:
        final = [w / final_total for w in final]

    return final


# ============================================================================
# Utility Functions
# ============================================================================


def get_reward_config() -> dict:
    """Get current reward shaping configuration.

    Returns:
        Dictionary of all reward shaping parameters
    """
    return {
        "risk_free_rate": RISK_FREE_RATE,
        "transaction_cost_rate": TRANSACTION_COST_RATE,
        "turnover_penalty_factor": TURNOVER_PENALTY_FACTOR,
        "max_position_weight": MAX_POSITION_WEIGHT,
        "min_position_weight": MIN_POSITION_WEIGHT,
        "target_turnover": TARGET_TURNOVER,
        "sharpe_ratio_target": SHARPE_RATIO_TARGET,
        "volatility_target": VOLATILITY_TARGET,
        "diversification_target": DIVERSIFICATION_TARGET,
        "reward_weights": {
            "return": REWARD_RETURN_WEIGHT,
            "risk": REWARD_RISK_WEIGHT,
            "transaction_cost": REWARD_TRANSACTION_COST_WEIGHT,
            "turnover": REWARD_TURNOVER_WEIGHT,
            "constraint": REWARD_CONSTRAINT_WEIGHT,
        },
    }


def validate_reward_config() -> Tuple[bool, List[str]]:
    """Validate reward shaping configuration.

    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []

    if RISK_FREE_RATE < 0:
        issues.append("RISK_FREE_RATE must be non-negative")

    if TRANSACTION_COST_RATE < 0:
        issues.append("TRANSACTION_COST_RATE must be non-negative")

    if MAX_POSITION_WEIGHT <= MIN_POSITION_WEIGHT:
        issues.append("MAX_POSITION_WEIGHT must be > MIN_POSITION_WEIGHT")

    if MAX_POSITION_WEIGHT > 1.0:
        issues.append("MAX_POSITION_WEIGHT must be <= 1.0")

    if MIN_POSITION_WEIGHT < 0:
        issues.append("MIN_POSITION_WEIGHT must be >= 0")

    if SHARPE_RATIO_TARGET <= 0:
        issues.append("SHARPE_RATIO_TARGET must be positive")

    if VOLATILITY_TARGET <= 0:
        issues.append("VOLATILITY_TARGET must be positive")

    is_valid = len(issues) == 0
    return is_valid, issues


if __name__ == "__main__":
    # Test reward calculation
    test_returns = [0.01, 0.02, -0.01, 0.015, 0.005]
    test_vols = [0.15, 0.20, 0.18, 0.12, 0.25]
    test_current = [0.2, 0.2, 0.2, 0.2, 0.2]
    test_new = [0.25, 0.15, 0.20, 0.25, 0.15]

    reward, components = calculate_reward(
        returns=test_returns,
        volatilities=test_vols,
        current_weights=test_current,
        new_weights=test_new,
        prev_weights=test_current,
    )

    print("Reward Components:")
    for key, value in components.items():
        print(f"  {key}: {value:.6f}")
    print(f"\nTotal Reward: {reward:.6f}")

    print("\nNormalized weights test:")
    raw = [0.5, 0.3, 0.8, -0.1, 0.2]
    normalized = normalize_weights(raw)
    print(f"  Raw: {raw}")
    print(f"  Normalized: {normalized}")
    print(f"  Sum: {sum(normalized):.6f}")
