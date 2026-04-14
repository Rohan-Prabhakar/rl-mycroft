"""
Test suite for reward shaping components.
Tests individual reward components in isolation.
"""
import pytest
import numpy as np
from orchestration.reward_shaping import (
    calculate_sharpe_ratio,
    calculate_drawdown_penalty,
    calculate_transaction_cost_penalty,
    calculate_turnover_penalty,
    calculate_risk_adjusted_return,
    RewardConfig
)


class TestRewardConfig:
    """Test reward configuration constants."""

    def test_config_defaults(self):
        """Test default configuration values."""
        assert RewardConfig.SHARPE_COEFF > 0
        assert RewardConfig.DRAWDOWN_COEFF > 0
        assert RewardConfig.TRANSACTION_COST_COEFF > 0
        assert RewardConfig.TURNOVER_COEFF >= 0
        assert RewardConfig.RISK_FREE_RATE >= 0


class TestSharpeRatio:
    """Test Sharpe ratio calculation."""

    def test_positive_returns(self):
        """Test Sharpe ratio with positive returns."""
        returns = np.array([0.05, 0.03, 0.04, 0.02])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe > 0
        assert np.isfinite(sharpe)

    def test_negative_returns(self):
        """Test Sharpe ratio with negative returns."""
        returns = np.array([-0.05, -0.03, -0.04, -0.02])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert sharpe < 0
        assert np.isfinite(sharpe)

    def test_zero_volatility(self):
        """Test Sharpe ratio with zero volatility (constant returns)."""
        returns = np.array([0.02, 0.02, 0.02, 0.02])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        # Should handle division by zero gracefully
        assert np.isfinite(sharpe) or np.isinf(sharpe)

    def test_single_return(self):
        """Test Sharpe ratio with single return value."""
        returns = np.array([0.05])
        sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        assert np.isfinite(sharpe) or np.isnan(sharpe)  # NaN is acceptable for single value

    def test_with_risk_free_rate(self):
        """Test Sharpe ratio adjusts for risk-free rate."""
        returns = np.array([0.10, 0.08, 0.12, 0.09])
        sharpe_no_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_with_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.05)
        
        # Higher risk-free rate should reduce Sharpe when returns > rf
        assert sharpe_with_rf < sharpe_no_rf


class TestDrawdownPenalty:
    """Test drawdown penalty calculation."""

    def test_no_drawdown(self):
        """Test penalty when no drawdown occurs."""
        portfolio_values = np.array([100, 110, 120, 130])
        penalty = calculate_drawdown_penalty(portfolio_values)
        assert penalty == 0.0

    def test_simple_drawdown(self):
        """Test penalty with simple drawdown."""
        portfolio_values = np.array([100, 90, 85, 95])
        penalty = calculate_drawdown_penalty(portfolio_values)
        assert penalty > 0
        assert penalty <= 1.0  # Normalized penalty

    def test_multiple_drawdowns(self):
        """Test penalty with multiple drawdown periods."""
        portfolio_values = np.array([100, 80, 90, 70, 85, 60])
        penalty = calculate_drawdown_penalty(portfolio_values)
        assert penalty > 0
        assert np.isfinite(penalty)

    def test_flat_portfolio(self):
        """Test penalty with flat portfolio values."""
        portfolio_values = np.array([100, 100, 100, 100])
        penalty = calculate_drawdown_penalty(portfolio_values)
        assert penalty == 0.0

    def test_two_points(self):
        """Test penalty with only two data points."""
        portfolio_values = np.array([100, 90])
        penalty = calculate_drawdown_penalty(portfolio_values)
        assert penalty >= 0


class TestTransactionCostPenalty:
    """Test transaction cost penalty calculation."""

    def test_no_trading(self):
        """Test penalty when no trading occurs."""
        current_weights = np.array([0.5, 0.3, 0.2])
        previous_weights = np.array([0.5, 0.3, 0.2])
        penalty = calculate_transaction_cost_penalty(
            current_weights, previous_weights, cost_rate=0.001
        )
        assert penalty == 0.0

    def test_large_turnover(self):
        """Test penalty with large portfolio turnover."""
        current_weights = np.array([0.8, 0.1, 0.1])
        previous_weights = np.array([0.1, 0.1, 0.8])
        penalty = calculate_transaction_cost_penalty(
            current_weights, previous_weights, cost_rate=0.001
        )
        assert penalty > 0
        assert np.isfinite(penalty)

    def test_cost_rate_scaling(self):
        """Test that penalty scales with cost rate."""
        current_weights = np.array([0.9, 0.1])
        previous_weights = np.array([0.1, 0.9])
        
        penalty_low = calculate_transaction_cost_penalty(
            current_weights, previous_weights, cost_rate=0.0001
        )
        penalty_high = calculate_transaction_cost_penalty(
            current_weights, previous_weights, cost_rate=0.01
        )
        
        assert penalty_high > penalty_low

    def test_zero_cost_rate(self):
        """Test penalty with zero cost rate."""
        current_weights = np.array([0.9, 0.1])
        previous_weights = np.array([0.1, 0.9])
        penalty = calculate_transaction_cost_penalty(
            current_weights, previous_weights, cost_rate=0.0
        )
        assert penalty == 0.0


class TestTurnoverPenalty:
    """Test turnover penalty calculation."""

    def test_no_turnover(self):
        """Test penalty when no turnover."""
        weights = np.array([0.5, 0.3, 0.2])
        prev_weights = np.array([0.5, 0.3, 0.2])
        penalty = calculate_turnover_penalty(weights, prev_weights)
        assert penalty == 0.0

    def test_complete_turnover(self):
        """Test penalty with complete turnover."""
        weights = np.array([0.0, 0.5, 0.5])
        prev_weights = np.array([0.5, 0.5, 0.0])
        penalty = calculate_turnover_penalty(weights, prev_weights)
        assert penalty > 0

    def test_penalty_bounds(self):
        """Test that turnover penalty is within bounds."""
        weights = np.random.dirichlet([1, 1, 1])
        prev_weights = np.random.dirichlet([1, 1, 1])
        penalty = calculate_turnover_penalty(weights, prev_weights)
        
        assert 0.0 <= penalty <= 2.0  # Max turnover is 2 (complete flip)


class TestRiskAdjustedReturn:
    """Test combined risk-adjusted return calculation."""

    def test_basic_calculation(self):
        """Test basic risk-adjusted return."""
        returns = np.array([0.05, 0.03, -0.02, 0.04])
        portfolio_values = np.array([100, 105, 108, 106, 110])
        current_weights = np.array([0.5, 0.3, 0.2])
        previous_weights = np.array([0.4, 0.4, 0.2])
        
        reward = calculate_risk_adjusted_return(
            returns=returns,
            portfolio_values=portfolio_values,
            current_weights=current_weights,
            previous_weights=previous_weights,
            config=RewardConfig()
        )
        
        assert np.isfinite(reward)

    def test_high_drawdown_impact(self):
        """Test that high drawdown reduces reward."""
        returns_good = np.array([0.05, 0.05, 0.05])
        returns_bad = np.array([0.05, -0.20, 0.05])
        
        portfolio_good = np.array([100, 105, 110, 115])
        portfolio_bad = np.array([100, 105, 84, 88])
        
        weights = np.array([0.5, 0.5])
        
        reward_good = calculate_risk_adjusted_return(
            returns=returns_good,
            portfolio_values=portfolio_good,
            current_weights=weights,
            previous_weights=weights,
            config=RewardConfig()
        )
        
        reward_bad = calculate_risk_adjusted_return(
            returns=returns_bad,
            portfolio_values=portfolio_bad,
            current_weights=weights,
            previous_weights=weights,
            config=RewardConfig()
        )
        
        # Good portfolio should have higher reward
        assert reward_good > reward_bad

    def test_high_transaction_costs_impact(self):
        """Test that high transaction costs reduce reward."""
        returns = np.array([0.05, 0.03, 0.04])
        portfolio_values = np.array([100, 105, 108, 111])
        
        weights_stable = np.array([0.5, 0.3, 0.2])
        weights_volatile = np.array([0.1, 0.1, 0.8])
        
        reward_stable = calculate_risk_adjusted_return(
            returns=returns,
            portfolio_values=portfolio_values,
            current_weights=weights_stable,
            previous_weights=weights_stable,
            config=RewardConfig()
        )
        
        reward_volatile = calculate_risk_adjusted_return(
            returns=returns,
            portfolio_values=portfolio_values,
            current_weights=weights_volatile,
            previous_weights=weights_stable,
            config=RewardConfig()
        )
        
        # Stable portfolio should have higher reward due to lower transaction costs
        assert reward_stable > reward_volatile


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
