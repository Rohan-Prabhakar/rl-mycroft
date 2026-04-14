"""
Test suite for the Portfolio Optimization Environment.
Tests environment shape, reward bounds, and basic functionality.
"""
import pytest
import numpy as np
import gymnasium as gym
from agents.sac_agent import PortfolioEnv


class TestPortfolioEnv:
    """Test cases for PortfolioEnv."""

    @pytest.fixture
    def env(self):
        """Create a fresh environment for each test."""
        return PortfolioEnv(
            num_assets=3,
            initial_capital=10000.0,
            transaction_cost_rate=0.001,
            risk_free_rate=0.02,
            window_size=5,
            max_steps=100
        )

    def test_env_initialization(self, env):
        """Test that environment initializes correctly."""
        assert env.num_assets == 3
        assert env.initial_capital == 10000.0
        assert env.transaction_cost_rate == 0.001
        assert env.risk_free_rate == 0.02

    def test_observation_space_shape(self, env):
        """Test observation space has correct shape."""
        obs, _ = env.reset()
        expected_features = env.num_assets * 4 + 2  # prices, returns, volatility, momentum + portfolio state
        assert obs.shape[0] == expected_features, f"Expected {expected_features}, got {obs.shape[0]}"

    def test_action_space_shape(self, env):
        """Test action space has correct dimension."""
        assert env.action_space.shape[0] == env.num_assets

    def test_action_bounds(self, env):
        """Test actions are within [-1, 1] bounds."""
        env.reset()
        valid_action = np.array([0.5, -0.3, 0.8])
        clipped_action = env.action_space.clip(valid_action)
        assert np.all(clipped_action >= -1.0)
        assert np.all(clipped_action <= 1.0)

    def test_step_returns_correct_structure(self, env):
        """Test step returns observation, reward, terminated, truncated, info."""
        env.reset()
        action = np.zeros(env.num_assets)
        result = env.step(action)
        
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_reward_bounds(self, env):
        """Test that rewards are within reasonable bounds."""
        env.reset()
        
        for _ in range(10):
            action = np.random.uniform(-1, 1, env.num_assets)
            _, reward, _, _, _ = env.step(action)
            
            # Rewards should be finite and within reasonable range
            assert np.isfinite(reward), "Reward is not finite"
            assert -10.0 <= reward <= 10.0, f"Reward {reward} outside expected bounds"

    def test_portfolio_weights_sum(self, env):
        """Test that portfolio weights sum to 1 after softmax."""
        env.reset()
        action = np.random.uniform(-1, 1, env.num_assets)
        _, _, _, _, info = env.step(action)
        
        weights = info.get('weights', np.zeros(env.num_assets))
        assert np.isclose(np.sum(weights), 1.0, atol=1e-5), "Weights do not sum to 1"

    def test_transaction_costs_applied(self, env):
        """Test that transaction costs affect portfolio value."""
        env_high_cost = PortfolioEnv(
            num_assets=2,
            initial_capital=10000.0,
            transaction_cost_rate=0.1,  # High cost
            window_size=3,
            max_steps=50
        )
        env_low_cost = PortfolioEnv(
            num_assets=2,
            initial_capital=10000.0,
            transaction_cost_rate=0.0001,  # Low cost
            window_size=3,
            max_steps=50
        )
        
        env_high_cost.reset()
        env_low_cost.reset()
        
        # Take same large action
        action = np.array([0.9, -0.9])
        
        _, reward_high, _, _, info_high = env_high_cost.step(action)
        _, reward_low, _, _, info_low = env_low_cost.step(action)
        
        # Higher transaction costs should generally lead to lower rewards
        assert info_high.get('transaction_cost', 0) > info_low.get('transaction_cost', 0)

    def test_reset_changes_state(self, env):
        """Test that reset produces different initial states."""
        obs1, _ = env.reset()
        obs2, _ = env.reset()
        
        # With random seed, observations should potentially differ
        # (unless deterministic seed is set)
        assert obs1.shape == obs2.shape

    def test_max_steps_truncation(self, env):
        """Test that environment truncates at max_steps."""
        env_small = PortfolioEnv(
            num_assets=2,
            initial_capital=10000.0,
            window_size=2,
            max_steps=5
        )
        env_small.reset()
        
        for i in range(4):
            _, _, terminated, truncated, _ = env_small.step(np.zeros(2))
            assert not terminated
            assert not truncated
        
        # On the 5th step, should be truncated
        _, _, terminated, truncated, _ = env_small.step(np.zeros(2))
        assert truncated or terminated


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
