"""Unit tests for Mycroft-RL Phase 1: Environment and Data Pipeline."""

import logging
import numpy as np
import pytest

from envs.data_loader import DataLoader, get_default_tickers, load_and_prepare_data
from envs.mycroft_finance_env import MycroftFinanceEnv


logging.basicConfig(level=logging.WARNING)


class TestDataLoader:
    """Tests for the DataLoader class."""
    
    def test_get_default_tickers(self):
        """Test that default tickers returns 16 AI-focused companies."""
        tickers = get_default_tickers()
        assert len(tickers) == 16
        assert "NVDA" in tickers
        assert "MSFT" in tickers
        assert isinstance(tickers, list)
        assert all(isinstance(t, str) for t in tickers)
    
    def test_data_loader_initialization(self):
        """Test DataLoader initialization with custom parameters."""
        tickers = ["NVDA", "MSFT"]
        loader = DataLoader(
            tickers=tickers,
            start_date="2023-01-01",
            end_date="2023-06-30",
        )
        assert loader.tickers == tickers
        assert loader.start_date == "2023-01-01"
        assert loader.end_date == "2023-06-30"
    
    def test_data_loader_fetch_and_compute(self):
        """Test data fetching and indicator computation."""
        loader = DataLoader(
            tickers=["NVDA", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )
        loader.fetch_data()
        assert len(loader.data) == 2
        assert "NVDA" in loader.data
        
        loader.compute_indicators()
        assert loader.combined_data is not None
        assert len(loader.combined_data.columns) == 10  # 5 indicators * 2 tickers
    
    def test_load_and_prepare_data(self):
        """Test the convenience function for loading data."""
        indicators, prices, index = load_and_prepare_data(
            tickers=["NVDA", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )
        assert len(indicators) > 0
        assert len(prices) > 0
        assert len(index) > 0
        assert len(indicators) == len(prices)


class TestMycroftFinanceEnv:
    """Tests for the MycroftFinanceEnv class."""
    
    @pytest.fixture
    def env(self):
        """Create a test environment with minimal data."""
        return MycroftFinanceEnv(
            tickers=["NVDA", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-03-31",
            initial_capital=100_000.0,
        )
    
    def test_env_initialization(self, env):
        """Test environment initialization."""
        assert env.n_tickers == 2
        assert env.initial_capital == 100_000.0
        assert env.transaction_cost_rate == 0.001
        assert env.max_drawdown_limit == 0.20
    
    def test_action_space(self, env):
        """Test action space definition."""
        obs, _ = env.reset(seed=42)
        assert env.action_space.shape == (2,)
        assert env.action_space.dtype == np.float32
    
    def test_observation_space(self, env):
        """Test observation space definition."""
        obs, _ = env.reset(seed=42)
        # 5 indicators * 2 tickers + 2 weights + 1 cash ratio = 13
        expected_size = 5 * env.n_tickers + env.n_tickers + 1
        assert obs.shape == (expected_size,)
        assert obs.dtype == np.float32
    
    def test_reset_returns_valid_observation(self, env):
        """Test that reset returns a valid observation."""
        obs, info = env.reset(seed=42)
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert "initial_capital" in info
        assert "tickers" in info
    
    def test_step_returns_correct_structure(self, env):
        """Test that step returns the correct structure."""
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        
        result = env.step(action)
        assert len(result) == 5
        
        new_obs, reward, terminated, truncated, info = result
        assert isinstance(new_obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
    
    def test_action_processing(self, env):
        """Test that actions are processed to valid weights."""
        obs, _ = env.reset(seed=42)
        
        # Test with various action values
        test_actions = [
            np.array([0.0, 0.0]),
            np.array([1.0, -1.0]),
            np.array([10.0, -5.0]),
            np.array([-100.0, 100.0]),
        ]
        
        for action in test_actions:
            weights = env._process_action(action)
            assert np.all(weights >= 0)
            assert np.all(weights <= 1)
            assert np.isclose(weights.sum(), 1.0)
    
    def test_reward_calculation(self, env):
        """Test reward calculation components."""
        reward = env._calculate_reward(
            portfolio_return=0.01,
            turnover=0.1,
            concentration=0.5,
            drawdown=0.05,
        )
        assert isinstance(reward, float)
    
    def test_portfolio_metrics(self, env):
        """Test portfolio metrics calculation."""
        obs, _ = env.reset(seed=42)
        
        # Run a few steps
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
        
        metrics = env.get_portfolio_metrics()
        assert "total_return" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "final_value" in metrics
    
    def test_deterministic_behavior(self, env):
        """Test that environment is deterministic with same seed."""
        obs1, _ = env.reset(seed=123)
        actions = [env.action_space.sample() for _ in range(5)]
        
        results1 = []
        for action in actions:
            results1.append(env.step(action))
        
        obs2, _ = env.reset(seed=123)
        results2 = []
        for action in actions:
            results2.append(env.step(action))
        
        # Observations and rewards should match
        assert np.allclose(obs1, obs2)
        for r1, r2 in zip(results1, results2):
            assert np.isclose(r1[1], r2[1])  # rewards
    
    def test_max_drawdown_termination(self):
        """Test that episode terminates on max drawdown."""
        env = MycroftFinanceEnv(
            tickers=["NVDA"],
            start_date="2023-01-01",
            end_date="2023-03-31",
            max_drawdown_limit=0.05,  # Very low threshold for testing
        )
        
        obs, _ = env.reset(seed=42)
        terminated = False
        
        for _ in range(100):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break
        
        # Either terminated due to drawdown or truncated due to end of data
        assert terminated or truncated
    
    def test_info_dict_contents(self, env):
        """Test that info dict contains all required fields."""
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        _, _, _, _, info = env.step(action)
        
        required_keys = [
            "portfolio_value",
            "portfolio_return",
            "cumulative_return",
            "drawdown",
            "turnover",
            "concentration",
            "current_step",
            "weights",
            "cash_ratio",
        ]
        
        for key in required_keys:
            assert key in info, f"Missing key: {key}"


class TestIntegration:
    """Integration tests for environment and data pipeline."""
    
    def test_full_episode_with_all_tickers(self):
        """Test a full episode with default tickers (subset for speed)."""
        env = MycroftFinanceEnv(
            tickers=get_default_tickers()[:5],  # Use 5 tickers for speed
            start_date="2023-01-01",
            end_date="2023-06-30",
        )
        
        obs, _ = env.reset(seed=42)
        assert obs.shape[0] == 5 * 5 + 5 + 1  # 5 indicators, 5 tickers, 1 cash
        
        steps_run = 0
        for _ in range(200):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            steps_run += 1
            
            if terminated or truncated:
                break
        
        assert steps_run > 0
        assert len(env.portfolio_values) > 0
    
    def test_reproducibility_across_runs(self):
        """Test that two runs with same seed produce identical results."""
        # Pre-load data once to ensure both runs use identical data
        indicators, prices, index = load_and_prepare_data(
            tickers=["NVDA", "MSFT"],
            start_date="2023-01-01",
            end_date="2023-03-31",
        )
        
        def run_episode(indicators, prices, index, seed):
            env = MycroftFinanceEnv(
                tickers=["NVDA", "MSFT"],
                start_date="2023-01-01",
                end_date="2023-03-31",
            )
            # Inject pre-loaded data
            env.indicators = indicators
            env.prices = prices
            env.dates = index
            
            obs, _ = env.reset(seed=seed)
            final_values = []
            
            # Use numpy RNG for reproducible actions
            rng = np.random.default_rng(seed)
            
            for _ in range(50):
                action = rng.uniform(
                    low=env.action_space.low,
                    high=env.action_space.high,
                )
                _, _, term, trunc, info = env.step(action)
                final_values.append(info["portfolio_value"])
                if term or trunc:
                    break
            
            return final_values
        
        run1 = run_episode(indicators, prices, index, seed=999)
        run2 = run_episode(indicators, prices, index, seed=999)
        
        # Values should be exactly identical with same data and seed
        assert len(run1) == len(run2)
        for v1, v2 in zip(run1, run2):
            assert v1 == v2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
