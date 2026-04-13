"""Custom Gymnasium environment for portfolio optimization.

This module implements a Gymnasium environment for optimizing portfolio
allocation across AI-focused public companies using reinforcement learning.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from envs.data_loader import DataLoader, get_default_tickers, load_and_prepare_data

logger = logging.getLogger(__name__)


class MycroftFinanceEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment for portfolio optimization with SAC.
    
    This environment simulates trading across multiple AI-focused stocks,
    where the agent learns to allocate portfolio weights to maximize
    risk-adjusted returns while respecting various constraints.
    
    State Space:
        - Price returns (n_tickers)
        - Volatility (n_tickers)
        - Volume z-score (n_tickers)
        - RSI (n_tickers)
        - MACD histogram (n_tickers)
        - Current weights (n_tickers)
        - Cash ratio (1)
        Total: 5 * n_tickers + 1
        
    Action Space:
        - Continuous allocation weights for each ticker
        - Actions are transformed to sum to 1 and clipped to [0, 1]
        
    Reward:
        - Risk-adjusted return (daily Sharpe ratio)
        - Minus transaction cost penalty
        - Minus concentration penalty
        - Minus drawdown penalty
        
    Attributes:
        tickers: List of ticker symbols.
        indicators: DataFrame with technical indicators.
        prices: DataFrame with closing prices.
        initial_capital: Starting portfolio value.
        transaction_cost_rate: Cost per unit of turnover.
        max_drawdown_limit: Maximum allowed drawdown (e.g., 0.20 for 20%).
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        initial_capital: float = 1_000_000.0,
        transaction_cost_rate: float = 0.001,
        max_drawdown_limit: float = 0.20,
        risk_free_rate: float = 0.02,
        window_size: int = 20,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initialize the environment.
        
        Args:
            tickers: List of ticker symbols. Defaults to AI-focused companies.
            start_date: Start date for historical data.
            end_date: End date for historical data. Defaults to today.
            initial_capital: Starting portfolio value in dollars.
            transaction_cost_rate: Transaction cost as fraction of trade value.
            max_drawdown_limit: Maximum drawdown before episode termination.
            risk_free_rate: Annual risk-free rate for Sharpe calculation.
            window_size: Window for rolling statistics.
            render_mode: Render mode ('human', 'rgb_array', or None).
        """
        super().__init__()
        
        self.tickers = tickers if tickers is not None else get_default_tickers()
        self.n_tickers = len(self.tickers)
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        self.max_drawdown_limit = max_drawdown_limit
        self.risk_free_rate = risk_free_rate
        self.window_size = window_size
        self.render_mode = render_mode
        
        # Data containers (set during reset/load_data)
        self.indicators: Optional[pd.DataFrame] = None
        self.prices: Optional[pd.DataFrame] = None
        self.dates: Optional[pd.DatetimeIndex] = None
        
        # State tracking
        self.current_step: int = 0
        self.portfolio_value: float = initial_capital
        self.peak_value: float = initial_capital
        self.current_weights: np.ndarray = np.zeros(self.n_tickers)
        self.cash_ratio: float = 1.0
        self.portfolio_values: List[float] = []
        self.returns_history: List[float] = []
        
        # Done flag
        self._terminated: bool = False
        self._truncated: bool = False
        
        # Define action and observation spaces
        self._define_spaces()
        
        logger.info(
            f"Initialized MycroftFinanceEnv with {self.n_tickers} tickers, "
            f"observation space: {self.observation_space.shape}, "
            f"action space: {self.action_space.shape}"
        )
    
    def _define_spaces(self) -> None:
        """Define the action and observation spaces."""
        # Action space: continuous weights for each ticker (will be normalized)
        self.action_space = spaces.Box(
            low=-1.0,  # Can be negative before softmax/normalization
            high=1.0,
            shape=(self.n_tickers,),
            dtype=np.float32,
        )
        
        # Observation space:
        # - 5 indicators per ticker (returns, volatility, volume_zscore, rsi, macd)
        # - current weights per ticker
        # - cash ratio (1)
        n_features = 5 * self.n_tickers + self.n_tickers + 1
        
        # Define reasonable bounds for observations
        obs_low = np.full(n_features, -np.inf, dtype=np.float32)
        obs_high = np.full(n_features, np.inf, dtype=np.float32)
        
        # Set bounds for known ranges
        # Returns: typically [-1, 1]
        obs_low[:self.n_tickers] = -1.0
        obs_high[:self.n_tickers] = 1.0
        
        # Volatility: [0, 1]
        obs_low[self.n_tickers:2*self.n_tickers] = 0.0
        obs_high[self.n_tickers:2*self.n_tickers] = 1.0
        
        # Volume z-score: typically [-5, 5]
        obs_low[2*self.n_tickers:3*self.n_tickers] = -10.0
        obs_high[2*self.n_tickers:3*self.n_tickers] = 10.0
        
        # RSI: [0, 100]
        obs_low[3*self.n_tickers:4*self.n_tickers] = 0.0
        obs_high[3*self.n_tickers:4*self.n_tickers] = 100.0
        
        # MACD: unbounded but typically [-10, 10] for normalized prices
        obs_low[4*self.n_tickers:5*self.n_tickers] = -50.0
        obs_high[4*self.n_tickers:5*self.n_tickers] = 50.0
        
        # Weights: [0, 1]
        obs_low[5*self.n_tickers:6*self.n_tickers] = 0.0
        obs_high[5*self.n_tickers:6*self.n_tickers] = 1.0
        
        # Cash ratio: [0, 1]
        obs_low[-1] = 0.0
        obs_high[-1] = 1.0
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )
    
    def _load_data(self) -> None:
        """Load and prepare data for the environment."""
        logger.info("Loading data for environment...")
        
        indicators, prices, dates = load_and_prepare_data(
            tickers=self.tickers,
            start_date=self.start_date,
            end_date=self.end_date,
        )
        
        self.indicators = indicators
        self.prices = prices
        self.dates = dates
        
        logger.info(f"Loaded {len(dates)} time steps for {self.n_tickers} tickers.")
    
    def _get_observation(self) -> np.ndarray:
        """Construct the current observation vector.
        
        Returns:
            Observation array of shape (5 * n_tickers + n_tickers + 1,).
        """
        if self.indicators is None or self.prices is None:
            raise ValueError("Data not loaded. Call reset() first.")
        
        # Get current row of indicators
        current_date = self.dates[self.current_step]
        row = self.indicators.loc[current_date]
        
        # Extract features for each ticker
        features = []
        
        for ticker in self.tickers:
            # Price returns
            returns = float(row[(ticker, "returns")]) if not np.isnan(row[(ticker, "returns")]) else 0.0
            features.append(returns)
            
            # Volatility
            vol = float(row[(ticker, "volatility")]) if not np.isnan(row[(ticker, "volatility")]) else 0.0
            features.append(vol)
            
            # Volume z-score
            vol_z = float(row[(ticker, "volume_zscore")]) if not np.isnan(row[(ticker, "volume_zscore")]) else 0.0
            features.append(vol_z)
            
            # RSI
            rsi = float(row[(ticker, "rsi")]) if not np.isnan(row[(ticker, "rsi")]) else 50.0
            features.append(rsi)
            
            # MACD histogram
            macd = float(row[(ticker, "macd")]) if not np.isnan(row[(ticker, "macd")]) else 0.0
            features.append(macd)
        
        # Add current weights
        features.extend(self.current_weights.tolist())
        
        # Add cash ratio
        features.append(self.cash_ratio)
        
        obs = np.array(features, dtype=np.float32)
        
        # Clip to observation space bounds
        obs = np.clip(obs, self.observation_space.low, self.observation_space.high)
        
        return obs
    
    def _process_action(self, action: np.ndarray) -> np.ndarray:
        """Process raw action into valid portfolio weights.
        
        Transforms action to weights that:
        - Are non-negative (long-only)
        - Sum to 1.0
        
        Args:
            action: Raw action from agent.
            
        Returns:
            Valid portfolio weights.
        """
        # Apply softmax-like transformation to ensure positive weights summing to 1
        # First, shift to positive range
        shifted = action - action.min() + 1e-8
        
        # Normalize to sum to 1
        weights = shifted / shifted.sum()
        
        # Ensure no single position exceeds a reasonable limit (e.g., 40%)
        max_weight = 0.4
        weights = np.clip(weights, 0.0, max_weight)
        
        # Re-normalize after clipping
        weights = weights / weights.sum()
        
        return weights.astype(np.float32)
    
    def _calculate_reward(
        self,
        portfolio_return: float,
        turnover: float,
        concentration: float,
        drawdown: float,
    ) -> float:
        """Calculate the reward for the current step.
        
        Args:
            portfolio_return: Daily portfolio return.
            turnover: Portfolio turnover (sum of absolute weight changes).
            concentration: Herfindahl index of portfolio weights.
            drawdown: Current drawdown from peak.
            
        Returns:
            Scalar reward value.
        """
        # Risk-adjusted return (daily Sharpe approximation)
        daily_rf = self.risk_free_rate / 252  # Trading days per year
        excess_return = portfolio_return - daily_rf
        
        # Use recent volatility for Sharpe calculation
        if len(self.returns_history) >= self.window_size:
            recent_returns = np.array(self.returns_history[-self.window_size:])
            vol = np.std(recent_returns)
            if vol > 1e-8:
                sharpe = excess_return / vol
            else:
                sharpe = excess_return * 10  # Penalize zero volatility
        else:
            sharpe = excess_return * 10  # Not enough history
        
        # Transaction cost penalty
        transaction_penalty = self.transaction_cost_rate * turnover
        
        # Concentration penalty (encourage diversification)
        # Herfindahl index ranges from 1/n (equal weight) to 1 (single stock)
        concentration_penalty = 0.1 * (concentration - 1.0 / self.n_tickers)
        
        # Drawdown penalty
        drawdown_penalty = 0.5 * drawdown
        
        # Total reward
        reward = sharpe - transaction_penalty - concentration_penalty - drawdown_penalty
        
        return float(reward)
    
    def _update_portfolio(self, new_weights: np.ndarray) -> Tuple[float, float, float]:
        """Update portfolio based on new weights.
        
        Args:
            new_weights: New target portfolio weights.
            
        Returns:
            Tuple of (portfolio_return, turnover, concentration).
        """
        if self.prices is None:
            raise ValueError("Prices not loaded.")
        
        # Get current prices
        current_date = self.dates[self.current_step]
        prev_date = self.dates[self.current_step - 1] if self.current_step > 0 else current_date
        
        current_prices = self.prices.loc[current_date].values
        prev_prices = self.prices.loc[prev_date].values
        
        # Calculate asset returns
        asset_returns = (current_prices - prev_prices) / (prev_prices + 1e-8)
        
        # Calculate turnover (L1 distance between old and new weights)
        turnover = np.sum(np.abs(new_weights - self.current_weights))
        
        # Update weights
        self.current_weights = new_weights.copy()
        
        # Assume we rebalance instantly, so portfolio return is based on new weights
        # In reality, there would be a lag, but for simplicity we use new weights
        portfolio_return = float(np.dot(self.current_weights, asset_returns))
        
        # Update portfolio value
        prev_value = self.portfolio_value
        self.portfolio_value *= (1 + portfolio_return)
        
        # Account for transaction costs
        transaction_cost = self.transaction_cost_rate * turnover * prev_value
        self.portfolio_value -= transaction_cost
        
        # Update cash ratio (simplified: assume any weight change requires cash adjustment)
        # In this model, we're always fully invested, so cash_ratio represents
        # the portion not allocated to the top positions
        self.cash_ratio = 0.0  # Fully invested model
        
        # Track portfolio values
        self.portfolio_values.append(self.portfolio_value)
        
        # Update peak and drawdown
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value
        
        # Track returns for Sharpe calculation
        self.returns_history.append(portfolio_return)
        
        # Calculate concentration (Herfindahl index)
        concentration = float(np.sum(self.current_weights ** 2))
        
        return portfolio_return, turnover, concentration
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment.
        
        Args:
            action: Action from the agent (raw weights before processing).
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        if self._terminated or self._truncated:
            raise RuntimeError(
                "Episode has ended. Call reset() to start a new episode."
            )
        
        # Process action into valid weights
        new_weights = self._process_action(action)
        
        # Update portfolio and get metrics
        portfolio_return, turnover, concentration = self._update_portfolio(new_weights)
        
        # Calculate current drawdown
        drawdown = (self.peak_value - self.portfolio_value) / (self.peak_value + 1e-8)
        
        # Calculate reward
        reward = self._calculate_reward(
            portfolio_return=portfolio_return,
            turnover=turnover,
            concentration=concentration,
            drawdown=drawdown,
        )
        
        # Advance time step
        self.current_step += 1
        
        # Check termination conditions
        self._terminated = False
        self._truncated = False
        
        # Terminate if max drawdown exceeded
        if drawdown >= self.max_drawdown_limit:
            self._terminated = True
            logger.info(f"Episode terminated: Max drawdown ({drawdown:.2%}) exceeded.")
        
        # Truncate if we've reached the end of data
        if self.current_step >= len(self.dates) - 1:
            self._truncated = True
            logger.info("Episode truncated: End of data reached.")
        
        # Get new observation
        observation = self._get_observation()
        
        # Build info dict
        info = {
            "portfolio_value": self.portfolio_value,
            "portfolio_return": portfolio_return,
            "cumulative_return": (self.portfolio_value - self.initial_capital) / self.initial_capital,
            "drawdown": drawdown,
            "turnover": turnover,
            "concentration": concentration,
            "sharpe": reward + self.transaction_cost_rate * turnover + 0.1 * (concentration - 1.0 / self.n_tickers) + 0.5 * drawdown,  # Approximate
            "current_step": self.current_step,
            "total_steps": len(self.dates),
            "weights": self.current_weights.copy(),
            "cash_ratio": self.cash_ratio,
        }
        
        return observation, reward, self._terminated, self._truncated, info
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).
            
        Returns:
            Tuple of (initial observation, info dict).
        """
        super().reset(seed=seed)
        
        # Set seeds for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Load data if not already loaded
        if self.indicators is None:
            self._load_data()
        
        # Reset state
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.peak_value = self.initial_capital
        
        # Initialize with equal weights
        self.current_weights = np.ones(self.n_tickers) / self.n_tickers
        self.cash_ratio = 0.0  # Fully invested
        
        self.portfolio_values = [self.initial_capital]
        self.returns_history = []
        
        self._terminated = False
        self._truncated = False
        
        # Get initial observation
        observation = self._get_observation()
        
        info = {
            "initial_capital": self.initial_capital,
            "tickers": self.tickers,
            "start_date": str(self.dates[0]),
        }
        
        logger.debug(f"Environment reset at step 0, portfolio value: ${self.portfolio_value:,.2f}")
        
        return observation, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment.
        
        Returns:
            RGB array if render_mode is 'rgb_array', None otherwise.
        """
        if self.render_mode == "human":
            print(f"\nStep: {self.current_step}/{len(self.dates)}")
            print(f"Date: {self.dates[self.current_step]}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cumulative Return: {(self.portfolio_value - self.initial_capital) / self.initial_capital:.2%}")
            print(f"Drawdown: {(self.peak_value - self.portfolio_value) / self.peak_value:.2%}")
            print(f"Weights: {dict(zip(self.tickers, np.round(self.current_weights, 4)))}")
        elif self.render_mode == "rgb_array":
            # Return a simple representation (could be extended for visualization)
            return np.zeros((64, 64, 3), dtype=np.uint8)
        
        return None
    
    def close(self) -> None:
        """Clean up resources."""
        logger.debug("Environment closed.")
    
    def get_portfolio_metrics(self) -> Dict[str, float]:
        """Get comprehensive portfolio metrics.
        
        Returns:
            Dictionary of portfolio metrics.
        """
        if not self.portfolio_values:
            return {}
        
        values = np.array(self.portfolio_values)
        returns = np.diff(values) / (values[:-1] + 1e-8)
        
        total_return = (values[-1] - values[0]) / values[0]
        
        if len(returns) > 1:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe = (avg_return * 252 - self.risk_free_rate) / (std_return * np.sqrt(252) + 1e-8)
            max_dd = np.max(np.maximum.accumulate(values) - values) / np.max(values)
        else:
            avg_return = 0.0
            std_return = 0.0
            sharpe = 0.0
            max_dd = 0.0
        
        return {
            "total_return": float(total_return),
            "annualized_return": float((1 + total_return) ** (252 / len(values)) - 1),
            "volatility": float(std_return * np.sqrt(252)),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "final_value": float(values[-1]),
        }


if __name__ == "__main__":
    # Test the environment
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    print("=" * 60)
    print("Testing MycroftFinanceEnv")
    print("=" * 60)
    
    # Create environment
    env = MycroftFinanceEnv(
        tickers=get_default_tickers()[:5],  # Use fewer tickers for faster test
        start_date="2023-01-01",
        end_date="2023-12-31",
        render_mode=None,
    )
    
    # Reset environment
    obs, info = env.reset(seed=42)
    
    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Expected shape: {env.observation_space.shape}")
    assert obs.shape == env.observation_space.shape, "Observation shape mismatch!"
    
    print(f"\nAction space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    print(f"\nInitial info: {info}")
    
    # Run 100 random steps
    print("\n" + "-" * 60)
    print("Running 100 random steps...")
    print("-" * 60)
    
    rewards = []
    portfolio_values = []
    
    for step in range(100):
        # Sample random action
        action = env.action_space.sample()
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        portfolio_values.append(info["portfolio_value"])
        
        if step % 20 == 0:
            print(
                f"Step {step}: reward={reward:.4f}, "
                f"portfolio=${info['portfolio_value']:,.2f}, "
                f"drawdown={info['drawdown']:.2%}"
            )
        
        # Check for episode end
        if terminated or truncated:
            print(f"\nEpisode ended at step {step}: terminated={terminated}, truncated={truncated}")
            obs, info = env.reset(seed=42)
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    print(f"\nObservation shape check: {obs.shape} == {env.observation_space.shape} ✓")
    print(f"Action shape check: {env.action_space.shape[0]} == {env.n_tickers} ✓")
    
    print(f"\nReward statistics:")
    print(f"  Mean: {np.mean(rewards):.4f}")
    print(f"  Std:  {np.std(rewards):.4f}")
    print(f"  Min:  {np.min(rewards):.4f}")
    print(f"  Max:  {np.max(rewards):.4f}")
    
    print(f"\nPortfolio value:")
    print(f"  Initial: ${env.initial_capital:,.2f}")
    print(f"  Final:   ${portfolio_values[-1]:,.2f}")
    print(f"  Change:  {(portfolio_values[-1] - env.initial_capital) / env.initial_capital:.2%}")
    
    metrics = env.get_portfolio_metrics()
    print(f"\nPortfolio metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
    
    print("\n✓ Environment test completed successfully!")
    
    # Close environment
    env.close()
