"""
Mycroft Finance Environment for Portfolio Optimization

A Gymnasium environment for training RL agents to optimize portfolio allocation
across AI companies based on market data.
"""

import os
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Optional, Tuple, Any
import gymnasium as gym
from gymnasium import spaces
import logging

logger = logging.getLogger(__name__)


class MycroftFinanceEnv(gym.Env):
    """
    Custom Gymnasium environment for portfolio optimization with AI company stocks.
    
    This environment simulates trading in a portfolio of AI-related companies,
    providing observations based on market data and accepting actions that
    represent portfolio allocation weights.
    
    Attributes:
        tickers: List of stock ticker symbols to include in the portfolio.
        pickle_path: Path to pickle file containing historical price data.
        initial_capital: Starting capital for the portfolio.
        transaction_cost_rate: Cost rate for each transaction (e.g., 0.001 for 0.1%).
        max_drawdown_limit: Maximum allowed drawdown before episode termination.
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
    }
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        pickle_path: str = "data/sp500_full.pkl",
        initial_capital: float = 100000.0,
        transaction_cost_rate: float = 0.001,
        max_drawdown_limit: float = 0.20,
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        
        self.tickers = tickers or self._load_default_tickers()
        self.pickle_path = pickle_path
        self.initial_capital = initial_capital
        self.transaction_cost_rate = transaction_cost_rate
        self.max_drawdown_limit = max_drawdown_limit
        self.render_mode = render_mode
        
        # Load price data
        self.price_data = self._load_price_data()
        
        # Filter price data to only include available tickers
        self.available_tickers = [t for t in self.tickers if t in self.price_data.columns]
        if len(self.available_tickers) == 0:
            raise ValueError("No valid tickers found in price data")
        
        self.n_assets = len(self.available_tickers)
        self.n_features = 5  # Open, High, Low, Close, Volume normalized
        
        # Calculate min/max prices for normalization
        self.min_prices = self.price_data[self.available_tickers].min().min()
        self.max_prices = self.price_data[self.available_tickers].max().max()
        
        # Action space: Portfolio weights (continuous, sum to 1)
        # Each action is a weight between 0 and 1 for each asset
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )
        
        # Observation space: Normalized prices + portfolio state
        # Features per asset: [norm_price, norm_volume, price_change_1d, price_change_5d, price_change_10d]
        # Plus: cash_ratio, current_return, day_of_episode
        obs_dim = self.n_assets * self.n_features + 3
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # State variables
        self.current_step = 0
        self.portfolio_value = initial_capital
        self.cash = initial_capital
        self.shares = np.zeros(self.n_assets)
        self.initial_portfolio_value = initial_capital
        self.peak_portfolio_value = initial_capital
        self.current_drawdown = 0.0
        
        logger.info(
            "MycroftFinanceEnv initialized with %d assets: %s",
            self.n_assets,
            self.available_tickers
        )
    
    def _load_default_tickers(self) -> List[str]:
        """Load default list of AI-related company tickers."""
        # Major AI companies
        return [
            'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN',  # Tech giants
            'TSLA', 'AMD', 'INTC', 'CRM', 'ORCL',     # Hardware/Software
            'IBM', 'BABA', 'TCEHY', 'SONY', 'SAP',    # International
            'PLTR', 'SNOW', 'PATH', 'AI', 'BBAI'      # Pure-play AI
        ]
    
    def _load_price_data(self) -> pd.DataFrame:
        """Load price data from pickle file."""
        if not os.path.exists(self.pickle_path):
            logger.warning("Pickle file not found: %s. Creating dummy data.", self.pickle_path)
            return self._create_dummy_data()
        
        try:
            with open(self.pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            if isinstance(data, pd.DataFrame):
                return data
            elif isinstance(data, dict) and 'prices' in data:
                return data['prices']
            else:
                logger.warning("Unexpected data format in pickle. Creating dummy data.")
                return self._create_dummy_data()
        except Exception as e:
            logger.error("Error loading pickle: %s. Creating dummy data.", e)
            return self._create_dummy_data()
    
    def _create_dummy_data(self) -> pd.DataFrame:
        """Create dummy price data for testing."""
        dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='D')
        n_days = len(dates)
        
        data = {}
        for ticker in self.tickers:
            # Generate random walk with drift
            np.random.seed(hash(ticker) % 2**32)
            returns = np.random.normal(0.0005, 0.02, n_days)
            prices = 100 * np.exp(np.cumsum(returns))
            
            # Create OHLCV data
            ohlcv = pd.DataFrame({
                f'{ticker}_Open': prices * (1 + np.random.uniform(-0.01, 0.01, n_days)),
                f'{ticker}_High': prices * (1 + np.random.uniform(0, 0.03, n_days)),
                f'{ticker}_Low': prices * (1 - np.random.uniform(0, 0.03, n_days)),
                f'{ticker}_Close': prices,
                f'{ticker}_Volume': np.random.randint(1000000, 10000000, n_days)
            }, index=dates)
            data[ticker] = ohlcv
        
        # Combine into single DataFrame with Close prices
        close_prices = pd.DataFrame({
            ticker: data[ticker][f'{ticker}_Close'] for ticker in self.tickers
        })
        
        return close_prices
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.portfolio_value = self.initial_capital
        self.cash = self.initial_capital
        self.shares = np.zeros(self.n_assets)
        self.initial_portfolio_value = self.initial_capital
        self.peak_portfolio_value = self.initial_capital
        self.current_drawdown = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        logger.debug("Environment reset. Initial portfolio value: %.2f", self.portfolio_value)
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one time step in the environment.
        
        Args:
            action: Array of portfolio weights (should sum to 1).
            
        Returns:
            observation: New observation after taking the action.
            reward: Reward signal from taking the action.
            terminated: Whether the episode has ended due to terminal condition.
            truncated: Whether the episode has ended due to time limit.
            info: Additional information for debugging.
        """
        # Normalize action to ensure weights sum to 1
        if np.sum(action) > 0:
            action = action / np.sum(action)
        else:
            # Equal weights if all zeros
            action = np.ones(self.n_assets) / self.n_assets
        
        # Execute trades based on new allocation
        self._execute_trades(action)
        
        # Advance time step
        self.current_step += 1
        
        # Update portfolio value based on new prices
        self._update_portfolio_value()
        
        # Calculate reward (portfolio return)
        reward = self._calculate_reward()
        
        # Check termination conditions
        terminated = self._check_termination()
        truncated = self.current_step >= len(self.price_data) - 1
        
        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_trades(self, target_weights: np.ndarray):
        """Execute trades to achieve target portfolio weights."""
        if self.current_step >= len(self.price_data) - 1:
            return
        
        # Get current prices
        current_prices = self._get_current_prices()
        
        # Calculate target values for each asset
        target_values = target_weights * self.portfolio_value
        
        # Calculate current values
        current_values = self.shares * current_prices
        
        # Calculate trades (positive = buy, negative = sell)
        trade_values = target_values - current_values
        
        # Calculate transaction costs
        total_turnover = np.sum(np.abs(trade_values))
        transaction_costs = total_turnover * self.transaction_cost_rate
        
        # Deduct transaction costs from cash
        self.cash -= transaction_costs
        
        # Update shares based on trades
        for i in range(self.n_assets):
            if trade_values[i] > 0:
                # Buy
                shares_to_buy = trade_values[i] / current_prices[i]
                self.shares[i] += shares_to_buy
                self.cash -= trade_values[i]
            elif trade_values[i] < 0:
                # Sell
                shares_to_sell = abs(trade_values[i]) / current_prices[i]
                shares_to_sell = min(shares_to_sell, self.shares[i])
                self.shares[i] -= shares_to_sell
                self.cash += shares_to_sell * current_prices[i]
    
    def _get_current_prices(self) -> np.ndarray:
        """Get current prices for all assets."""
        step = min(self.current_step, len(self.price_data) - 1)
        prices = []
        for ticker in self.available_tickers:
            if ticker in self.price_data.columns:
                price = self.price_data.iloc[step][ticker]
                prices.append(price)
            else:
                prices.append(100.0)  # Default price
        return np.array(prices)
    
    def _update_portfolio_value(self):
        """Update portfolio value based on current prices."""
        current_prices = self._get_current_prices()
        stock_value = np.sum(self.shares * current_prices)
        self.portfolio_value = self.cash + stock_value
        
        # Update peak and drawdown
        if self.portfolio_value > self.peak_portfolio_value:
            self.peak_portfolio_value = self.portfolio_value
        
        if self.peak_portfolio_value > 0:
            self.current_drawdown = (self.peak_portfolio_value - self.portfolio_value) / self.peak_portfolio_value
    
    def _calculate_reward(self) -> float:
        """Calculate reward based on portfolio performance."""
        # Simple reward: portfolio return
        portfolio_return = (self.portfolio_value - self.initial_portfolio_value) / self.initial_portfolio_value
        
        # Penalize large drawdowns
        drawdown_penalty = -abs(self.current_drawdown) * 0.5
        
        # Penalize excessive trading (approximated by transaction costs)
        # This is already captured in the portfolio value
        
        reward = portfolio_return + drawdown_penalty
        
        return float(reward)
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate."""
        # Terminate if drawdown exceeds limit
        if self.current_drawdown > self.max_drawdown_limit:
            logger.debug("Episode terminated: Max drawdown exceeded (%.2f%%)", self.current_drawdown * 100)
            return True
        
        # Terminate if portfolio value goes to zero
        if self.portfolio_value <= 0:
            logger.debug("Episode terminated: Portfolio value depleted")
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        step = min(self.current_step, len(self.price_data) - 1)
        
        # Price-based features for each asset
        price_features = []
        for ticker in self.available_tickers:
            if ticker not in self.price_data.columns:
                continue
            
            # Get price history
            if step >= 10:
                price_history = self.price_data[ticker].iloc[step-10:step+1].values
            else:
                price_history = self.price_data[ticker].iloc[:step+1].values
            
            current_price = price_history[-1]
            
            # Normalized price
            norm_price = (current_price - self.min_prices) / (self.max_prices - self.min_prices + 1e-8)
            
            # Normalized volume (using dummy volume)
            norm_volume = 0.5  # Placeholder
            
            # Price changes
            price_change_1d = (price_history[-1] - price_history[-2]) / (price_history[-2] + 1e-8) if len(price_history) >= 2 else 0.0
            price_change_5d = (price_history[-1] - price_history[-6]) / (price_history[-6] + 1e-8) if len(price_history) >= 6 else 0.0
            price_change_10d = (price_history[-1] - price_history[-11]) / (price_history[-11] + 1e-8) if len(price_history) >= 11 else 0.0
            
            price_features.extend([
                norm_price,
                norm_volume,
                price_change_1d,
                price_change_5d,
                price_change_10d
            ])
        
        # Portfolio state features
        cash_ratio = self.cash / (self.portfolio_value + 1e-8)
        current_return = (self.portfolio_value - self.initial_portfolio_value) / (self.initial_portfolio_value + 1e-8)
        day_normalized = self.current_step / len(self.price_data)
        
        portfolio_features = [cash_ratio, current_return, day_normalized]
        
        observation = np.array(price_features + portfolio_features, dtype=np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Return additional information about the current state."""
        return {
            'portfolio_value': float(self.portfolio_value),
            'cash': float(self.cash),
            'shares': self.shares.tolist(),
            'current_drawdown': float(self.current_drawdown),
            'current_step': int(self.current_step),
            'tickers': self.available_tickers,
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Cash: ${self.cash:,.2f}")
            print(f"Drawdown: {self.current_drawdown*100:.2f}%")
            print("-" * 50)
    
    def close(self):
        """Clean up resources."""
        pass
