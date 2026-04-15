"""
Training configuration for SAC portfolio optimization.
"""

import os
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration class for SAC training."""
    
    # Environment settings
    ticker_set: List[str]
    initial_capital: float = 100000.0
    transaction_cost_rate: float = 0.001
    max_drawdown_limit: float = 0.20
    
    # SAC hyperparameters
    learning_rate: float = 1e-4
    buffer_size: int = 50000
    batch_size: int = 128
    entropy_coef: float = 0.2
    gamma: float = 0.99
    tau: float = 0.005
    
    # Training settings
    total_timesteps: int = 100000
    seed: int = 42
    device: str = "auto"
    
    # Paths
    log_dir: str = "./logs"
    model_dir: str = "./models"
    
    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)


def get_config(
    ticker_set: Optional[List[str]] = None,
    timesteps: Optional[int] = None,
    log_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    seed: Optional[int] = None,
) -> TrainingConfig:
    """
    Get training configuration with optional overrides.
    
    Args:
        ticker_set: Custom list of tickers to use.
        timesteps: Override total training timesteps.
        log_dir: Override log directory.
        model_dir: Override model directory.
        seed: Override random seed.
        
    Returns:
        TrainingConfig instance with specified settings.
    """
    # Default AI company tickers
    default_tickers = [
        'NVDA', 'MSFT', 'GOOGL', 'META', 'AMZN',
        'TSLA', 'AMD', 'INTC', 'CRM', 'ORCL',
        'IBM', 'BABA', 'TCEHY', 'SONY', 'SAP',
        'PLTR', 'SNOW', 'PATH', 'AI', 'BBAI'
    ]
    
    config = TrainingConfig(
        ticker_set=ticker_set or default_tickers,
        total_timesteps=timesteps or 100000,
        log_dir=log_dir or "./logs",
        model_dir=model_dir or "./models",
        seed=seed or 42,
    )
    
    return config
