from dataclasses import dataclass, field
from typing import Optional, List
import os


@dataclass
class TrainingConfig:
    """Configuration for SAC training hyperparameters and paths."""
    
    # Hyperparameters
    learning_rate: float = field(default=3e-4, metadata={"description": "Learning rate for SAC"})
    buffer_size: int = field(default=100000, metadata={"description": "Replay buffer size"})
    batch_size: int = field(default=256, metadata={"description": "Batch size for training"})
    entropy_coef: float = field(default=0.01, metadata={"description": "Entropy coefficient"})
    gamma: float = field(default=0.99, metadata={"description": "Discount factor"})
    tau: float = field(default=0.005, metadata={"description": "Target network update rate"})
    total_timesteps: int = field(default=100000, metadata={"description": "Total training steps"})
    
    # Environment settings
    ticker_set: Optional[List[str]] = None  # Will use default if None
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    initial_capital: float = 100000.0
    transaction_cost_rate: float = 0.001
    max_drawdown_limit: float = 0.20
    
    # Paths and Logging
    log_dir: str = "./logs"
    model_dir: str = "./models"
    seed: int = 42
    
    # Device
    device: str = "auto"


def get_config(
    ticker_set: Optional[List[str]] = None,
    timesteps: Optional[int] = None,
    log_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> TrainingConfig:
    """Factory to create config with CLI overrides."""
    config = TrainingConfig()
    
    if ticker_set is not None:
        config.ticker_set = ticker_set
    if timesteps is not None:
        config.total_timesteps = timesteps
    if log_dir is not None:
        config.log_dir = log_dir
    if model_dir is not None:
        config.model_dir = model_dir
    if seed is not None:
        config.seed = seed
        
    # Ensure directories exist
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)
    
    return config
