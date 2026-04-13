from pydantic import BaseSettings, Field
from typing import Optional, List
import os

class TrainingConfig(BaseSettings):
    """Configuration for SAC training hyperparameters and paths."""
    
    # Hyperparameters
    learning_rate: float = Field(default=3e-4, description="Learning rate for SAC")
    buffer_size: int = Field(default=100000, description="Replay buffer size")
    batch_size: int = Field(default=256, description="Batch size for training")
    entropy_coef: float = Field(default=0.01, description="Entropy coefficient")
    gamma: float = Field(default=0.99, description="Discount factor")
    tau: float = Field(default=0.005, description="Target network update rate")
    total_timesteps: int = Field(default=100000, description="Total training steps")
    
    # Environment settings
    start_date: str = "2020-01-01"
    end_date: Optional[str] = None
    initial_portfolio_value: float = 100000.0
    transaction_cost: float = 0.001
    max_drawdown_limit: float = 0.20
    
    # Paths and Logging
    log_dir: str = "./logs"
    model_dir: str = "./models"
    seed: int = 42
    
    # Device
    device: str = "auto"
    
    class Config:
        env_prefix = "MYCROFT_"
        case_sensitive = False

def get_config(
    ticker_set: Optional[List[str]] = None,
    timesteps: Optional[int] = None,
    log_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    seed: Optional[int] = None
) -> TrainingConfig:
    """Factory to create config with CLI overrides."""
    config = TrainingConfig()
    
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
