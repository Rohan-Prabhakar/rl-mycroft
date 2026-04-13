import os
import numpy as np
import torch as th
import gymnasium as gym
from typing import Any, Dict, List, Optional, Callable
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.logger import TensorBoardOutputFormat
import logging

logger = logging.getLogger(__name__)

class PortfolioEvalCallback(BaseCallback):
    """
    Custom callback to evaluate portfolio performance during training.
    Logs cumulative return, Sharpe ratio, max drawdown, and policy entropy to TensorBoard.
    Saves the best model based on Sharpe ratio.
    """
    
    def __init__(
        self,
        eval_freq: int = 10000,
        log_dir: str = "./logs",
        best_model_save_path: str = "./models/best",
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.log_dir = log_dir
        self.best_model_save_path = best_model_save_path
        self.best_sharpe = -np.inf
        
        # Metrics storage for the current evaluation episode
        self.episode_returns: List[float] = []
        self.episode_rewards: List[float] = []
        self.portfolio_values: List[float] = []
        self.current_portfolio_value = 0.0
        self.last_eval_metrics: Dict[str, float] = {}
        
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_training_start(self) -> None:
        self.episode_returns = []
        self.episode_rewards = []
        self.portfolio_values = []

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        # Accumulate reward
        rewards = self.locals.get('rewards', [])
        if len(rewards) > 0:
            reward = rewards[0]
            if not np.isnan(reward) and not np.isinf(reward):
                self.episode_rewards.append(float(reward))
        
        # Extract info from the environment
        infos = self.locals.get('infos', [])
        dones = self.locals.get('dones', [False])
        
        for i, (info, done) in enumerate(zip(infos, dones)):
            if 'portfolio_value' in info:
                self.current_portfolio_value = float(info['portfolio_value'])
                self.portfolio_values.append(self.current_portfolio_value)
            
            # Check if episode ended (terminated or truncated)
            if done:
                self._evaluate_episode()
                
        if self.n_calls % self.eval_freq == 0:
            self._log_metrics()
            
        return True

    def _evaluate_episode(self):
        """Calculate metrics for the completed episode."""
        if len(self.portfolio_values) < 2:
            self.episode_returns = []
            self.episode_rewards = []
            self.portfolio_values = []
            return

        # Cumulative Return
        start_val = self.portfolio_values[0]
        end_val = self.portfolio_values[-1]
        cum_return = (end_val - start_val) / start_val if start_val > 0 else 0.0
        
        # Max Drawdown
        peak = np.maximum.accumulate(self.portfolio_values)
        drawdown = (peak - self.portfolio_values) / peak
        drawdown = np.where(np.isfinite(drawdown), drawdown, 0.0)
        max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
        
        # Sharpe Ratio (approximate using daily rewards assuming step=day)
        if len(self.episode_rewards) > 1:
            mean_ret = np.mean(self.episode_rewards)
            std_ret = np.std(self.episode_rewards)
            sharpe = (mean_ret / std_ret) * np.sqrt(252) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0
            
        # Save best model
        if sharpe > self.best_sharpe:
            self.best_sharpe = sharpe
            if self.verbose > 0:
                logger.info(f"New best model found! Sharpe: {sharpe:.4f}")
            self.model.save(os.path.join(self.best_model_save_path, "best_sac_model"))
            
        # Store for logging
        self.last_eval_metrics = {
            'eval/cumulative_return': cum_return,
            'eval/sharpe_ratio': sharpe,
            'eval/max_drawdown': max_drawdown,
            'eval/portfolio_value': end_val
        }
        
        # Reset for next episode
        self.episode_returns = []
        self.episode_rewards = []
        self.portfolio_values = []

    def _log_metrics(self):
        """Log metrics to TensorBoard."""
        if self.last_eval_metrics:
            for key, value in self.last_eval_metrics.items():
                self.logger.record(key, value)

class SacAgent:
    """
    Wrapper around Stable-Baselines3 SAC agent with portfolio-specific configurations.
    """
    
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        buffer_size: int = 100000,
        batch_size: int = 256,
        entropy_coef: float = 0.01,
        gamma: float = 0.99,
        tau: float = 0.005,
        total_timesteps: int = 100000,
        seed: Optional[int] = None,
        device: str = "auto"
    ):
        self.seed = seed
        self.total_timesteps = total_timesteps
        
        # Set seeds for reproducibility
        if seed is not None:
            env.action_space.seed(seed)
            env.observation_space.seed(seed)
            np.random.seed(seed)
            th.manual_seed(seed)
        
        self.model = SAC(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            batch_size=batch_size,
            entropy_coef=entropy_coef,
            gamma=gamma,
            tau=tau,
            train_freq=(1, "step"),
            gradient_steps=1,
            action_noise=None,
            target_update_interval=1,
            device=device,
            seed=seed,
            verbose=1
        )
        
        logger.info(f"SAC Agent initialized with seed={seed}, device={device}")

    def train(
        self,
        log_dir: str = "./logs",
        callback: Optional[BaseCallback] = None,
        tb_log_name: str = "SAC_Portfolio"
    ):
        """Run training."""
        os.makedirs(log_dir, exist_ok=True)
        
        if callback is None:
            callback = PortfolioEvalCallback(
                eval_freq=5000,
                log_dir=log_dir,
                best_model_save_path="./models/best",
                verbose=1
            )
        
        logger.info(f"Starting training for {self.total_timesteps} timesteps...")
        
        self.model.learn(
            total_timesteps=self.total_timesteps,
            callback=callback,
            log_dir=log_dir,
            tb_log_name=tb_log_name,
            reset_num_timesteps=True
        )
        
        logger.info("Training completed.")

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: str, env: gym.Env, device: str = "auto"):
        """Load a trained model."""
        model = SAC.load(path, env=env, device=device)
        agent = cls.__new__(cls)
        agent.model = model
        agent.seed = None
        agent.total_timesteps = 0
        return agent

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Predict action with NaN/Inf handling."""
        if np.any(np.isnan(observation)) or np.any(np.isinf(observation)):
            logger.warning("Observation contains NaN/Inf. Replacing with zeros.")
            observation = np.nan_to_num(observation, nan=0.0, posinf=0.0, neginf=0.0)
        
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action
