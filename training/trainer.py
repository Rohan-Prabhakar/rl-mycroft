import argparse
import logging
import sys
import os
from typing import Optional, List

# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.mycroft_finance_env import MycroftFinanceEnv
from agents.sac_agent import SacAgent, PortfolioEvalCallback
from training.config import get_config, TrainingConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train SAC agent for portfolio optimization")
    parser.add_argument("--env-ticker-set", type=str, nargs='+', default=None, help="Custom list of tickers")
    parser.add_argument("--timesteps", type=int, default=None, help="Total training timesteps")
    parser.add_argument("--log-dir", type=str, default=None, help="Directory for TensorBoard logs")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory to save models")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode (5k steps)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Determine timesteps
    timesteps = args.timesteps
    if args.debug:
        timesteps = 5000
        logger.info("Running in DEBUG mode with 5000 timesteps")
    
    # Load configuration
    config = get_config(
        ticker_set=args.env_ticker_set,
        timesteps=timesteps,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        seed=args.seed
    )
    
    logger.info(f"Configuration loaded: Seed={config.seed}, Timesteps={config.total_timesteps}")
    logger.info(f"Log directory: {config.log_dir}")
    logger.info(f"Model directory: {config.model_dir}")
    
    # Initialize Environment
    env = MycroftFinanceEnv(
        start_date=config.start_date,
        end_date=config.end_date,
        initial_portfolio_value=config.initial_portfolio_value,
        transaction_cost=config.transaction_cost,
        max_drawdown_limit=config.max_drawdown_limit
    )
    
    # Reset env to set spaces and seed
    env.reset(seed=config.seed)
    
    # Initialize Agent
    agent = SacAgent(
        env=env,
        learning_rate=config.learning_rate,
        buffer_size=config.buffer_size,
        batch_size=config.batch_size,
        entropy_coef=config.entropy_coef,
        gamma=config.gamma,
        tau=config.tau,
        total_timesteps=config.total_timesteps,
        seed=config.seed,
        device=config.device
    )
    
    # Setup Callback
    callback = PortfolioEvalCallback(
        eval_freq=max(1000, config.total_timesteps // 20),
        log_dir=config.log_dir,
        best_model_save_path=config.model_dir,
        verbose=1
    )
    
    # Train
    try:
        agent.train(
            log_dir=config.log_dir,
            callback=callback,
            tb_log_name="SAC_Mycroft"
        )
        
        # Save final model
        final_model_path = os.path.join(config.model_dir, "final_sac_model")
        agent.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        interrupt_path = os.path.join(config.model_dir, "interrupted_sac_model")
        agent.save(interrupt_path)
        logger.info(f"Checkpoint saved to {interrupt_path}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
