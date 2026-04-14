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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SAC agent for portfolio optimization"
    )
    parser.add_argument(
        "--env-ticker-set",
        type=str,
        nargs="+",
        default=None,
        help="Custom list of tickers"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to data file (CSV or pickle). If provided, overrides start/end dates."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to save models"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode (5k steps)"
    )
    return parser.parse_args()


def main():
    """Main training entry point."""
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
    
    logger.info(
        "Configuration loaded: Seed=%d, Timesteps=%d",
        config.seed,
        config.total_timesteps
    )
    logger.info("Log directory: %s", config.log_dir)
    logger.info("Model directory: %s", config.model_dir)
    logger.info("Data path: %s", args.data_path if args.data_path else "Using default date range")
    
    # Determine pickle_path if data file is provided
    pickle_path = None
    start_date = config.start_date
    end_date = config.end_date
    
    if args.data_path:
        if not os.path.exists(args.data_path):
            raise FileNotFoundError(f"Data file not found: {args.data_path}")
        
        if args.data_path.endswith('.pkl') or args.data_path.endswith('.pickle'):
            pickle_path = args.data_path
            logger.info("Loading data from pickle file: %s", pickle_path)
            # When using pickle, dates are embedded in the file
            start_date = None
            end_date = None
        elif args.data_path.endswith('.csv'):
            logger.info("CSV data path provided but not yet supported for direct loading. Use pickle format.")
            raise ValueError("CSV direct loading not yet implemented. Please convert to pickle first.")
        else:
            raise ValueError("Unsupported file format. Use .pkl or .pickle files.")
    
    # Initialize Environment
    env = MycroftFinanceEnv(
        tickers=config.ticker_set,
        start_date=start_date or "2020-01-01",
        end_date=end_date,
        pickle_path=pickle_path,
        initial_capital=config.initial_capital,
        transaction_cost_rate=config.transaction_cost_rate,
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
        logger.info("Final model saved to %s", final_model_path)
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        interrupt_path = os.path.join(config.model_dir, "interrupted_sac_model")
        agent.save(interrupt_path)
        logger.info("Checkpoint saved to %s", interrupt_path)
    except Exception as e:
        logger.error("Training failed with error: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    main()
