import argparse
import logging
import sys
import os
import multiprocessing
from typing import Optional, List
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

multiprocessing.set_start_method("spawn", force=True)  # Prevents Windows fork hangs

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
        help="Path to pickle file with all 500 tickers"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/sp500_stocks",
        help="Directory containing Kaggle sp500_stocks.csv"
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


def ensure_pickle_exists(data_dir: str, pickle_path: str) -> str:
    """Ensure pickle file exists, create from Kaggle CSV if needed."""

    if os.path.exists(pickle_path):
        logger.info(f"Using existing pickle: {pickle_path}")
        return pickle_path

    logger.info(f"Pickle not found at {pickle_path}")
    logger.info(f"Creating pickle from Kaggle CSV in {data_dir}...")

    # Import from sp500_data_loader (which now has the fixed converter)
    from envs.sp500_data_loader import create_mycroft_pickle
    create_mycroft_pickle(data_dir, pickle_path)

    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Failed to create pickle at {pickle_path}")

    return pickle_path


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

    # Determine pickle path
    pickle_path = args.data_path if args.data_path else "data/sp500_full.pkl"

    # Ensure pickle exists (create from Kaggle CSV if needed)
    pickle_path = ensure_pickle_exists(args.data_dir, pickle_path)

    logger.info("Loading data from pickle file: %s", pickle_path)

    # Initialize Environment with pickle
    env = MycroftFinanceEnv(
        tickers=config.ticker_set,
        pickle_path=pickle_path,
        initial_capital=config.initial_capital,
        transaction_cost_rate=config.transaction_cost_rate,
        max_drawdown_limit=config.max_drawdown_limit
    )

    # Reset env to set spaces and seed
    env.reset(seed=config.seed)
    logger.info("Wrapping environment with VecNormalize for stability...")
    env = DummyVecEnv([lambda: env])
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99,
        epsilon=1e-8
    )
    logger.info("✅ Environment normalized (obs/reward clipped to [-10, 10])")

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
        eval_freq=10000,          # Less frequent evals
        log_dir=config.log_dir,
        best_model_save_path=config.model_dir,
        verbose=0
    )

    # Train
    try:
        agent.train(
            log_dir=config.log_dir,
            callback=callback,
            tb_log_name=None
        )
        # Quick progress tracker (low-overhead)
        import time
        start = time.time()
        for step in range(0, 50000, 5000):
            time.sleep(1)  # Just to yield control
            elapsed = time.time() - start
            print(f"\r[Progress] Step {step}/50000 | ETA: {(50000-step)/max(1, step/elapsed)*elapsed/60:.1f} min", end="", flush=True)

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
