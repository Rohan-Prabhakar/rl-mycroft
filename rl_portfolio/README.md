# RL Portfolio Agents for Mycroft

Reinforcement Learning framework for Mycroft's Portfolio Agents to adapt investment strategies based on market feedback. Implements Soft Actor-Critic (SAC) to optimize portfolio allocation across AI companies.

## Overview

This module provides:
- **SAC Agent**: Soft Actor-Critic implementation using Stable-Baselines3
- **Mycroft Finance Environment**: Gymnasium environment for portfolio optimization
- **Training Pipeline**: Complete training script with evaluation callbacks
- **Frontend Integration**: API routes and UI components for the Next.js frontend

## Directory Structure

```
rl_portfolio/
├── agents/
│   └── sac_agent.py          # SAC agent implementation
├── envs/
│   └── mycroft_finance_env.py  # Gymnasium environment
├── training/
│   ├── config.py             # Training configuration
│   └── train.py              # Training script
├── models/                    # Saved model checkpoints
├── logs/                      # TensorBoard logs
└── data/                      # Market data files
```

## Installation

```bash
pip install stable-baselines3 gymnasium pandas numpy torch tensorboard
```

## Usage

### Training the SAC Agent

```bash
cd rl_portfolio
python training/train.py --debug  # Quick test with 5k steps
python training/train.py --timesteps 100000  # Full training
```

### Command Line Options

- `--env-ticker-set`: Custom list of stock tickers
- `--data-path`: Path to pickle file with historical data
- `--timesteps`: Total training timesteps
- `--log-dir`: Directory for TensorBoard logs
- `--model-dir`: Directory to save models
- `--seed`: Random seed for reproducibility
- `--debug`: Run in debug mode (5k steps)

### Using the Trained Model

```python
from agents.sac_agent import SacAgent
from envs.mycroft_finance_env import MycroftFinanceEnv

# Load environment
env = MycroftFinanceEnv(tickers=['NVDA', 'MSFT', 'GOOGL'])

# Load trained model
agent = SacAgent.load("models/final_sac_model", env)

# Get predictions
observation, _ = env.reset()
action = agent.predict(observation, deterministic=True)
print(f"Recommended allocation: {action}")
```

## Features

### SAC Agent
- Configurable hyperparameters (learning rate, buffer size, batch size, etc.)
- Custom evaluation callback for portfolio metrics
- Automatic best model saving based on Sharpe ratio
- NaN/Inf handling in observations

### Environment
- Supports multiple AI company stocks
- Realistic transaction costs
- Maximum drawdown protection
- Normalized observations for stable training

### Training Metrics
- Cumulative Return
- Sharpe Ratio
- Maximum Drawdown
- Portfolio Value tracking

## Frontend Integration

The RL framework integrates with the Mycroft Next.js frontend through:

1. **API Routes** (`app/api/portfolio/`): Endpoints for model predictions
2. **UI Components**: Portfolio allocation visualizations
3. **Real-time Updates**: WebSocket support for live predictions

See `app/api/portfolio/route.ts` for the API implementation.

## Configuration

Edit `training/config.py` to customize:
- Default ticker set
- Initial capital
- Transaction cost rate
- SAC hyperparameters
- Training settings

## License

MIT License - See LICENSE file for details
