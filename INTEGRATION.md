# Mycroft RL Portfolio Integration

## Overview

This directory contains the Reinforcement Learning components for Mycroft's Portfolio Agents, featuring a Soft Actor-Critic (SAC) agent that optimizes portfolio allocation across AI companies.

## Architecture

```
rl_portfolio/
├── agents/
│   └── sac_agent.py          # SAC agent with portfolio-specific configurations
├── envs/
│   ├── mycroft_finance_env.py  # Gymnasium environment for portfolio optimization
│   └── sp500_data_loader.py    # Data loading utilities
├── training/
│   ├── config.py              # Training configuration
│   └── train.py               # Training script
├── serve_model.py             # FastAPI backend for model serving
├── data/                      # Pickle data files
├── models/                    # Trained model checkpoints
└── logs/                      # TensorBoard logs
```

## Quick Start

### 1. Install Dependencies

```bash
pip install stable-baselines3 gymnasium numpy pandas torch fastapi uvicorn pydantic
```

### 2. Prepare Data

Download the S&P 500 dataset from Kaggle:
https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks

Place it in `data/sp500_stocks/` and create the pickle file:

```bash
python rl_portfolio/envs/sp500_data_loader.py --mode convert --data_dir data/sp500_stocks --output data/sp500_full.pkl
```

### 3. Train the SAC Agent

```bash
# Debug mode (5k steps)
python rl_portfolio/training/train.py --debug

# Full training (100k steps)
python rl_portfolio/training/train.py --timesteps 100000

# Custom tickers
python rl_portfolio/training/train.py --env-ticker-set NVDA MSFT GOOGL META AMZN
```

### 4. Serve the Model

```bash
# Start the FastAPI server
python rl_portfolio/serve_model.py --port 8000
```

The API will be available at `http://localhost:8000/api/portfolio`

### 5. View the Frontend

Navigate to `/portfolio` in your Next.js application to see the interactive dashboard.

## API Endpoints

### GET /api/portfolio?action=predict

Returns portfolio allocation predictions from the SAC agent.

**Response:**
```json
{
  "success": true,
  "allocations": [
    {"ticker": "NVDA", "weight": 0.18, "companyName": "NVIDIA Corporation"},
    {"ticker": "MSFT", "weight": 0.15, "companyName": "Microsoft Corporation"}
  ],
  "metrics": {
    "sharpeRatio": 1.85,
    "cumulativeReturn": 0.234,
    "maxDrawdown": 0.087,
    "portfolioValue": 123400.00
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### GET /api/portfolio?action=metrics

Returns only portfolio metrics.

### GET /api/portfolio?action=tickers

Returns list of available tickers.

## Configuration

Edit `training/config.py` to customize:

- **Learning rate**: Default 1e-4
- **Buffer size**: Default 50,000
- **Batch size**: Default 128
- **Entropy coefficient**: Default 0.2
- **Gamma (discount factor)**: Default 0.99
- **Tau (target update)**: Default 0.005

## Environment Features

The `MycroftFinanceEnv` provides:

- **Observation Space**: Technical indicators (returns, volatility, RSI, MACD, volume z-score) + current weights + cash ratio
- **Action Space**: Continuous portfolio weights for each ticker
- **Reward Function**: Sharpe ratio - transaction costs - concentration penalty - drawdown penalty
- **Termination**: Max drawdown limit or end of data

## Model Checkpoints

Models are saved in `rl_portfolio/models/`:

- `best_sac_model.zip` - Best model by Sharpe ratio (auto-saved during training)
- `final_sac_model.zip` - Final model after training completes
- `interrupted_sac_model.zip` - Checkpoint if training is interrupted

## Monitoring

View training progress with TensorBoard:

```bash
tensorboard --logdir rl_portfolio/logs
```

Then open http://localhost:6006

## Production Deployment

For production:

1. Set specific CORS origins in `serve_model.py`
2. Use a model registry for version control
3. Add authentication to API endpoints
4. Implement caching for predictions
5. Set up monitoring and alerting
6. Use GPU acceleration if available

## Troubleshooting

### Model not loading

Ensure both the model file and pickle data exist:
```bash
ls rl_portfolio/models/final_sac_model.zip
ls rl_portfolio/data/sp500_full.pkl
```

### Slow training on Windows

The code includes optimizations for Windows:
- Disabled tqdm progress bar
- Set thread limits for NumPy/PyTorch
- Using spawn multiprocessing method

### NaN/Inf in observations

The environment automatically cleans NaN/Inf values. Check logs for warnings.

## License

MIT License - See main repository LICENSE file.
