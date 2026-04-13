# Phase 1: Custom Gym Environment + Data Pipeline

## Overview

This phase implements the core reinforcement learning environment for portfolio optimization using Gymnasium v0.29+ API. The environment simulates trading across AI-focused public companies with proper state representation, action processing, and reward calculation.

## Files Created

### `envs/__init__.py`
Package initialization that exports the main classes.

### `envs/data_loader.py`
Data loading and preprocessing utilities:
- Fetches historical OHLCV data from Yahoo Finance (yfinance)
- Computes technical indicators: Returns, Volatility, Volume Z-Score, RSI, MACD
- Aligns data across multiple tickers
- Handles missing data gracefully

### `envs/mycroft_finance_env.py`
Custom Gymnasium environment implementing:
- **State Space**: `[price returns, volatility, volume_zscore, RSI, MACD, current_weights, cash_ratio]`
- **Action Space**: Continuous allocation weights (transformed to valid portfolio weights)
- **Reward Function**: Risk-adjusted return - transaction costs - concentration penalty - drawdown penalty
- **Termination**: Max drawdown exceeded or end of data reached
- **Reproducibility**: Deterministic behavior with seed control

### `tests/test_phase1.py`
Comprehensive unit and integration tests covering:
- DataLoader functionality
- Environment initialization
- Action/observation spaces
- Step/reset behavior
- Reward calculation
- Portfolio metrics
- Deterministic behavior
- Reproducibility across runs

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install gymnasium stable-baselines3 pandas numpy yfinance matplotlib seaborn pytest
```

## Usage

### Test the DataLoader

```bash
python -m envs.data_loader
```

Expected output:
- Fetched data for 16 tickers
- Computed indicators with shape (N, 80) where N is number of trading days
- Price statistics summary

### Test the Environment

```bash
python -m envs.mycroft_finance_env
```

Expected output:
- Initial observation shape: (31,) for 5 tickers or (96,) for 15 tickers
- Action space: Box(-1.0, 1.0, (n_tickers,), float32)
- 100 random steps executed
- Reward statistics and portfolio metrics

### Run Tests

```bash
# Run all Phase 1 tests
pytest tests/test_phase1.py -v

# Run specific test class
pytest tests/test_phase1.py::TestMycroftFinanceEnv -v

# Run with coverage (if pytest-cov installed)
pytest tests/test_phase1.py --cov=envs
```

## Environment Configuration

The environment can be customized with these parameters:

```python
from envs import MycroftFinanceEnv, get_default_tickers

env = MycroftFinanceEnv(
    tickers=get_default_tickers(),      # List of ticker symbols
    start_date="2020-01-01",            # Start date for data
    end_date=None,                       # End date (default: today)
    initial_capital=1_000_000.0,        # Starting portfolio value
    transaction_cost_rate=0.001,        # 0.1% transaction cost
    max_drawdown_limit=0.20,            # 20% max drawdown
    risk_free_rate=0.02,                # 2% annual risk-free rate
    window_size=20,                     # Rolling window for stats
    render_mode=None,                   # 'human', 'rgb_array', or None
)
```

## State Space Details

For N tickers, the observation vector has size `5*N + N + 1`:
- Indices `0:N`: Price returns
- Indices `N:2N`: Volatility (20-day rolling std)
- Indices `2N:3N`: Volume z-score
- Indices `3N:4N`: RSI (14-day)
- Indices `4N:5N`: MACD histogram
- Indices `5N:6N`: Current portfolio weights
- Index `6N`: Cash ratio

## Action Processing

Raw actions from the agent are transformed as follows:
1. Shift to positive range: `shifted = action - min(action) + epsilon`
2. Normalize to sum to 1: `weights = shifted / sum(shifted)`
3. Clip to max weight (40%): `weights = clip(weights, 0, 0.4)`
4. Re-normalize: `weights = weights / sum(weights)`

## Reward Function

```
reward = sharpe_ratio 
         - transaction_cost_rate * turnover
         - 0.1 * (concentration - 1/N)
         - 0.5 * drawdown
```

Where:
- `sharpe_ratio`: Daily excess return / recent volatility
- `turnover`: Sum of absolute weight changes
- `concentration`: Herfindahl index of portfolio weights
- `drawdown`: Current drawdown from peak value

## Default Tickers (16 AI-focused companies)

1. NVDA - NVIDIA
2. MSFT - Microsoft
3. GOOGL - Alphabet
4. META - Meta Platforms
5. AMZN - Amazon
6. TSLA - Tesla
7. AMD - Advanced Micro Devices
8. INTC - Intel
9. CRM - Salesforce
10. ORCL - Oracle
11. IBM - IBM
12. ADBE - Adobe
13. NOW - ServiceNow
14. PLTR - Palantir
15. SNOW - Snowflake
16. AI - C3.ai

## Next Steps

After confirming Phase 1 works correctly, proceed to Phase 2 which will implement:
- SAC agent wrapper using Stable-Baselines3
- Training loop with logging
- Configuration management via Pydantic
- Evaluation callbacks
