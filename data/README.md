# S&P 500 Dataset Setup

## Download the Dataset

1. Go to https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks
2. Download the dataset (you need a Kaggle account)
3. Extract the files to `/workspace/data/sp500_stocks/`

## Expected Files

After extraction, your directory should contain:
```
/workspace/data/sp500_stocks/
├── sp500_stocks.csv      # Main price data (Date, Symbol, Open, High, Low, Close, Adj Close, Volume)
├── sp500_companies.csv   # Company metadata (Symbol, Security, GICS Sector, etc.)
└── sp500_index.csv       # Index-level data (Date, S&P500)
```

## Usage

### Basic Usage with Environment

```python
from envs import MycroftFinanceEnv

# Initialize environment with S&P 500 dataset
env = MycroftFinanceEnv(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
    sp500_data_dir="/workspace/data/sp500_stocks",
    start_date="2020-01-01",
    end_date="2023-12-31",
)

observation, info = env.reset()
```

### Direct Data Loading

```python
from envs import SP500DataLoader, load_sp500_data

# Option 1: Using the loader class
loader = SP500DataLoader(
    data_dir="/workspace/data/sp500_stocks",
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2023-12-31",
)
loader.fetch_data()
loader.compute_indicators()
indicators, dates = loader.get_aligned_data()

# Option 2: Using convenience function
indicators, prices, dates = load_sp500_data(
    data_dir="/workspace/data/sp500_stocks",
    tickers=["AAPL", "MSFT", "GOOGL"],
    start_date="2020-01-01",
    end_date="2023-12-31",
)
```

### Get Available Tickers

```python
from envs import get_sp500_tickers_from_dataset

tickers = get_sp500_tickers_from_dataset("/workspace/data/sp500_stocks")
print(f"Available tickers: {tickers[:10]}...")  # Show first 10
```

## Technical Indicators Computed

The data loader automatically computes these indicators:
- **Returns**: Daily price returns
- **Volatility**: Rolling 20-day standard deviation of returns
- **Volume Z-score**: Z-score of volume relative to 20-day rolling mean
- **RSI**: 14-day Relative Strength Index
- **MACD**: Moving Average Convergence Divergence (histogram)

## Notes

- The loader uses **Adjusted Close** prices for more accurate return calculations
- Date filtering is applied before computing indicators
- All indicators are aligned to common dates across all tickers
- Rows with NaN values (from rolling windows) are automatically dropped
