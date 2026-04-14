import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
import argparse
import os
from datetime import datetime, timedelta
from typing import List, Optional

def get_alpaca_api() -> tradeapi.REST:
    """Initialize Alpaca API client."""
    api_key = os.getenv('ALPACA_API_KEY')
    secret_key = os.getenv('ALPACA_SECRET_KEY')
    
    if not api_key or not secret_key:
        raise ValueError(
            "Alpaca API credentials not found. Please set:\n"
            "  export ALPACA_API_KEY='your_api_key'\n"
            "  export ALPACA_SECRET_KEY='your_secret_key'\n"
            "\nGet free keys at: https://app.alpaca.markets/paper/dashboard/overview"
        )
    
    return tradeapi.REST(api_key, secret_key, base_url='https://data.alpaca.markets')

def download_data(tickers: List[str], start_date: str, end_date: str, timeframe: str = 'Day') -> pd.DataFrame:
    """
    Download adjusted close prices from Alpaca Markets.
    
    Args:
        tickers: List of stock symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        timeframe: 'Day', 'Hour', 'Minute'
    
    Returns:
        DataFrame with dates as index and tickers as columns
    """
    print(f"📥 Fetching data for {len(tickers)} tickers from Alpaca...")
    
    api = get_alpaca_api()
    data_dict = {}
    failed = []
    
    # Convert dates to ISO format for Alpaca
    start_dt = pd.Timestamp(start_date).isoformat()
    end_dt = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).isoformat()  # Include end date
    
    for i, ticker in enumerate(tickers):
        try:
            print(f"   [{i+1}/{len(tickers)}] Downloading {ticker}...", end=" ")
            
            # Fetch bars from Alpaca
            bars = api.get_bars(
                symbol=ticker,
                timeframe=tradeapi.TimeFrame.Day,
                start=start_dt,
                end=end_dt,
                adjustment='all'  # Get adjusted prices
            ).df
            
            if bars.empty:
                print("❌ No data")
                failed.append(ticker)
                continue
            
            # Extract adjusted close prices
            # Alpaca returns multi-index DataFrame: [symbol, timestamp]
            if isinstance(bars.index, pd.MultiIndex):
                prices = bars.xs(ticker, level=0)['close']
            else:
                prices = bars['close']
            
            data_dict[ticker] = prices
            print(f"✅ {len(prices)} days")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:50]}")
            failed.append(ticker)
    
    if failed:
        print(f"\n⚠️ {len(failed)} tickers failed: {failed[:10]}{'...' if len(failed) > 10 else ''}")
    
    if not data_dict:
        return pd.DataFrame()
    
    # Combine into single DataFrame
    df_combined = pd.DataFrame(data_dict)
    
    # Forward fill then backward fill to handle slight date misalignments
    df_combined = df_combined.ffill().bfill()
    
    return df_combined

def clean_data(df: pd.DataFrame, min_days_ratio: float = 0.8) -> pd.DataFrame:
    """Remove tickers with too much missing data."""
    initial_shape = df.shape
    total_days = len(df)
    min_days = int(total_days * min_days_ratio)
    
    # Count non-NaN values per column
    valid_counts = df.count()
    keep_cols = valid_counts[valid_counts >= min_days].index.tolist()
    
    df_clean = df[keep_cols].copy()
    
    removed = initial_shape[1] - df_clean.shape[1]
    print(f"🧹 Cleaning: Removed {removed} tickers with excessive missing data.")
    print(f"   Final dataset shape: {df_clean.shape} (Days x Tickers)")
    
    return df_clean

def save_to_pickle(df: pd.DataFrame, output_path: str):
    """Save the final DataFrame to pickle."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_pickle(output_path)
    print(f"💾 Data saved to {output_path}")
    print(f"   Size: {df.shape[0]} days, {df.shape[1]} tickers")
    print(f"   Date Range: {df.index.min()} to {df.index.max()}")

def main():
    parser = argparse.ArgumentParser(description="Fetch large universe stock data from Alpaca")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated list of tickers")
    parser.add_argument("--period", type=str, default="5y", help="Period (e.g., 2y, 5y, max)")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output", type=str, default="data/alpaca_5y.pkl", help="Output pickle path")
    parser.add_argument("--min-data-ratio", type=float, default=0.7, help="Min ratio of non-null data to keep ticker")
    
    args = parser.parse_args()

    # Determine Date Range
    end_date = args.end_date
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    start_date = args.start_date
    if not start_date:
        # Calculate start based on period if not provided
        years = int(args.period.replace('y', '')) if 'y' in args.period else 5
        start_dt = datetime.now() - timedelta(days=years*365)
        start_date = start_dt.strftime("%Y-%m-%d")

    print(f"📅 Fetching data from {start_date} to {end_date}...")

    # Get Ticker List
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(',')]
    else:
        # Default to top 30 tech/finance stocks
        tickers = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B",
            "JPM", "V", "JNJ", "WMT", "PG", "MA", "UNH", "HD", "DIS", "PYPL",
            "BAC", "ADBE", "CRM", "NFLX", "CMCSA", "XOM", "VZ", "KO", "PFE",
            "INTC", "CSCO", "ABT"
        ]
        print(f"🎯 Using default list of {len(tickers)} major tickers...")
    
    print(f"🎯 Targeting {len(tickers)} tickers...")

    # Download
    df = download_data(tickers, start_date, end_date)

    if df.empty:
        print("❌ Critical Error: No data downloaded at all.")
        print("💡 Check your Alpaca API credentials and internet connection.")
        raise ValueError("Failed to download data. Please verify API keys.")

    # Clean
    df_clean = clean_data(df, args.min_data_ratio)

    if df_clean.empty:
        raise ValueError("Dataset is empty after cleaning. Try reducing --min-data-ratio or checking date range.")

    # Save
    save_to_pickle(df_clean, args.output)
    print("\n✅ Success! Ready for training.")

if __name__ == "__main__":
    main()
