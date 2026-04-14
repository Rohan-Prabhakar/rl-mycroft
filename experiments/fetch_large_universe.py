"""
Fetch large universe of stocks (e.g., S&P 500) and save as pickle.
This script downloads adjusted close prices for all constituents,
cleans the data, and saves it as a DataFrame pickle for training.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import argparse
from datetime import datetime
import os

# S&P 500 Tickers (Static list for reliability, can be scraped dynamically)
SP500_TICKERS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "GOOG", "META", "TSLA", "BRK.B", "UNH",
    "JNJ", "V", "XOM", "JPM", "PG", "MA", "HD", "CVX", "LLY", "MRK",
    "ABBV", "PEP", "KO", "AVGO", "COST", "WMT", "MCD", "TMO", "CSCO", "ACN",
    "BAC", "ABT", "ADBE", "LIN", "TXN", "NEE", "DHR", "VZ", "CRM", "ORCL",
    "WFC", "AMD", "PM", "INTC", "NKE", "DIS", "UPS", "RTX", "QCOM", "HON",
    "LOW", "AMGN", "SPGI", "IBM", "GS", "CAT", "ELV", "INTU", "BLK", "DE",
    "SYK", "GILD", "BKNG", "AXP", "MDT", "ADI", "TJX", "VRTX", "LMT", "ADP",
    "SCHW", "PLD", "MU", "CI", "CB", "TMUS", "ZTS", "MDLZ", "REGN", "PYPL",
    "DUK", "SO", "ISRG", "BDX", "TGT", "CL", "FIS", "ITW", "BSX", "APD",
    "EQIX", "SHW", "EOG", "CME", "ICE", "NSC", "WM", "GD", "FDX", "EMR",
    "PSA", "MMM", "FCX", "USB", "PNC", "AON", "CLX", "HCA", "MCO", "DXCM",
    "KLAC", "SNPS", "CDNS", "ORLY", "ADSK", "NXPI", "ROP", "ROST", "IDXX", "CTAS",
    "FAST", "PCAR", "MCHP", "PAYX", "KMB", "EA", "VRSK", "CTSH", "EXC", "XEL",
    "WELL", "ODFL", "CPRT", "CSGP", "BIIB", "ILMN", "EW", "ALGN", "ENPH", "SEDG",
    "MRNA", "ZM", "DOCU", "PTON", "ROKU", "SQ", "SHOP", "UBER", "LYFT", "SNAP"
]

def fetch_data(tickers, start_date, end_date):
    """Download adjusted close prices for a list of tickers."""
    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")
    
    data = yf.download(tickers, start=start_date, end=end_date, progress=True)['Adj Close']
    
    # Handle multi-level columns if any (sometimes happens with yfinance)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    return data

def clean_data(df, min_days_ratio=0.8):
    """
    Clean the dataset:
    - Remove tickers with too many missing values.
    - Forward fill small gaps.
    - Drop rows with too many missing values.
    """
    print("Cleaning data...")
    initial_shape = df.shape
    
    # Calculate missing ratio per column
    missing_ratio = df.isnull().sum() / len(df)
    valid_tickers = missing_ratio[missing_ratio <= (1 - min_days_ratio)].index.tolist()
    
    print(f"Removed {len(df.columns) - len(valid_tickers)} tickers with excessive missing data.")
    df = df[valid_tickers]
    
    # Forward fill then backward fill for small gaps
    df = df.ffill().bfill()
    
    # Drop rows where > 10% of assets are missing (should be rare after ffill/bfill)
    row_missing_ratio = df.isnull().mean(axis=1)
    df = df[row_missing_ratio < 0.1]
    
    # Drop any remaining rows with NaNs
    df = df.dropna()
    
    print(f"Final dataset shape: {df.shape} (Days x Tickers)")
    return df

def main():
    parser = argparse.ArgumentParser(description="Fetch large stock universe data")
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated list of tickers (default: S&P 500)")
    parser.add_argument("--period", type=str, default="5y", help="Period to fetch (e.g., 2y, 5y, max)")
    parser.add_argument("--output", type=str, default="data/sp500_5y.pkl", help="Output pickle file path")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD), overrides period")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    # Determine tickers
    if args.tickers:
        ticker_list = [t.strip() for t in args.tickers.split(',')]
    else:
        ticker_list = SP500_TICKERS
    
    # Determine dates
    end_date = datetime.now() if not args.end_date else datetime.strptime(args.end_date, "%Y-%m-%d")
    if args.start_date:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    else:
        # Simple period calculation
        if args.period == "2y":
            start_date = datetime(end_date.year - 2, end_date.month, end_date.day)
        elif args.period == "5y":
            start_date = datetime(end_date.year - 5, end_date.month, end_date.day)
        elif args.period == "10y":
            start_date = datetime(end_date.year - 10, end_date.month, end_date.day)
        else:
            start_date = datetime(end_date.year - 5, end_date.month, end_date.day) # Default 5y

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    try:
        # Fetch
        df = fetch_data(ticker_list, start_date, end_date)
        
        # Clean
        df_clean = clean_data(df)
        
        if df_clean.empty:
            raise ValueError("Dataset is empty after cleaning. Try adjusting parameters.")
        
        # Save
        df_clean.to_pickle(args.output)
        print(f"✅ Successfully saved data to {args.output}")
        print(f"   Columns (Tickers): {len(df_clean.columns)}")
        print(f"   Rows (Days): {len(df_clean)}")
        print(f"   Date Range: {df_clean.index.min()} to {df_clean.index.max()}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

if __name__ == "__main__":
    main()
