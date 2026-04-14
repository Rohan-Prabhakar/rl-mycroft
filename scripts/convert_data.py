#!/usr/bin/env python3
"""Convert S&P 500 CSV data to pickle format for training."""

import pandas as pd
import os
import argparse

def convert_to_pickle(data_dir: str, output_path: str):
    """Load all stock CSVs and save as a single pickle file."""
    
    stocks_file = os.path.join(data_dir, "sp500_stocks.csv")
    companies_file = os.path.join(data_dir, "sp500_companies.csv")
    
    if not os.path.exists(stocks_file):
        raise FileNotFoundError(f"Could not find {stocks_file}")
    
    # Load the main stocks data
    print(f"Loading stock data from {stocks_file}...")
    df = pd.read_csv(stocks_file)
    
    # Ensure Date column is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Sort by date and symbol
    df = df.sort_values(['Symbol', 'Date'])
    
    print(f"Loaded {len(df)} rows for {df['Symbol'].nunique()} symbols")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Save to pickle
    print(f"Saving to {output_path}...")
    df.to_pickle(output_path)
    print("Done!")
    
    return output_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert S&P 500 CSV to pickle")
    parser.add_argument("--data-dir", type=str, default="/workspace/data/sp500_stocks",
                        help="Directory containing CSV files")
    parser.add_argument("--output", type=str, default="/workspace/data/sp500_data.pkl",
                        help="Output pickle file path")
    
    args = parser.parse_args()
    convert_to_pickle(args.data_dir, args.output)
