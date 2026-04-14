#!/usr/bin/env python3
"""
Fetch real market data from Yahoo Finance using yfinance.

This script downloads historical price data for specified tickers
and saves it in a format compatible with the training pipeline.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_market_data(
    tickers: list[str],
    start_date: str | None = None,
    end_date: str | None = None,
    period: str = "2y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical market data for multiple tickers from Yahoo Finance.

    Args:
        tickers: List of stock ticker symbols (e.g., ["AAPL", "GOOGL", "MSFT"])
        start_date: Start date in YYYY-MM-DD format (optional if period is specified)
        end_date: End date in YYYY-MM-DD format (optional, defaults to today)
        period: Time period to fetch (e.g., "1mo", "3mo", "1y", "2y", "5y", "max")
        interval: Data interval (e.g., "1m", "5m", "1h", "1d", "1wk", "1mo")

    Returns:
        DataFrame with columns: Date, Open, High, Low, Close, Volume for each ticker
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")
    
    logger.info(
        "Fetching data for tickers: %s from %s to %s (period: %s)",
        tickers,
        start_date or period,
        end_date,
        interval,
    )

    # Download data for all tickers
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        period=period if start_date is None else None,
        interval=interval,
        group_by="ticker",
        progress=False,
        threads=True,
    )

    if data.empty:
        raise ValueError("No data retrieved. Check ticker symbols or date range.")

    # Handle multi-level columns if multiple tickers
    if len(tickers) > 1:
        # Flatten column names: (Ticker, PriceType) -> Ticker_PriceType
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ["_".join(col).strip() for col in data.columns.values]
    else:
        # Single ticker: add ticker prefix
        ticker = tickers[0]
        if not any(col.startswith(ticker) for col in data.columns):
            data.columns = [f"{ticker}_{col}" for col in data.columns]

    # Reset index to make Date a column
    data = data.reset_index()

    # Rename Date column if needed
    if "Date" not in data.columns and "date" in [c.lower() for c in data.columns]:
        data.rename(
            columns={c: "Date" for c in data.columns if c.lower() == "date"},
            inplace=True,
        )

    # Drop rows with missing data
    initial_rows = len(data)
    data = data.dropna()
    dropped_rows = initial_rows - len(data)
    if dropped_rows > 0:
        logger.warning("Dropped %d rows with missing data", dropped_rows)

    logger.info("Retrieved %d rows of data for %d tickers", len(data), len(tickers))

    return data


def prepare_training_data(
    raw_data: pd.DataFrame,
    tickers: list[str],
    output_path: Path,
) -> pd.DataFrame:
    """
    Prepare raw market data for training by selecting close prices.

    Args:
        raw_data: Raw DataFrame from yfinance
        tickers: List of ticker symbols
        output_path: Path to save the prepared data

    Returns:
        DataFrame with Date and Close prices for each ticker
    """
    # Select only close prices
    close_columns = [f"{ticker}_Close" for ticker in tickers]
    
    # Check if columns exist
    available_columns = [col for col in close_columns if col in raw_data.columns]
    
    if not available_columns:
        raise ValueError(
            f"No close price columns found. Available: {list(raw_data.columns)}"
        )

    # Create clean dataset
    training_data = raw_data[["Date"] + available_columns].copy()
    
    # Rename columns to ticker names only
    rename_map = {col: col.replace("_Close", "") for col in available_columns}
    training_data.rename(columns=rename_map, inplace=True)

    # Ensure Date is first column
    if "Date" not in training_data.columns:
        # Try to find date column
        date_cols = [c for c in training_data.columns if "date" in c.lower()]
        if date_cols:
            training_data.rename(columns={date_cols[0]: "Date"}, inplace=True)

    # Sort by date
    training_data = training_data.sort_values("Date").reset_index(drop=True)

    # Save to CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    training_data.to_csv(output_path, index=False)
    
    logger.info("Saved prepared data to %s (%d rows, %d columns)", 
                output_path, len(training_data), len(training_data.columns))

    return training_data


def main():
    """Main entry point for fetching market data."""
    parser = argparse.ArgumentParser(
        description="Fetch real market data from Yahoo Finance"
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default="AAPL,GOOGL,MSFT,AMZN,TSLA",
        help="Comma-separated list of ticker symbols (default: AAPL,GOOGL,MSFT,AMZN,TSLA)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/market_data.csv",
        help="Output CSV file path (default: data/market_data.csv)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date in YYYY-MM-DD format (optional)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date in YYYY-MM-DD format (default: today)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="2y",
        choices=["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"],
        help="Time period to fetch (default: 2y)",
    )
    parser.add_argument(
        "--interval",
        type=str,
        default="1d",
        choices=["1m", "5m", "15m", "30m", "1h", "1d", "5d", "1wk", "1mo", "3mo"],
        help="Data interval (default: 1d)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Parse tickers
    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    
    if not tickers:
        raise ValueError("At least one ticker symbol is required")

    output_path = Path(args.output)

    try:
        # Fetch raw data
        raw_data = fetch_market_data(
            tickers=tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            period=args.period,
            interval=args.interval,
        )

        # Prepare for training
        training_data = prepare_training_data(raw_data, tickers, output_path)

        # Print summary
        print("\n" + "=" * 60)
        print("DATA FETCH SUMMARY")
        print("=" * 60)
        print(f"Tickers: {', '.join(tickers)}")
        print(f"Period: {args.period}")
        print(f"Interval: {args.interval}")
        print(f"Date range: {training_data['Date'].min()} to {training_data['Date'].max()}")
        print(f"Total rows: {len(training_data)}")
        print(f"Output file: {output_path.absolute()}")
        print("=" * 60)
        print("\nSample data:")
        print(training_data.head())
        print("\nData statistics:")
        print(training_data.describe())

    except Exception as e:
        logger.error("Failed to fetch data: %s", str(e))
        raise


if __name__ == "__main__":
    main()
