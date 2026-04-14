#!/usr/bin/env python3
"""
Load S&P 500 stock data from Kaggle dataset CSV files.

This module provides functionality to load historical OHLCV data 
from the S&P 500 Stocks Kaggle dataset instead of fetching from Yahoo Finance.

Dataset: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks

Expected files in data directory:
1. sp500_stocks.csv - Main price data (Date, Symbol, Open, High, Low, Close, Adj Close, Volume)
2. sp500_companies.csv - Company metadata (Symbol, Security, GICS Sector, etc.)
3. sp500_index.csv - Index-level data (Date, S&P500)

This script reads directly from these three files - no conversion needed.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_sp500_tickers_from_dataset(data_dir: str) -> List[str]:
    """Extract available ticker symbols from sp500_stocks.csv.
    
    Args:
        data_dir: Path to the directory containing the Kaggle dataset files.
        
    Returns:
        List of ticker symbols found in sp500_stocks.csv.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    stocks_file = data_path / "sp500_stocks.csv"
    if not stocks_file.exists():
        raise FileNotFoundError(f"sp500_stocks.csv not found in {data_dir}")
    
    logger.info("Reading tickers from sp500_stocks.csv")
    try:
        # Read just the Symbol column to get unique tickers
        df = pd.read_csv(stocks_file, usecols=["Symbol"], nrows=50000)
        tickers = df["Symbol"].unique().tolist()
        return sorted([str(t).upper() for t in tickers])
    except Exception as e:
        logger.error(f"Could not read sp500_stocks.csv: {e}")
        raise


class SP500DataLoader:
    """Load and preprocess S&P 500 stock data from Kaggle dataset.
    
    This class loads historical OHLCV data directly from sp500_stocks.csv
    in the S&P 500 Stocks Kaggle dataset format.
    
    Attributes:
        data_dir: Path to directory containing the Kaggle dataset files.
        tickers: List of ticker symbols to load.
        data: Dictionary mapping ticker symbols to DataFrames.
        combined_data: Combined DataFrame with all indicators.
    """
    
    def __init__(
        self,
        data_dir: str,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
        """Initialize the SP500DataLoader.
        
        Args:
            data_dir: Path to directory containing sp500_stocks.csv, sp500_companies.csv, sp500_index.csv.
            tickers: List of ticker symbols to load. If None, loads all available.
            start_date: Start date filter (YYYY-MM-DD). If None, uses all available.
            end_date: End date filter (YYYY-MM-DD). If None, uses all available.
        """
        self.data_dir = Path(data_dir)
        self.start_date = start_date
        self.end_date = end_date
        self.tickers = tickers
        self.data: Dict[str, pd.DataFrame] = {}
        self.combined_data: Optional[pd.DataFrame] = None
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        
        stocks_file = self.data_dir / "sp500_stocks.csv"
        if not stocks_file.exists():
            raise FileNotFoundError(f"sp500_stocks.csv not found in {self.data_dir}")
        
        # If no tickers specified, discover them from the dataset
        if self.tickers is None:
            self.tickers = get_sp500_tickers_from_dataset(str(self.data_dir))
            logger.info(f"Discovered {len(self.tickers)} tickers in dataset")
    
    def fetch_data(self) -> "SP500DataLoader":
        """Load historical OHLCV data from sp500_stocks.csv.
        
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If no data could be loaded for any ticker.
        """
        stocks_file = self.data_dir / "sp500_stocks.csv"
        logger.info(f"Loading data from sp500_stocks.csv")
        return self._load_stocks_data(stocks_file)
    
    def _load_stocks_data(self, csv_path: Path) -> "SP500DataLoader":
        """Load data from sp500_stocks.csv.
        
        Expected columns: Date, Symbol, Open, High, Low, Close, Adj Close, Volume
        
        Args:
            csv_path: Path to sp500_stocks.csv file.
            
        Returns:
            Self for method chaining.
        """
        try:
            # Read the entire file
            df = pd.read_csv(csv_path)
            
            # Standardize column names (lowercase, strip whitespace)
            df.columns = df.columns.str.strip().str.lower()
            
            # Convert date column
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            
            # Apply date filters if specified
            if self.start_date or self.end_date:
                mask = pd.Series(True, index=df.index)
                if self.start_date:
                    mask &= df["date"] >= pd.to_datetime(self.start_date)
                if self.end_date:
                    mask &= df["date"] <= pd.to_datetime(self.end_date)
                df = df[mask]
            
            # Ensure numeric types for price/volume columns
            numeric_cols = ["open", "high", "low", "close", "adj close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Drop rows with NaN in essential columns
            df = df.dropna(subset=["open", "high", "low", "close", "volume"])
            
            # Use 'close' or 'adj close' as the primary close price
            if "adj close" in df.columns:
                # Use adjusted close for more accurate returns
                df["close"] = df["adj close"]
            
            # Split by symbol and store as separate DataFrames
            loaded_count = 0
            target_tickers = set([t.upper() for t in self.tickers]) if self.tickers else None
            
            for symbol in df["symbol"].unique():
                symbol_upper = symbol.upper()
                if target_tickers and symbol_upper not in target_tickers:
                    continue
                    
                symbol_df = df[df["symbol"] == symbol].copy()
                symbol_df = symbol_df.set_index("date")
                symbol_df = symbol_df.sort_index()
                
                # Keep only the columns we need
                keep_cols = ["open", "high", "low", "close", "volume"]
                symbol_df = symbol_df[keep_cols]
                
                self.data[symbol_upper] = symbol_df
                loaded_count += 1
                logger.debug(f"Loaded {len(symbol_df)} rows for {symbol_upper}")
            
            if not self.data:
                raise ValueError("No data could be loaded for any ticker.")
            
            logger.info(f"Successfully loaded data for {loaded_count} tickers.")
            return self
            
        except Exception as e:
            logger.error(f"Error loading data from sp500_stocks.csv: {e}")
            raise
    
    def compute_indicators(self) -> "SP500DataLoader":
        """Compute technical indicators for all tickers.
        
        Computes:
            - Returns: Daily price returns
            - Volatility: Rolling 20-day standard deviation of returns
            - Volume_zscore: Z-score of volume relative to 20-day rolling mean
            - RSI: 14-day Relative Strength Index
            - MACD: Moving Average Convergence Divergence
            
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If data has not been fetched yet.
        """
        if not self.data:
            raise ValueError("Must call fetch_data() before compute_indicators().")
        
        logger.info("Computing technical indicators...")
        
        indicator_dfs = {}
        
        for ticker, df in self.data.items():
            try:
                close = df["close"]
                volume = df["volume"]
                
                # Daily returns
                returns = close.pct_change()
                
                # Rolling volatility (20-day)
                volatility = returns.rolling(window=20).std()
                
                # Volume z-score (20-day rolling)
                volume_mean = volume.rolling(window=20).mean()
                volume_std = volume.rolling(window=20).std()
                volume_zscore = (volume - volume_mean) / (volume_std + 1e-8)
                
                # RSI (14-day)
                delta = close.diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                
                # MACD (12, 26, 9)
                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                macd_hist = macd - signal
                
                # Combine indicators
                indicators = pd.DataFrame({
                    "returns": returns,
                    "volatility": volatility,
                    "volume_zscore": volume_zscore,
                    "rsi": rsi,
                    "macd": macd_hist,
                })
                
                indicator_dfs[ticker] = indicators
                
            except Exception as e:
                logger.error(f"Error computing indicators for {ticker}: {e}")
                continue
        
        if not indicator_dfs:
            raise ValueError("Could not compute indicators for any ticker.")
        
        # Combine all indicators into a single DataFrame with MultiIndex columns
        combined = pd.concat(indicator_dfs, axis=1)
        
        # Drop rows with NaN values (from rolling windows)
        combined = combined.dropna()
        
        self.combined_data = combined
        logger.info(f"Computed indicators. Data shape: {combined.shape}")
        
        return self
    
    def get_aligned_data(self) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Get aligned data ready for environment use.
        
        Returns:
            Tuple of (combined DataFrame with MultiIndex columns, DatetimeIndex).
            
        Raises:
            ValueError: If indicators have not been computed.
        """
        if self.combined_data is None:
            raise ValueError(
                "Must call fetch_data() and compute_indicators() first."
            )
        
        return self.combined_data, self.combined_data.index
    
    def get_prices(self) -> pd.DataFrame:
        """Get aligned closing prices for all tickers.
        
        Returns:
            DataFrame with closing prices, columns are ticker symbols.
            
        Raises:
            ValueError: If data has not been fetched.
        """
        if not self.data:
            raise ValueError("Must call fetch_data() first.")
        
        # Get common index
        common_index = None
        price_dfs = {}
        
        for ticker, df in self.data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
            price_dfs[ticker] = df["close"]
        
        # Align all price series to common index
        aligned = pd.DataFrame({
            ticker: series.reindex(common_index)
            for ticker, series in price_dfs.items()
        })
        
        return aligned.dropna()


def load_sp500_data(
    data_dir: str,
    tickers: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Convenience function to load and prepare S&P 500 data.
    
    Args:
        data_dir: Path to directory containing stock CSV files.
        tickers: List of ticker symbols. If None, loads all available.
        start_date: Start date filter (YYYY-MM-DD).
        end_date: End date filter (YYYY-MM-DD).
        
    Returns:
        Tuple of (indicators DataFrame, prices DataFrame, DatetimeIndex).
    """
    loader = SP500DataLoader(
        data_dir=data_dir,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    loader.fetch_data()
    loader.compute_indicators()
    
    indicators, index = loader.get_aligned_data()
    prices = loader.get_prices()
    
    # Align prices to indicators index
    prices = prices.reindex(index).dropna()
    
    return indicators, prices, index


if __name__ == "__main__":
    # Test the SP500DataLoader
    logging.basicConfig(level=logging.INFO)
    
    print("Testing SP500DataLoader...")
    
    # Default path - adjust as needed
    data_dir = "/workspace/data/sp500_stocks"
    
    if not Path(data_dir).exists():
        print(f"\n⚠ Data directory not found: {data_dir}")
        print("Please download the S&P 500 dataset from:")
        print("https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks")
        print("And extract it to the data directory.")
    else:
        # Load a subset of tickers for testing
        test_tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        
        loader = SP500DataLoader(data_dir=data_dir, tickers=test_tickers)
        loader.fetch_data()
        loader.compute_indicators()
        
        indicators, index = loader.get_aligned_data()
        prices = loader.get_prices()
        
        print(f"\nIndicators shape: {indicators.shape}")
        print(f"Prices shape: {prices.shape}")
        print(f"Date range: {index[0]} to {index[-1]}")
        print(f"Number of tickers loaded: {len(loader.data)}")
        
        print("\nIndicator columns sample:")
        print(indicators.columns[:10].tolist())
        
        print("\nPrice statistics:")
        print(prices.describe())
        
        print("\n✓ SP500DataLoader test completed successfully!")
