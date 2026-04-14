"""Data loading and preprocessing utilities for Mycroft-RL.

This module handles fetching historical OHLCV data from Yahoo Finance,
computing technical indicators, and preparing data for the environment.
It also supports loading pre-computed data from pickle files.
"""

import logging
from datetime import datetime
from typing import List, Optional, Tuple
import os

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def get_default_tickers() -> List[str]:
    """Return the default list of 15 AI-focused public company tickers.
    
    Returns:
        List of ticker symbols as strings.
    """
    return [
        "NVDA",  # NVIDIA
        "MSFT",  # Microsoft
        "GOOGL",  # Alphabet
        "META",  # Meta Platforms
        "AMZN",  # Amazon
        "TSLA",  # Tesla
        "AMD",  # Advanced Micro Devices
        "INTC",  # Intel
        "CRM",  # Salesforce
        "ORCL",  # Oracle
        "IBM",  # IBM
        "ADBE",  # Adobe
        "NOW",  # ServiceNow
        "PLTR",  # Palantir
        "SNOW",  # Snowflake
        "AI",   # C3.ai
    ]


class DataLoader:
    """Load and preprocess financial data for portfolio optimization.
    
    This class fetches historical OHLCV data from Yahoo Finance and computes
    technical indicators required by the environment state space.
    It can also load pre-computed data from pickle files.
    
    Attributes:
        tickers: List of ticker symbols to load.
        start_date: Start date for historical data (YYYY-MM-DD).
        end_date: End date for historical data (YYYY-MM-DD).
        data: Dictionary mapping ticker symbols to DataFrames.
        pickle_path: Path to pickle file if loading from disk.
    """
    
    def __init__(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = "2020-01-01",
        end_date: Optional[str] = None,
        pickle_path: Optional[str] = None,
    ) -> None:
        """Initialize the DataLoader.
        
        Args:
            tickers: List of ticker symbols. Defaults to AI-focused companies.
            start_date: Start date for historical data.
            end_date: End date for historical data. Defaults to today.
            pickle_path: Path to pickle file with pre-computed prices. If provided,
                data will be loaded from this file instead of fetching from Yahoo.
        """
        self.tickers = tickers if tickers is not None else get_default_tickers()
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime("%Y-%m-%d")
        self.pickle_path = pickle_path
        self.data: dict[str, pd.DataFrame] = {}
        self.combined_data: Optional[pd.DataFrame] = None
        
    def fetch_data(self) -> "DataLoader":
        """Fetch historical OHLCV data from Yahoo Finance or load from pickle.
        
        Returns:
            Self for method chaining.
            
        Raises:
            ValueError: If no data could be fetched for any ticker.
            FileNotFoundError: If pickle_path is provided but file doesn't exist.
        """
        if self.pickle_path and os.path.exists(self.pickle_path):
            return self._load_from_pickle()
        elif self.pickle_path:
            logger.warning(f"Pickle file {self.pickle_path} not found, fetching from Yahoo Finance...")
            return self._fetch_from_yahoo()
        else:
            return self._fetch_from_yahoo()
    
    def _load_from_pickle(self) -> "DataLoader":
        """Load price data from a pickle file.
        
        The pickle file should contain a DataFrame with:
        - Rows: dates (DatetimeIndex)
        - Columns: ticker symbols
        - Values: adjusted close prices
        
        Returns:
            Self for method chaining.
        """
        logger.info(f"Loading data from pickle: {self.pickle_path}")
        
        try:
            df_prices = pd.read_pickle(self.pickle_path)
            
            # Validate structure
            if not isinstance(df_prices, pd.DataFrame):
                raise ValueError("Pickle file must contain a pandas DataFrame")
            
            if df_prices.empty:
                raise ValueError("Pickle file contains empty DataFrame")
            
            # Filter to only requested tickers if they exist in the pickle
            available_tickers = [t for t in self.tickers if t in df_prices.columns]
            if not available_tickers:
                logger.warning(f"No requested tickers found in pickle. Using all available: {list(df_prices.columns)}")
                available_tickers = list(df_prices.columns)
                self.tickers = available_tickers
            
            df_prices = df_prices[available_tickers]
            
            # Convert prices to OHLCV format (simplified - use close for all)
            for ticker in available_tickers:
                ticker_df = pd.DataFrame({
                    'Open': df_prices[ticker],
                    'High': df_prices[ticker],
                    'Low': df_prices[ticker],
                    'Close': df_prices[ticker],
                    'Volume': np.random.randint(1e6, 1e8, size=len(df_prices))  # Mock volume
                })
                self.data[ticker] = ticker_df
            
            logger.info(f"Loaded {len(available_tickers)} tickers from pickle with {len(df_prices)} rows")
            return self
            
        except Exception as e:
            logger.error(f"Error loading pickle: {e}")
            raise
    
    def _fetch_from_yahoo(self) -> "DataLoader":
        """Fetch historical OHLCV data from Yahoo Finance."""
        logger.info(
            f"Fetching data for {len(self.tickers)} tickers from "
            f"{self.start_date} to {self.end_date}"
        )
        
        for ticker in self.tickers:
            try:
                df = yf.download(
                    ticker,
                    start=self.start_date,
                    end=self.end_date,
                    progress=False,
                )
                
                # Handle multi-level columns if present (yfinance v0.2+)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                
                if df.empty:
                    logger.warning(f"No data fetched for {ticker}, skipping.")
                    continue
                    
                # Ensure required columns exist
                required_cols = ["Open", "High", "Low", "Close", "Volume"]
                missing_cols = set(required_cols) - set(df.columns)
                if missing_cols:
                    logger.warning(
                        f"Missing columns {missing_cols} for {ticker}, skipping."
                    )
                    continue
                
                self.data[ticker] = df
                logger.debug(f"Fetched {len(df)} rows for {ticker}")
                
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                continue
        
        if not self.data:
            raise ValueError("No data could be fetched for any ticker.")
        
        logger.info(f"Successfully fetched data for {len(self.data)} tickers.")
        return self
    
    def compute_indicators(self) -> "DataLoader":
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
                close = df["Close"]
                volume = df["Volume"]
                
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
                    "macd": macd_hist,  # Use MACD histogram
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
            price_dfs[ticker] = df["Close"]
        
        # Align all price series to common index
        aligned = pd.DataFrame({
            ticker: series.reindex(common_index)
            for ticker, series in price_dfs.items()
        })
        
        return aligned.dropna()


def load_and_prepare_data(
    tickers: Optional[List[str]] = None,
    start_date: str = "2020-01-01",
    end_date: Optional[str] = None,
    pickle_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DatetimeIndex]:
    """Convenience function to load and prepare all data.
    
    Args:
        tickers: List of ticker symbols.
        start_date: Start date for historical data.
        end_date: End date for historical data.
        pickle_path: Path to pickle file with pre-computed prices.
        
    Returns:
        Tuple of (indicators DataFrame, prices DataFrame, DatetimeIndex).
    """
    loader = DataLoader(
        tickers=tickers, 
        start_date=start_date, 
        end_date=end_date,
        pickle_path=pickle_path
    )
    loader.fetch_data()
    loader.compute_indicators()
    
    indicators, index = loader.get_aligned_data()
    prices = loader.get_prices()
    
    # Align prices to indicators index
    prices = prices.reindex(index).dropna()
    
    return indicators, prices, index


if __name__ == "__main__":
    # Test the DataLoader
    logging.basicConfig(level=logging.INFO)
    
    print("Testing DataLoader...")
    
    # Load default tickers
    loader = DataLoader()
    loader.fetch_data()
    loader.compute_indicators()
    
    indicators, index = loader.get_aligned_data()
    prices = loader.get_prices()
    
    print(f"\nIndicators shape: {indicators.shape}")
    print(f"Prices shape: {prices.shape}")
    print(f"Date range: {index[0]} to {index[-1]}")
    print(f"Number of tickers: {len(loader.data)}")
    
    print("\nIndicator columns sample:")
    print(indicators.columns[:10].tolist())
    
    print("\nPrice statistics:")
    print(prices.describe())
    
    print("\n✓ DataLoader test completed successfully!")
