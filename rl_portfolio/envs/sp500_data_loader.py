#!/usr/bin/env python3
"""
Load S&P 500 stock data from Kaggle dataset CSV files.

This module provides:
1. Original SP500DataLoader class for loading from CSV
2. Fixed create_mycroft_pickle function for creating pickle with all 500 tickers
3. load_sp500_data convenience function

Dataset: https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks 

Expected files in data directory:
1. sp500_stocks.csv - Main price data (Date, Symbol, Open, High, Low, Close, Adj Close, Volume)
2. sp500_companies.csv - Company metadata (Symbol, Security, GICS Sector, etc.)
3. sp500_index.csv - Index-level data (Date, S&P500)
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import os
import pickle

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_sp500_tickers_from_dataset(data_dir: str) -> List[str]:
    """Extract available ticker symbols from sp500_stocks.csv."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    stocks_file = data_path / "sp500_stocks.csv"
    if not stocks_file.exists():
        raise FileNotFoundError(f"sp500_stocks.csv not found in {data_dir}")

    logger.info("Reading tickers from sp500_stocks.csv")
    try:
        df = pd.read_csv(stocks_file, usecols=["Symbol"], nrows=50000)
        tickers = df["Symbol"].unique().tolist()
        return sorted([str(t).upper() for t in tickers])
    except Exception as e:
        logger.error(f"Could not read sp500_stocks.csv: {e}")
        raise


class SP500DataLoader:
    """Load and preprocess S&P 500 stock data from Kaggle dataset."""

    def __init__(
        self,
        data_dir: str,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> None:
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

        if self.tickers is None:
            self.tickers = get_sp500_tickers_from_dataset(str(self.data_dir))
            logger.info(f"Discovered {len(self.tickers)} tickers in dataset")

    def fetch_data(self) -> "SP500DataLoader":
        """Load historical OHLCV data from sp500_stocks.csv."""
        stocks_file = self.data_dir / "sp500_stocks.csv"
        logger.info(f"Loading data from sp500_stocks.csv")
        return self._load_stocks_data(stocks_file)

    def _load_stocks_data(self, csv_path: Path) -> "SP500DataLoader":
        """Load data from sp500_stocks.csv."""
        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip().str.lower()

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])

            if self.start_date or self.end_date:
                mask = pd.Series(True, index=df.index)
                if self.start_date:
                    mask &= df["date"] >= pd.to_datetime(self.start_date)
                if self.end_date:
                    mask &= df["date"] <= pd.to_datetime(self.end_date)
                df = df[mask]

            numeric_cols = ["open", "high", "low", "close", "adj close", "volume"]
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            df = df.dropna(subset=["open", "high", "low", "close", "volume"])

            if "adj close" in df.columns:
                df["close"] = df["adj close"]

            loaded_count = 0
            target_tickers = set([t.upper() for t in self.tickers]) if self.tickers else None

            for symbol in df["symbol"].unique():
                symbol_upper = symbol.upper()
                if target_tickers and symbol_upper not in target_tickers:
                    continue

                symbol_df = df[df["symbol"] == symbol].copy()
                symbol_df = symbol_df.set_index("date")
                symbol_df = symbol_df.sort_index()

                keep_cols = ["open", "high", "low", "close", "volume"]
                symbol_df = symbol_df[keep_cols]

                self.data[symbol_upper] = symbol_df
                loaded_count += 1

            if not self.data:
                raise ValueError("No data could be loaded for any ticker.")

            logger.info(f"Successfully loaded data for {loaded_count} tickers.")
            return self

        except Exception as e:
            logger.error(f"Error loading data from sp500_stocks.csv: {e}")
            raise

    def compute_indicators(self) -> "SP500DataLoader":
        """Compute technical indicators for all tickers."""
        if not self.data:
            raise ValueError("Must call fetch_data() before compute_indicators().")

        logger.info("Computing technical indicators...")

        indicator_dfs = {}

        for ticker, df in self.data.items():
            try:
                close = df["close"]
                volume = df["volume"]

                returns = close.pct_change()
                volatility = returns.rolling(window=20).std()

                volume_mean = volume.rolling(window=20).mean()
                volume_std = volume.rolling(window=20).std()
                volume_zscore = (volume - volume_mean) / (volume_std + 1e-8)

                delta = close.diff()
                gain = delta.where(delta > 0, 0.0)
                loss = -delta.where(delta < 0, 0.0)
                avg_gain = gain.rolling(window=14).mean()
                avg_loss = loss.rolling(window=14).mean()
                rs = avg_gain / (avg_loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))

                ema12 = close.ewm(span=12, adjust=False).mean()
                ema26 = close.ewm(span=26, adjust=False).mean()
                macd = ema12 - ema26
                signal = macd.ewm(span=9, adjust=False).mean()
                macd_hist = macd - signal

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

        combined = pd.concat(indicator_dfs, axis=1)
        combined = combined.dropna()

        self.combined_data = combined
        logger.info(f"Computed indicators. Data shape: {combined.shape}")

        return self

    def get_aligned_data(self) -> Tuple[pd.DataFrame, pd.DatetimeIndex]:
        """Get aligned data ready for environment use."""
        if self.combined_data is None:
            raise ValueError("Must call fetch_data() and compute_indicators() first.")

        return self.combined_data, self.combined_data.index

    def get_prices(self) -> pd.DataFrame:
        """Get aligned closing prices for all tickers."""
        if not self.data:
            raise ValueError("Must call fetch_data() first.")

        common_index = None
        price_dfs = {}

        for ticker, df in self.data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
            price_dfs[ticker] = df["close"]

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
    """Convenience function to load and prepare S&P 500 data."""
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

    prices = prices.reindex(index).dropna()

    return indicators, prices, index


def create_mycroft_pickle(data_dir: str, output_path: str):
    """Create pickle with ALL tickers AND all 5 indicators per ticker."""
    from pandas import MultiIndex
    
    csv_path = Path(data_dir) / "sp500_stocks.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    logger.info(f"Reading {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'])

    tickers = sorted(df['symbol'].unique())
    logger.info(f"Found {len(tickers)} tickers")

    indicator_dfs = {}
    all_prices = {}
    valid_tickers = []

    for ticker in tickers:
        try:
            ticker_df = df[df['symbol'] == ticker].copy()
            ticker_df = ticker_df.set_index('date').sort_index()

            if len(ticker_df) < 30:
                logger.warning(f"Skipping {ticker}: only {len(ticker_df)} days")
                continue

            close_col = 'adj close' if 'adj close' in ticker_df.columns else 'close'
            close = ticker_df[close_col]
            volume = ticker_df['volume']

            returns = close.pct_change().fillna(0)
            volatility = returns.rolling(window=20, min_periods=1).std().fillna(0)
            
            volume_mean = volume.rolling(window=20, min_periods=1).mean()
            volume_std = volume.rolling(window=20, min_periods=1).std()
            volume_zscore = ((volume - volume_mean) / (volume_std + 1e-8)).fillna(0)
            
            delta = close.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(window=14, min_periods=1).mean()
            avg_loss = loss.rolling(window=14, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = (100 - (100 / (1 + rs))).fillna(50)
            
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9, adjust=False).mean()
            macd_hist = (macd - signal).fillna(0)

            indicators = pd.DataFrame({
                "returns": returns,
                "volatility": volatility,
                "volume_zscore": volume_zscore,
                "rsi": rsi,
                "macd": macd_hist,
            }, index=ticker_df.index)
            
            if indicators.isna().any().any():
                indicators = indicators.fillna(0)
                
            indicator_dfs[ticker] = indicators
            all_prices[ticker] = close.ffill().bfill()
            valid_tickers.append(ticker)

        except Exception as e:
            logger.warning(f"Error processing {ticker}: {e}")
            continue

    if not valid_tickers:
        raise ValueError("No valid tickers found!")

    logger.info(f"Processed {len(valid_tickers)} tickers")

    indicators_df = pd.concat(indicator_dfs, axis=1)
    prices_df = pd.DataFrame(all_prices)

    logger.info(f"Indicators shape: {indicators_df.shape}")
    
    common_index = indicators_df.index.intersection(prices_df.index)
    indicators_df = indicators_df.loc[common_index].dropna(how='all')
    prices_df = prices_df.loc[common_index].dropna(how='all')

    if len(indicators_df) == 0:
        raise ValueError("No data left after cleaning!")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        'indicators': indicators_df,
        'prices': prices_df,
        'dates': indicators_df.index,
        'tickers': valid_tickers
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    logger.info(f"✅ Saved to {output_path}")
    logger.info(f"   Tickers: {len(valid_tickers)}, Days: {len(indicators_df)}")
    
    return data


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['load', 'convert'], default='convert')
    parser.add_argument("--data_dir", default="rl_portfolio/data/sp500_stocks")
    parser.add_argument("--output", default="rl_portfolio/data/sp500_full.pkl")
    args = parser.parse_args()

    if args.mode == 'convert':
        create_mycroft_pickle(args.data_dir, args.output)
    else:
        tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
        indicators, prices, dates = load_sp500_data(args.data_dir, tickers)
        print(f"Loaded {len(dates)} days for {len(prices.columns)} tickers")
