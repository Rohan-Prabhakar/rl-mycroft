#!/usr/bin/env python3
"""
FastAPI backend for serving SAC portfolio predictions.

This service loads the trained SAC model and provides real-time
portfolio allocation predictions based on current market data.

Usage:
    python serve_model.py --model-path models/final_sac_model --port 8000
"""

import argparse
import logging
import os
import sys
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Add parent directory to path to allow imports when running from within rl_portfolio
# This fixes "ModuleNotFoundError: No module named 'rl_portfolio'"
current_dir = Path(__file__).parent.parent  # Go up one level to workspace root
sys.path.insert(0, str(current_dir))

from rl_portfolio.envs.mycroft_finance_env import MycroftFinanceEnv
from rl_portfolio.agents.sac_agent import SacAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Mycroft SAC Portfolio API")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model and environment
model_service = None


class PredictionRequest(BaseModel):
    tickers: Optional[List[str]] = None
    days_lookback: int = 30


class AllocationResponse(BaseModel):
    ticker: str
    weight: float
    companyName: str


class MetricsResponse(BaseModel):
    sharpeRatio: float
    cumulativeReturn: float
    maxDrawdown: float
    portfolioValue: float


class PortfolioResponse(BaseModel):
    success: bool
    allocations: List[AllocationResponse]
    metrics: MetricsResponse
    timestamp: str
    message: Optional[str] = None


class ModelService:
    """Service to manage SAC model lifecycle and predictions."""
    
    def __init__(
        self,
        model_path: str,
        pickle_path: str,
        tickers: List[str],
        device: str = "auto"
    ):
        self.model_path = model_path
        self.pickle_path = pickle_path
        self.tickers = tickers
        self.device = device
        
        logger.info(f"Loading SAC model from {model_path}")
        logger.info(f"Using data from {pickle_path}")
        
        # Initialize environment
        self.env = MycroftFinanceEnv(
            tickers=tickers,
            pickle_path=pickle_path,
            initial_capital=1_000_000.0,
            transaction_cost_rate=0.001,
            max_drawdown_limit=0.20,
        )
        
        # Reset environment to initialize spaces
        obs, info = self.env.reset()
        
        # Load trained model
        self.agent = SacAgent.load(model_path, env=self.env, device=device)
        
        logger.info(f"Model loaded successfully with {len(tickers)} tickers")
        
        # Company names mapping
        self.company_names = {
            'NVDA': 'NVIDIA Corporation',
            'MSFT': 'Microsoft Corporation',
            'GOOGL': 'Alphabet Inc.',
            'META': 'Meta Platforms Inc.',
            'AMZN': 'Amazon.com Inc.',
            'TSLA': 'Tesla Inc.',
            'AMD': 'Advanced Micro Devices',
            'INTC': 'Intel Corporation',
            'CRM': 'Salesforce Inc.',
            'ORCL': 'Oracle Corporation',
            'IBM': 'IBM Corporation',
            'PLTR': 'Palantir Technologies',
            'SNOW': 'Snowflake Inc.',
            'AI': 'C3.ai Inc.',
        }
    
    def get_prediction(self) -> Dict[str, Any]:
        """Get portfolio allocation prediction from SAC model."""
        try:
            # Get current observation
            obs = self.env._get_observation()
            
            # Get action from SAC model
            action = self.agent.predict(obs, deterministic=True)
            
            # Process action into weights
            weights = self.env._process_action(action)
            
            # Create allocations
            allocations = []
            for i, ticker in enumerate(self.env.tickers):
                weight = float(weights[i])
                if weight > 0.001:  # Only include meaningful allocations
                    allocations.append({
                        'ticker': ticker,
                        'weight': round(weight, 4),
                        'companyName': self.company_names.get(ticker, ticker)
                    })
            
            # Sort by weight descending
            allocations.sort(key=lambda x: x['weight'], reverse=True)
            
            # Calculate metrics from environment state
            metrics = self._calculate_metrics()
            
            return {
                'success': True,
                'allocations': allocations,
                'metrics': metrics,
                'timestamp': pd.Timestamp.now().isoformat(),
                'message': 'Portfolio allocation optimized by SAC agent'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            raise
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """Calculate portfolio metrics from environment state."""
        # Use environment's tracked values
        portfolio_value = self.env.portfolio_value
        peak_value = self.env.peak_value
        
        # Calculate drawdown
        drawdown = (peak_value - portfolio_value) / (peak_value + 1e-8)
        
        # Calculate cumulative return
        cum_return = (portfolio_value - self.env.initial_capital) / self.env.initial_capital
        
        # Calculate Sharpe ratio from returns history
        if len(self.env.returns_history) >= 20:
            recent_returns = self.env.returns_history[-20:]
            mean_ret = np.mean(recent_returns)
            std_ret = np.std(recent_returns)
            daily_rf = 0.02 / 252  # Risk-free rate
            sharpe = ((mean_ret - daily_rf) / (std_ret + 1e-8)) * np.sqrt(252)
        else:
            sharpe = 0.0
        
        return {
            'sharpeRatio': round(float(sharpe), 2),
            'cumulativeReturn': round(float(cum_return), 4),
            'maxDrawdown': round(float(drawdown), 4),
            'portfolioValue': round(float(portfolio_value), 2)
        }
    
    def get_tickers(self) -> List[Dict[str, str]]:
        """Get list of available tickers."""
        return [
            {'ticker': t, 'name': self.company_names.get(t, t)}
            for t in self.env.tickers
        ]


@app.on_event("startup")
async def startup_event():
    """Initialize model service on startup."""
    global model_service
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="rl_portfolio/models/final_sac_model")
    parser.add_argument("--pickle-path", type=str, default="rl_portfolio/data/sp500_full.pkl")
    parser.add_argument("--tickers", type=str, nargs="+", default=None)
    parser.add_argument("--device", type=str, default="auto")
    
    # Parse known args only (ignore unknown from uvicorn)
    args, _ = parser.parse_known_args()
    
    # Check if files exist
    model_exists = Path(args.model_path).exists() or Path(args.model_path + ".zip").exists()
    pickle_exists = Path(args.pickle_path).exists()
    
    if not model_exists:
        logger.warning(f"Model not found at {args.model_path}. Running in simulation mode.")
        model_service = None
        return
    
    if not pickle_exists:
        logger.warning(f"Pickle data not found at {args.pickle_path}. Running in simulation mode.")
        model_service = None
        return
    
    try:
        tickers = args.tickers if args.tickers else None
        model_service = ModelService(
            model_path=args.model_path,
            pickle_path=args.pickle_path,
            tickers=tickers,
            device=args.device
        )
        logger.info("Model service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model service: {e}")
        model_service = None


@app.get("/api/portfolio", response_model=PortfolioResponse)
async def get_portfolio(action: str = "predict"):
    """Get portfolio allocation from SAC model."""
    global model_service
    
    if action == "predict":
        if model_service is None:
            # Return simulated data if model not loaded
            return _get_simulated_response()
        
        try:
            result = model_service.get_prediction()
            return PortfolioResponse(**result)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Fallback to simulation
            return _get_simulated_response()
    
    elif action == "metrics":
        if model_service is None:
            metrics = {
                'sharpeRatio': 1.85,
                'cumulativeReturn': 0.234,
                'maxDrawdown': 0.087,
                'portfolioValue': 123400.00
            }
        else:
            metrics = model_service._calculate_metrics()
        
        return {
            'success': True,
            'allocations': [],
            'metrics': metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    elif action == "tickers":
        if model_service is None:
            tickers = [
                {'ticker': 'NVDA', 'name': 'NVIDIA Corporation'},
                {'ticker': 'MSFT', 'name': 'Microsoft Corporation'},
                {'ticker': 'GOOGL', 'name': 'Alphabet Inc.'},
            ]
        else:
            tickers = model_service.get_tickers()
        
        return {
            'success': True,
            'tickers': tickers,
            'timestamp': pd.Timestamp.now().isoformat()
        }
    
    else:
        raise HTTPException(status_code=400, detail="Invalid action")


def _get_simulated_response() -> PortfolioResponse:
    """Return simulated portfolio data when model is not available."""
    allocations = [
        {'ticker': 'NVDA', 'weight': 0.18, 'companyName': 'NVIDIA Corporation'},
        {'ticker': 'MSFT', 'weight': 0.15, 'companyName': 'Microsoft Corporation'},
        {'ticker': 'GOOGL', 'weight': 0.12, 'companyName': 'Alphabet Inc.'},
        {'ticker': 'META', 'weight': 0.10, 'companyName': 'Meta Platforms Inc.'},
        {'ticker': 'AMZN', 'weight': 0.10, 'companyName': 'Amazon.com Inc.'},
        {'ticker': 'TSLA', 'weight': 0.08, 'companyName': 'Tesla Inc.'},
        {'ticker': 'AMD', 'weight': 0.07, 'companyName': 'Advanced Micro Devices'},
        {'ticker': 'INTC', 'weight': 0.05, 'companyName': 'Intel Corporation'},
        {'ticker': 'CRM', 'weight': 0.05, 'companyName': 'Salesforce Inc.'},
        {'ticker': 'ORCL', 'weight': 0.04, 'companyName': 'Oracle Corporation'},
        {'ticker': 'IBM', 'weight': 0.03, 'companyName': 'IBM Corporation'},
        {'ticker': 'PLTR', 'weight': 0.01, 'companyName': 'Palantir Technologies'},
        {'ticker': 'SNOW', 'weight': 0.01, 'companyName': 'Snowflake Inc.'},
        {'ticker': 'AI', 'weight': 0.01, 'companyName': 'C3.ai Inc.'},
    ]
    
    metrics = {
        'sharpeRatio': 1.85,
        'cumulativeReturn': 0.234,
        'maxDrawdown': 0.087,
        'portfolioValue': 123400.00
    }
    
    return PortfolioResponse(
        success=True,
        allocations=allocations,
        metrics=metrics,
        timestamp=pd.Timestamp.now().isoformat(),
        message='Simulated data (model not loaded)'
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    
    uvicorn.run(
        "serve_model:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )
