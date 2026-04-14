"""FastAPI Inference Endpoint for Portfolio Allocation.

This module provides a REST API for:
- Getting portfolio allocation predictions from trained SAC model
- Submitting feedback (realized returns) for online logging
- Health checks with model status and system info

Endpoints:
    /predict: POST - Get allocation weights + risk metrics
    /feedback: POST - Submit realized returns for logging
    /health: GET - Model status, env info, uptime

Usage:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Test Examples:
    # Health check
    curl http://localhost:8000/health

    # Predict endpoint
    curl -X POST http://localhost:8000/predict \\
      -H "Content-Type: application/json" \\
      -d '{
        "market_state": {
          "returns": [0.01, -0.02, 0.015, 0.005, -0.01],
          "volatilities": [0.15, 0.20, 0.18, 0.12, 0.25],
          "correlations": [[1.0, 0.5, 0.3], [0.5, 1.0, 0.4], [0.3, 0.4, 1.0]]
        },
        "portfolio_state": {
          "current_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
          "cash_ratio": 0.1,
          "total_value": 100000.0
        }
      }'

    # Feedback endpoint
    curl -X POST http://localhost:8000/feedback \\
      -H "Content-Type: application/json" \\
      -d '{
        "timestamp": "2024-01-15T10:30:00",
        "realized_return": 0.025,
        "action_taken": [0.25, 0.15, 0.20, 0.25, 0.15],
        "market_conditions": "bull"
      }'
"""

import json
import logging
import math
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import reward shaping constants for consistency
try:
    from orchestration.reward_shaping import (
        RISK_FREE_RATE,
        TRANSACTION_COST_RATE,
        SHARPE_RATIO_TARGET,
        MAX_POSITION_WEIGHT,
        MIN_POSITION_WEIGHT,
        TURNOVER_PENALTY_FACTOR,
    )
except ImportError:
    # Fallback defaults if orchestration module not available
    RISK_FREE_RATE = 0.02
    TRANSACTION_COST_RATE = 0.001
    SHARPE_RATIO_TARGET = 1.5
    MAX_POSITION_WEIGHT = 0.4
    MIN_POSITION_WEIGHT = 0.0
    TURNOVER_PENALTY_FACTOR = 0.5

# ============================================================================
# Pydantic Models for Request/Response Validation
# ============================================================================


class MarketState(BaseModel):
    """Current market state input.

    Attributes:
        returns: List of recent returns for each asset
        volatilities: List of volatility measures for each asset
        correlations: Correlation matrix (optional, can be nested list or flat)
        prices: Current asset prices (optional)
        trend_indicators: Technical indicators (optional)
    """

    returns: List[float] = Field(..., description="Recent returns for each asset")
    volatilities: List[float] = Field(..., description="Volatility for each asset")
    correlations: Optional[List[List[float]]] = Field(
        None, description="Asset correlation matrix"
    )
    prices: Optional[List[float]] = Field(None, description="Current asset prices")
    trend_indicators: Optional[Dict[str, float]] = Field(
        None, description="Technical trend indicators"
    )

    @field_validator("returns", "volatilities")
    @classmethod
    def validate_non_empty(cls, v: List[float]) -> List[float]:
        """Validate that lists are not empty."""
        if not v:
            raise ValueError("List cannot be empty")
        return v

    @field_validator("returns", "volatilities")
    @classmethod
    def validate_no_nan(cls, v: List[float]) -> List[float]:
        """Validate that values are not NaN or Inf."""
        for i, val in enumerate(v):
            if math.isnan(val) or math.isinf(val):
                raise ValueError(f"Value at index {i} is NaN or Inf: {val}")
        return v

    @field_validator("correlations")
    @classmethod
    def validate_correlations(
        cls, v: Optional[List[List[float]]]
    ) -> Optional[List[List[float]]]:
        """Validate correlation matrix if provided."""
        if v is None:
            return v
        # Check for NaN/Inf
        for i, row in enumerate(v):
            for j, val in enumerate(row):
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(
                        f"Correlation at [{i}][{j}] is NaN or Inf: {val}"
                    )
                if val < -1.0 or val > 1.0:
                    raise ValueError(
                        f"Correlation at [{i}][{j}] out of range [-1, 1]: {val}"
                    )
        return v


class PortfolioState(BaseModel):
    """Current portfolio state input.

    Attributes:
        current_weights: Current allocation weights (must sum to ~1.0)
        cash_ratio: Fraction of portfolio in cash
        total_value: Total portfolio value
        last_rebalance: Timestamp of last rebalance (optional)
    """

    current_weights: List[float] = Field(
        ..., description="Current allocation weights for each asset"
    )
    cash_ratio: float = Field(
        0.0, ge=0.0, le=1.0, description="Fraction of portfolio in cash"
    )
    total_value: float = Field(..., gt=0, description="Total portfolio value")
    last_rebalance: Optional[str] = Field(
        None, description="ISO timestamp of last rebalance"
    )

    @field_validator("current_weights")
    @classmethod
    def validate_weights(cls, v: List[float]) -> List[float]:
        """Validate weights are valid probabilities."""
        if not v:
            raise ValueError("Weights list cannot be empty")
        for i, w in enumerate(v):
            if math.isnan(w) or math.isinf(w):
                raise ValueError(f"Weight at index {i} is NaN or Inf: {w}")
            if w < 0.0 or w > 1.0:
                raise ValueError(f"Weight at index {i} out of range [0, 1]: {w}")
        # Check sum is approximately 1.0 (allowing for cash)
        total = sum(v)
        if total < 0.9 or total > 1.1:
            logger.warning(f"Weights sum to {total}, expected ~1.0")
        return v


class PredictRequest(BaseModel):
    """Request model for /predict endpoint.

    Attributes:
        market_state: Current market conditions
        portfolio_state: Current portfolio state
        model_id: Optional model identifier for multi-model deployments
        include_risk_metrics: Whether to include detailed risk metrics
    """

    market_state: MarketState
    portfolio_state: PortfolioState
    model_id: Optional[str] = Field(None, description="Model identifier")
    include_risk_metrics: bool = Field(True, description="Include risk metrics")


class RiskMetrics(BaseModel):
    """Risk metrics output.

    Attributes:
        portfolio_volatility: Estimated portfolio volatility
        sharpe_ratio: Estimated Sharpe ratio
        max_drawdown: Estimated maximum drawdown
        var_95: Value at Risk at 95% confidence
        turnover: Expected turnover from rebalancing
        concentration: Herfindahl index of concentration
    """

    portfolio_volatility: float = Field(..., description="Portfolio volatility")
    sharpe_ratio: float = Field(..., description="Sharpe ratio estimate")
    max_drawdown: float = Field(..., description="Maximum drawdown estimate")
    var_95: float = Field(..., description="Value at Risk (95%)")
    turnover: float = Field(..., description="Expected turnover")
    concentration: float = Field(..., description="Concentration index")


class PredictResponse(BaseModel):
    """Response model for /predict endpoint.

    Attributes:
        allocation_weights: Recommended allocation weights
        risk_metrics: Detailed risk metrics
        model_used: Identifier of model used
        confidence: Model confidence score
        timestamp: Response timestamp
        fallback_used: Whether fallback strategy was used
    """

    allocation_weights: List[float] = Field(
        ..., description="Recommended allocation weights"
    )
    risk_metrics: Optional[RiskMetrics] = Field(
        None, description="Detailed risk metrics"
    )
    model_used: str = Field(..., description="Model identifier used")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    timestamp: str = Field(..., description="ISO timestamp")
    fallback_used: bool = Field(..., description="Whether fallback was used")


class FeedbackRequest(BaseModel):
    """Request model for /feedback endpoint.

    Attributes:
        timestamp: When the action was taken
        realized_return: Actual return achieved
        action_taken: Action that was executed
        market_conditions: Market regime descriptor
        metadata: Additional context (optional)
    """

    timestamp: str = Field(..., description="ISO timestamp of action")
    realized_return: float = Field(..., description="Realized return")
    action_taken: List[float] = Field(..., description="Action executed")
    market_conditions: Optional[str] = Field(
        None, description="Market regime (bull/bear/neutral)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        None, description="Additional metadata"
    )

    @field_validator("realized_return")
    @classmethod
    def validate_return(cls, v: float) -> float:
        """Validate return value."""
        if math.isnan(v) or math.isinf(v):
            raise ValueError("Realized return cannot be NaN or Inf")
        return v


class FeedbackResponse(BaseModel):
    """Response model for /feedback endpoint.

    Attributes:
        status: Operation status
        message: Descriptive message
        entries_logged: Total number of feedback entries
        timestamp: Response timestamp
    """

    status: str
    message: str
    entries_logged: int
    timestamp: str


class HealthResponse(BaseModel):
    """Response model for /health endpoint.

    Attributes:
        status: Overall health status
        model_loaded: Whether model is loaded
        model_path: Path to loaded model
        env_info: Environment information
        uptime_seconds: Service uptime in seconds
        version: API version
        timestamp: Response timestamp
    """

    status: str
    model_loaded: bool
    model_path: Optional[str]
    env_info: Dict[str, Any]
    uptime_seconds: float
    version: str
    timestamp: str


# ============================================================================
# FastAPI Application Setup
# ============================================================================

# Create FastAPI app
app = FastAPI(
    title="Portfolio Allocation API",
    description="""
    FastAPI-based inference endpoint for portfolio allocation using SAC reinforcement learning.

    ## Features
    * **Predict**: Get optimal allocation weights from trained model
    * **Feedback**: Submit realized returns for online learning
    * **Health**: Monitor model status and system health

    ## Authentication
    Currently no authentication required. Add rate limiting and auth as needed.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React frontend default
        "http://localhost:8080",  # Vue frontend default
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
        "*",  # Allow all origins in development; restrict in production
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ============================================================================
# Global State
# ============================================================================

START_TIME = time.time()
MODEL_PATH = os.getenv("MODEL_PATH", "models/sac_portfolio_model.zip")
FEEDBACK_LOG_PATH = os.getenv(
    "FEEDBACK_LOG_PATH", "logs/feedback_log.jsonl"
)

# Model instance (loaded on startup)
model_instance = None
model_loaded = False


def load_model() -> Optional[Any]:
    """Load trained SAC model from disk.

    Returns:
        Loaded model instance or None if loading fails
    """
    global model_instance, model_loaded

    try:
        from stable_baselines3 import SAC

        if os.path.exists(MODEL_PATH):
            model_instance = SAC.load(MODEL_PATH)
            model_loaded = True
            logger.info("Successfully loaded model from %s", MODEL_PATH)
            return model_instance
        else:
            logger.warning(
                "Model file not found at %s. Using equal-weight fallback.", MODEL_PATH
            )
            model_loaded = False
            return None
    except ImportError:
        logger.warning(
            "stable_baselines3 not installed. Using equal-weight fallback."
        )
        model_loaded = False
        return None
    except Exception as e:
        logger.error("Failed to load model: %s", str(e))
        model_loaded = False
        return None


def get_equal_weight_allocation(n_assets: int) -> List[float]:
    """Generate equal-weight allocation as fallback.

    Args:
        n_assets: Number of assets in portfolio

    Returns:
        List of equal weights summing to 1.0
    """
    weight = 1.0 / n_assets
    return [weight] * n_assets


def calculate_risk_metrics(
    weights: List[float],
    volatilities: List[float],
    returns: List[float],
    prev_weights: Optional[List[float]] = None,
) -> RiskMetrics:
    """Calculate portfolio risk metrics.

    Args:
        weights: Portfolio weights
        volatilities: Asset volatilities
        returns: Asset returns
        prev_weights: Previous weights for turnover calculation

    Returns:
        RiskMetrics object with calculated metrics
    """
    n = len(weights)

    # Portfolio volatility (simplified: assuming zero correlation for baseline)
    port_var = sum(w**2 * v**2 for w, v in zip(weights, volatilities))
    port_vol = math.sqrt(port_var)

    # Portfolio return
    port_return = sum(w * r for w, r in zip(weights, returns))

    # Sharpe ratio
    excess_return = port_return - RISK_FREE_RATE / 252  # Daily risk-free rate
    sharpe = (excess_return * 252) / (port_vol * math.sqrt(252)) if port_vol > 0 else 0.0

    # Max drawdown estimate (simplified)
    max_dd = port_vol * 2.0  # Rough estimate

    # VaR at 95% (assuming normal distribution)
    var_95 = port_vol * 1.645

    # Turnover
    if prev_weights:
        turnover = sum(abs(w - pw) for w, pw in zip(weights, prev_weights)) / 2.0
    else:
        turnover = 0.0

    # Concentration (Herfindahl index)
    concentration = sum(w**2 for w in weights)

    return RiskMetrics(
        portfolio_volatility=round(port_vol, 6),
        sharpe_ratio=round(sharpe, 4),
        max_drawdown=round(max_dd, 6),
        var_95=round(var_95, 6),
        turnover=round(turnover, 4),
        concentration=round(concentration, 4),
    )


def log_feedback_entry(entry: Dict[str, Any]) -> int:
    """Append feedback entry to JSONL log file.

    Args:
        entry: Feedback data dictionary

    Returns:
        Total number of entries in log file
    """
    log_path = Path(FEEDBACK_LOG_PATH)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Append entry
    with open(log_path, "a") as f:
        json.dump(entry, f)
        f.write("\n")

    # Count total entries
    count = 0
    with open(log_path, "r") as f:
        for _ in f:
            count += 1

    return count


# Rate limiting stub (simple in-memory implementation)
request_counts: Dict[str, List[float]] = {}
RATE_LIMIT_REQUESTS = 100  # requests per window
RATE_LIMIT_WINDOW = 60  # seconds


def check_rate_limit(client_ip: str) -> bool:
    """Check if client has exceeded rate limit.

    Args:
        client_ip: Client IP address

    Returns:
        True if request is allowed, False if rate limited
    """
    current_time = time.time()
    window_start = current_time - RATE_LIMIT_WINDOW

    if client_ip not in request_counts:
        request_counts[client_ip] = []

    # Remove old requests outside window
    request_counts[client_ip] = [
        t for t in request_counts[client_ip] if t > window_start
    ]

    # Check limit
    if len(request_counts[client_ip]) >= RATE_LIMIT_REQUESTS:
        return False

    # Record this request
    request_counts[client_ip].append(current_time)
    return True


# ============================================================================
# Event Handlers
# ============================================================================


@app.on_event("startup")
async def startup_event():
    """Load model on application startup."""
    logger.info("Starting Portfolio Allocation API...")
    load_model()
    logger.info("Startup complete. Model loaded: %s", model_loaded)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down Portfolio Allocation API...")
    uptime = time.time() - START_TIME
    logger.info("Total uptime: %.2f seconds", uptime)


# ============================================================================
# API Endpoints
# ============================================================================


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint.

    Returns model status, environment info, and uptime.

    Example:
        curl http://localhost:8000/health
    """
    uptime = time.time() - START_TIME

    env_info = {
        "python_version": os.popen("python --version").read().strip(),
        "platform": os.uname().sysname if hasattr(os, "uname") else "unknown",
        "model_path": MODEL_PATH,
        "feedback_log_path": FEEDBACK_LOG_PATH,
        "reward_shaping": {
            "risk_free_rate": RISK_FREE_RATE,
            "transaction_cost_rate": TRANSACTION_COST_RATE,
            "sharpe_target": SHARPE_RATIO_TARGET,
            "max_position": MAX_POSITION_WEIGHT,
            "min_position": MIN_POSITION_WEIGHT,
        },
    }

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_path=MODEL_PATH if model_loaded else None,
        env_info=env_info,
        uptime_seconds=round(uptime, 2),
        version="1.0.0",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    responses={
        200: {"description": "Successful prediction"},
        400: {"description": "Invalid input"},
        500: {"description": "Model error"},
        503: {"description": "Service unavailable"},
    },
)
async def predict(request: PredictRequest, req: Request):
    """Get portfolio allocation prediction.

    Accepts current market and portfolio state, returns optimal allocation
    weights along with risk metrics.

    Args:
        request: PredictRequest with market and portfolio state
        req: FastAPI request object for client info

    Returns:
        PredictResponse with allocation weights and risk metrics

    Raises:
        HTTPException: For invalid inputs or model errors
    """
    # Rate limiting check
    client_ip = req.client.host if req.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )

    try:
        # Validate inputs for NaN/Inf
        market = request.market_state
        portfolio = request.portfolio_state

        # Check for NaN in outputs
        if any(math.isnan(r) for r in market.returns):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Market returns contain NaN values",
            )

        if any(math.isnan(v) for v in market.volatilities):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Volatilities contain NaN values",
            )

        n_assets = len(market.returns)

        # Generate prediction
        fallback_used = False
        confidence = 0.0

        if model_loaded and model_instance is not None:
            try:
                # Prepare observation for model
                # Note: This assumes a specific observation space structure
                # Adjust based on your actual env observation space
                obs = market.returns + market.volatilities + portfolio.current_weights

                # Get action from model
                action, _ = model_instance.predict(obs, deterministic=True)

                # Post-process action to get valid weights
                # Apply softmax to ensure valid probability distribution
                import numpy as np

                action_array = np.array(action[:n_assets])
                exp_action = np.exp(action_array - np.max(action_array))  # Numerical stability
                weights = (exp_action / exp_action.sum()).tolist()

                confidence = 0.85  # Placeholder; implement actual confidence metric
                logger.info("Model prediction successful for %d assets", n_assets)

            except Exception as e:
                logger.warning("Model prediction failed: %s. Using fallback.", str(e))
                weights = get_equal_weight_allocation(n_assets)
                fallback_used = True
                confidence = 0.5
        else:
            weights = get_equal_weight_allocation(n_assets)
            fallback_used = True
            confidence = 0.5
            logger.info("Using equal-weight fallback for %d assets", n_assets)

        # Calculate risk metrics if requested
        risk_metrics = None
        if request.include_risk_metrics:
            risk_metrics = calculate_risk_metrics(
                weights=weights,
                volatilities=market.volatilities,
                returns=market.returns,
                prev_weights=portfolio.current_weights,
            )

        return PredictResponse(
            allocation_weights=[round(w, 6) for w in weights],
            risk_metrics=risk_metrics,
            model_used="SAC" if not fallback_used else "EqualWeight",
            confidence=confidence,
            timestamp=datetime.utcnow().isoformat(),
            fallback_used=fallback_used,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Prediction error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post(
    "/feedback",
    response_model=FeedbackResponse,
    tags=["Feedback"],
    responses={
        200: {"description": "Feedback logged successfully"},
        400: {"description": "Invalid input"},
        500: {"description": "Logging error"},
    },
)
async def submit_feedback(request: FeedbackRequest, req: Request):
    """Submit realized returns for online logging.

    Accepts feedback about actual outcomes to improve model over time.
    Data is appended to a JSONL file for later analysis/retraining.

    Args:
        request: FeedbackRequest with realized return data
        req: FastAPI request object for client info

    Returns:
        FeedbackResponse with confirmation

    Raises:
        HTTPException: For invalid inputs or logging errors
    """
    # Rate limiting check
    client_ip = req.client.host if req.client else "unknown"
    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )

    try:
        # Validate timestamp format
        try:
            datetime.fromisoformat(request.timestamp.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid timestamp format. Use ISO 8601 format.",
            )

        # Prepare log entry
        entry = {
            "timestamp": request.timestamp,
            "received_at": datetime.utcnow().isoformat(),
            "realized_return": request.realized_return,
            "action_taken": request.action_taken,
            "market_conditions": request.market_conditions,
            "metadata": request.metadata or {},
        }

        # Log entry
        entries_logged = log_feedback_entry(entry)

        logger.info(
            "Feedback logged: return=%.4f, entries_total=%d",
            request.realized_return,
            entries_logged,
        )

        return FeedbackResponse(
            status="success",
            message="Feedback logged successfully",
            entries_logged=entries_logged,
            timestamp=datetime.utcnow().isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Feedback logging error: %s", str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to log feedback: {str(e)}",
        )


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Portfolio Allocation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict (POST)",
        "feedback": "/feedback (POST)",
    }


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
