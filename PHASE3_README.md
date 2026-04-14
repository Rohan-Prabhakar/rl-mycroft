# Phase 3: FastAPI Inference Endpoint + Frontend Bridge

## Files Created

1. **`api/__init__.py`** - API module initialization
2. **`api/main.py`** - FastAPI application with `/predict`, `/feedback`, `/health` endpoints
3. **`orchestration/__init__.py`** - Orchestration module initialization  
4. **`orchestration/reward_shaping.py`** - Shared reward shaping constants and functions

## Features Implemented

### FastAPI Endpoints

- **`GET /health`** - Returns model status, environment info, uptime
- **`POST /predict`** - Accepts market + portfolio state, returns allocation weights + risk metrics
- **`POST /feedback`** - Accepts realized returns for online logging (appends to JSONL)
- **`GET /`** - Root endpoint with API information

### Key Capabilities

- ✅ Pydantic models for request/response validation
- ✅ Load trained SAC model on startup; fallback to equal-weight if missing
- ✅ Request validation (NaN/Inf checks, range validation)
- ✅ Rate limiting stub (in-memory, configurable)
- ✅ CORS configuration for frontend integration
- ✅ Comprehensive error handling
- ✅ Swagger/OpenAPI docs at `/docs` and `/redoc`
- ✅ Reward shaping module reusable by both env and API

---

## Startup Command

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Or run directly:
```bash
python -m api.main
```

---

## API Test Commands

### 1. Health Check

```bash
curl http://localhost:8000/health | jq
```

Expected response:
```json
{
  "status": "degraded",
  "model_loaded": false,
  "model_path": null,
  "env_info": {...},
  "uptime_seconds": 0.5,
  "version": "1.0.0",
  "timestamp": "2024-01-15T10:30:00"
}
```

### 2. Predict Endpoint

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "market_state": {
      "returns": [0.01, -0.02, 0.015, 0.005, -0.01],
      "volatilities": [0.15, 0.20, 0.18, 0.12, 0.25]
    },
    "portfolio_state": {
      "current_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
      "cash_ratio": 0.1,
      "total_value": 100000.0
    }
  }' | jq
```

Expected response:
```json
{
  "allocation_weights": [0.2, 0.2, 0.2, 0.2, 0.2],
  "risk_metrics": {
    "portfolio_volatility": 0.085,
    "sharpe_ratio": 0.1234,
    "max_drawdown": 0.17,
    "var_95": 0.14,
    "turnover": 0.0,
    "concentration": 0.2
  },
  "model_used": "EqualWeight",
  "confidence": 0.5,
  "timestamp": "2024-01-15T10:30:00",
  "fallback_used": true
}
```

### 3. Feedback Endpoint

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15T10:30:00",
    "realized_return": 0.025,
    "action_taken": [0.25, 0.15, 0.20, 0.25, 0.15],
    "market_conditions": "bull"
  }' | jq
```

Expected response:
```json
{
  "status": "success",
  "message": "Feedback logged successfully",
  "entries_logged": 1,
  "timestamp": "2024-01-15T10:30:00"
}
```

### 4. View Swagger Docs

Open in browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc
- OpenAPI JSON: http://localhost:8000/openapi.json

---

## Validation Examples

### Valid Request
```json
{
  "market_state": {
    "returns": [0.01, 0.02, -0.01],
    "volatilities": [0.15, 0.20, 0.18]
  },
  "portfolio_state": {
    "current_weights": [0.33, 0.34, 0.33],
    "total_value": 100000.0
  }
}
```

### Invalid Requests (will return 400 errors)

**Empty lists:**
```json
{
  "market_state": {
    "returns": [],
    "volatilities": []
  },
  ...
}
```

**NaN values:**
```json
{
  "market_state": {
    "returns": [0.01, null, 0.02],
    "volatilities": [0.15, 0.20, 0.18]
  },
  ...
}
```

**Weights out of range:**
```json
{
  "portfolio_state": {
    "current_weights": [1.5, -0.2, 0.3],
    "total_value": 100000.0
  },
  ...
}
```

**Invalid timestamp:**
```json
{
  "timestamp": "not-a-date",
  "realized_return": 0.025,
  "action_taken": [0.2, 0.3, 0.5]
}
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `models/sac_portfolio_model.zip` | Path to trained SAC model |
| `FEEDBACK_LOG_PATH` | `logs/feedback_log.jsonl` | Path to feedback log file |

---

## Shared Constants (from `orchestration.reward_shaping`)

The following constants are shared between the environment and API:

```python
RISK_FREE_RATE = 0.02           # Annual risk-free rate
TRANSACTION_COST_RATE = 0.001   # 0.1% per trade
MAX_POSITION_WEIGHT = 0.4       # Max 40% in single asset
MIN_POSITION_WEIGHT = 0.0       # No short selling
SHARPE_RATIO_TARGET = 1.5       # Target Sharpe ratio
TURNOVER_PENALTY_FACTOR = 0.5   # Turnover penalty multiplier
```

Import in your code:
```python
from orchestration.reward_shaping import (
    RISK_FREE_RATE,
    TRANSACTION_COST_RATE,
    MAX_POSITION_WEIGHT,
    calculate_reward,
    normalize_weights,
)
```

---

## Next Steps

1. **Start the server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

2. **Test endpoints** using curl commands above or visit http://localhost:8000/docs

3. **Train a model** and place it at `models/sac_portfolio_model.zip` to enable ML predictions

4. **Integrate with frontend** using the CORS-enabled endpoints

5. **Monitor feedback logs** at `logs/feedback_log.jsonl`
