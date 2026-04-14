# Portfolio Optimization with Deep Reinforcement Learning

A production-ready portfolio optimization system using Soft Actor-Critic (SAC) deep reinforcement learning. This project implements a complete ML pipeline from environment simulation to API deployment.

## ⚠️ Disclaimer

**For educational and research purposes only. Not financial advice.**

This software is provided for educational/portfolio management research purposes only. Do not use for actual financial trading or investment decisions without proper validation, backtesting, and professional financial advice. Past performance does not guarantee future results.

---

## 🏗️ Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A[Market Data] --> B[Portfolio Environment]
        C[Historical Prices] --> B
    end
    
    subgraph "RL Core"
        B --> D[SAC Agent]
        D --> E[Policy Network]
        D --> F[Value Networks]
        E --> G[Action: Allocation Weights]
    end
    
    subgraph "Reward Shaping"
        H[Sharpe Ratio] --> I[Composite Reward]
        J[Drawdown Penalty] --> I
        K[Transaction Costs] --> I
        L[Turnover Penalty] --> I
    end
    
    subgraph "API Layer"
        M[FastAPI Server] --> N[/predict Endpoint]
        M --> O[/feedback Endpoint]
        M --> P[/health Endpoint]
        N --> D
    end
    
    subgraph "Visualization"
        Q[Learning Curves] 
        R[Allocation Heatmap]
        S[Cumulative Returns]
    end
    
    I --> B
    G --> M
    B --> S
    D --> Q
```

### Directory Structure

```
portfolio-optimization/
├── agents/              # SAC agent and environment implementations
│   ├── __init__.py
│   └── sac_agent.py     # PortfolioEnv and SacAgent classes
├── api/                 # FastAPI inference endpoints
│   ├── __init__.py
│   └── main.py          # /predict, /feedback, /health endpoints
├── orchestration/       # Shared utilities and reward shaping
│   ├── __init__.py
│   └── reward_shaping.py
├── training/            # Training scripts and configuration
│   ├── __init__.py
│   ├── config.py        # Hyperparameters and settings
│   └── trainer.py       # Main training loop
├── tests/               # Pytest test suite
│   ├── test_env.py
│   ├── test_reward.py
│   └── test_api.py
├── experiments/         # Results, plots, and visualization
│   ├── visualize_results.py
│   ├── results/         # Training logs
│   └── plots/           # Generated visualizations
├── docker/              # Docker deployment files
│   ├── Dockerfile
│   └── docker-compose.yml
├── models/              # Saved model checkpoints (gitignored)
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip or conda
- Docker (optional, for containerized deployment)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd portfolio-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training the Model

```bash
# Train with default settings (10,000 steps)
python training/trainer.py --num-steps 10000

# Train with custom configuration
python training/trainer.py \
    --num-steps 50000 \
    --initial-capital 50000 \
    --transaction-cost 0.001 \
    --seed 42 \
    --log-interval 100

# Monitor training progress
tail -f experiments/results/training_log.json
```

### Running the API Server

```bash
# Development mode with auto-reload
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Generating Visualizations

```bash
# Generate all plots (uses synthetic data if no training results)
python experiments/visualize_results.py --output-dir experiments/plots

# Generate plots from training results
python experiments/visualize_results.py \
    --results-dir experiments/results \
    --output-dir experiments/plots
```

---

## 📡 API Usage

### Interactive Documentation

Once the API server is running, access interactive Swagger docs at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Endpoints

#### 1. `/predict` (POST)

Get portfolio allocation predictions.

**Request:**
```json
{
  "market_data": [
    {"price": 100.0, "return_1d": 0.01, "volatility": 0.02, "momentum": 0.05},
    {"price": 50.0, "return_1d": -0.02, "volatility": 0.03, "momentum": -0.01},
    {"price": 75.0, "return_1d": 0.005, "volatility": 0.015, "momentum": 0.02}
  ],
  "portfolio_state": {
    "current_weights": [0.4, 0.3, 0.3],
    "cash_ratio": 0.1,
    "total_value": 10000.0
  }
}
```

**Response:**
```json
{
  "allocation_weights": [0.35, 0.33, 0.32],
  "risk_metrics": {
    "sharpe_ratio": 1.25,
    "volatility": 0.015,
    "expected_return": 0.008
  },
  "model_confidence": 0.92
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "market_data": [
      {"price": 100.0, "return_1d": 0.01, "volatility": 0.02, "momentum": 0.05},
      {"price": 50.0, "return_1d": -0.02, "volatility": 0.03, "momentum": -0.01}
    ],
    "portfolio_state": {
      "current_weights": [0.5, 0.5],
      "cash_ratio": 0.1,
      "total_value": 10000.0
    }
  }'
```

#### 2. `/feedback` (POST)

Log realized returns for online learning.

**Request:**
```json
{
  "timestamp": "2024-01-15T10:30:00",
  "predicted_weights": [0.4, 0.3, 0.3],
  "realized_return": 0.025,
  "actual_weights": [0.38, 0.32, 0.30]
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": "2024-01-15T10:30:00",
    "predicted_weights": [0.4, 0.3, 0.3],
    "realized_return": 0.025,
    "actual_weights": [0.38, 0.32, 0.30]
  }'
```

#### 3. `/health` (GET)

Check system health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600,
  "environment_info": {
    "num_assets": 3,
    "initial_capital": 10000
  }
}
```

**cURL Example:**
```bash
curl "http://localhost:8000/health"
```

---

## 🧪 Testing

Run the complete test suite:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_env.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html

# Run API tests only
python -m pytest tests/test_api.py -v
```

### Test Coverage

- **Environment Tests** (`tests/test_env.py`): Shape validation, reward bounds, action constraints
- **Reward Tests** (`tests/test_reward.py`): Sharpe ratio, drawdown penalty, transaction costs
- **API Tests** (`tests/test_api.py`): Endpoint validation, error handling, CORS

---

## 🐳 Docker Deployment

### Build and Run with Docker Compose

```bash
# Navigate to docker directory
cd docker

# Build and start services
docker compose up --build

# Run in detached mode
docker compose up -d --build

# View logs
docker compose logs -f

# Stop services
docker compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t portfolio-optimization:latest -f docker/Dockerfile .

# Run container
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/experiments:/app/experiments \
  -v $(pwd)/models:/app/models \
  --name portfolio-api \
  portfolio-optimization:latest
```

### Access API in Docker

```bash
# Health check
curl http://localhost:8000/health

# Test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"market_data": [{"price": 100, "return_1d": 0.01, "volatility": 0.02, "momentum": 0.05}], "portfolio_state": {"current_weights": [1.0], "cash_ratio": 0.1, "total_value": 10000}}'
```

---

## 🎯 Configuration

### Training Parameters

Edit `training/config.py` or use CLI arguments:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `initial_capital` | 10000 | Starting portfolio value |
| `transaction_cost_rate` | 0.001 | Transaction cost (0.1%) |
| `risk_free_rate` | 0.02 | Annual risk-free rate |
| `num_assets` | 3 | Number of assets |
| `window_size` | 5 | Lookback window for features |
| `seed` | 42 | Random seed for reproducibility |

### Reward Shaping

Configure reward components in `orchestration/reward_shaping.py`:

```python
class RewardConfig:
    SHARPE_COEFF = 1.0          # Sharpe ratio weight
    DRAWDOWN_COEFF = 0.5        # Drawdown penalty weight
    TRANSACTION_COST_COEFF = 0.1 # Transaction cost weight
    TURNOVER_COEFF = 0.05       # Turnover penalty weight
    RISK_FREE_RATE = 0.02       # Annual risk-free rate
```

---

## 📊 Reproducibility

### Seed Control

All random operations use controlled seeds for reproducibility:

```bash
# Set seed via CLI
python training/trainer.py --seed 42

# Or in code
import numpy as np
np.random.seed(42)
```

### Environment Variables

```bash
export PYTHONHASHSEED=42
export CUDA_VISIBLE_DEVICES=0  # If using GPU
```

---

## ⚖️ Ethical & Risk Considerations

### Important Notes

1. **Not Financial Advice**: This system is for educational purposes only. Do not use for real trading without extensive validation.

2. **Market Risks**: 
   - RL models can overfit to historical data
   - Market regimes change; past performance ≠ future results
   - Black swan events are not captured in training data

3. **Model Limitations**:
   - Assumes frictionless markets (despite transaction cost modeling)
   - Does not account for liquidity constraints
   - No guarantee of positive returns

4. **Responsible Use**:
   - Always backtest thoroughly before any real deployment
   - Implement proper risk management controls
   - Monitor model drift in production
   - Maintain human oversight

5. **Regulatory Compliance**: Ensure compliance with local financial regulations before any production use.

---

## 📈 Expected Outputs

After training and visualization, you'll have:

```
experiments/
├── results/
│   └── training_log.json    # Episode-by-episode metrics
├── plots/
│   ├── learning_curves.png   # Reward, Sharpe, drawdown over time
│   ├── allocation_heatmap.png # Asset weights over time
│   └── cumulative_returns.png # SAC vs baseline comparison
└── models/
    └── sac_portfolio_model.zip  # Trained model checkpoint
```

### Sample Visualization Output

![Learning Curves](experiments/plots/learning_curves.png)
![Allocation Heatmap](experiments/plots/allocation_heatmap.png)
![Cumulative Returns](experiments/plots/cumulative_returns.png)

---

## 🔧 Troubleshooting

### Common Issues

**Model not found on API startup:**
```
FileNotFoundError: Model file not found
```
→ Train a model first or ensure `models/` directory contains `sac_portfolio_model.zip`

**Import errors:**
```
ModuleNotFoundError: No module named 'xxx'
```
→ Reinstall dependencies: `pip install -r requirements.txt`

**Docker build fails:**
```
ERROR: failed to solve: ...
```
→ Ensure you're in the correct directory and Docker daemon is running

**Tests fail:**
```
AssertionError: Reward outside expected bounds
```
→ Check random seed consistency; some variance is expected

---

## 📚 References

- [Soft Actor-Critic Paper](https://arxiv.org/abs/1801.01290)
- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

MIT License - See LICENSE file for details.

---

**Built with ❤️ for educational purposes**

*Remember: This is a learning tool, not a trading system. Always do your own research.*
