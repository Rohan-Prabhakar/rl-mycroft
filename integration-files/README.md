# RL Portfolio Integration for Mycroft

This directory contains the files needed to integrate the RL Portfolio optimization layer into the Mycroft Next.js frontend.

## Files to Copy

Copy these files/directories to your Mycroft repository (https://github.com/nikbearbrown/Mycroft):

```
lib/mycroft-rl.ts                      → lib/mycroft-rl.ts
app/rl-portfolio/page.tsx              → app/rl-portfolio/page.tsx
app/rl-portfolio/components/           → app/rl-portfolio/components/
```

## Setup Steps

### 1. Copy Integration Files

From this `integration-files` directory, copy:

```bash
# In your Mycroft repo root
cp /path/to/integration-files/lib/mycroft-rl.ts lib/
cp -r /path/to/integration-files/app/rl-portfolio app/
```

### 2. Update Environment Variables

Add to your `.env.local` (or create from `.env.local.example`):

```bash
NEXT_PUBLIC_RL_API_URL=http://localhost:8000
```

For production deployment, update to your backend URL:
```bash
NEXT_PUBLIC_RL_API_URL=https://your-backend-api.com
```

### 3. Verify Dependencies

The integration uses existing dependencies in Mycroft:
- ✅ `recharts` (already in package.json)
- ✅ `lucide-react` (already in package.json)

No additional npm installs needed!

### 4. Run the Backend

Start the FastAPI backend before using the frontend:

```bash
# In the RL backend repo
pip install -r requirements.txt
python experiments/fetch_real_data.py --output data/market_data.csv
python training/trainer.py --data-path data/market_data.csv --model-path models/sac_portfolio.zip
MODEL_PATH=models/sac_portfolio.zip uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Run the Frontend

```bash
# In Mycroft repo
npm install
npm run dev
```

Visit: http://localhost:3000/rl-portfolio

## Features

- **3 Market Scenarios**: Bull, Bear, and Volatile market simulations
- **Real-time Allocation**: RL model generates optimal portfolio weights
- **Risk Metrics**: HHI, Effective Assets, Max Weight, Confidence scores
- **Responsive Design**: Matches Mycroft's existing theme (light/dark mode)
- **Fallback Mode**: Shows equal-weight baseline if backend is unavailable

## Architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Mycroft UI    │─────▶│  FastAPI Backend │─────▶│  SAC Model      │
│  (Next.js 15)   │◀─────│   (Port 8000)    │◀─────│  (Loaded)       │
└─────────────────┘      └──────────────────┘      └─────────────────┘
       │                         │
       ▼                         ▼
  Recharts Charts          Pydantic Validation
  Lucide Icons             CORS Middleware
  Tailwind CSS             Error Handling
```

## API Endpoints Used

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Generate portfolio allocation |
| `/health` | GET | Check backend status |

## Example Request

```typescript
const marketState = {
  ticker_returns: [0.02, -0.01, 0.015],
  volatilities: [0.15, 0.20, 0.18],
  volume_zscores: [1.2, -0.5, 0.8],
  rsi_values: [55, 42, 68],
  macd_values: [0.02, -0.01, 0.015],
  current_weights: [0.33, 0.33, 0.34],
  cash_ratio: 0.05,
  portfolio_value: 100000,
  peak_value: 100000
};

const response = await fetchRLPrediction(marketState);
// Returns: { weights, tickers, confidence, risk_metrics, metadata }
```

## Troubleshooting

### "Connection Error"
- Ensure backend is running: `curl http://localhost:8000/health`
- Check `NEXT_PUBLIC_RL_API_URL` matches backend address
- Verify CORS settings in `api/main.py` include your frontend URL

### "Model not loaded"
- Train model first: `python training/trainer.py ...`
- Set `MODEL_PATH` env var when starting uvicorn
- Check model file exists at specified path

## License

Same as parent Mycroft repository.

---

**Disclaimer**: This is for educational/portfolio demonstration purposes only. Not financial advice.
