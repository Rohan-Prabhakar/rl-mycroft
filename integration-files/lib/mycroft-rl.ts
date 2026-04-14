// lib/mycroft-rl.ts
// RL Portfolio Integration Layer for Mycroft

export interface MarketState {
  ticker_returns: number[];
  volatilities: number[];
  volume_zscores: number[];
  rsi_values: number[];
  macd_values: number[];
  current_weights: number[];
  cash_ratio: number;
  portfolio_value: number;
  peak_value: number;
}

export interface RiskMetrics {
  max_weight: number;
  concentration_hhi: number;
  effective_n_assets: number;
  volatility_weighted: number;
}

export interface AllocationResponse {
  weights: number[];
  tickers: string[];
  confidence: number;
  risk_metrics: RiskMetrics;
  metadata?: Record<string, any>;
}

const API_URL = process.env.NEXT_PUBLIC_RL_API_URL || 'http://localhost:8000';

export async function fetchRLPrediction(state: MarketState): Promise<AllocationResponse> {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(state),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `API request failed with status ${response.status}`);
    }

    return await response.json();
  } catch (error) {
    console.error('Failed to fetch RL prediction:', error);
    throw error;
  }
}

// Predefined market scenarios for demo
export const getScenarioData = (type: 'bull' | 'bear' | 'volatile'): MarketState => {
  const baseTickers = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN'];
  const n = baseTickers.length;
  const equalWeight = 1 / n;
  const currentWeights = Array(n).fill(equalWeight);

  if (type === 'bull') {
    return {
      ticker_returns: [0.02, 0.015, 0.018, 0.025, 0.012],
      volatilities: [0.15, 0.18, 0.14, 0.25, 0.16],
      volume_zscores: [1.5, 1.2, 1.8, 2.0, 1.0],
      rsi_values: [65, 62, 68, 70, 58],
      macd_values: [0.03, 0.02, 0.025, 0.04, 0.015],
      current_weights: currentWeights,
      cash_ratio: 0.05,
      portfolio_value: 100000,
      peak_value: 100000,
    };
  }

  if (type === 'bear') {
    return {
      ticker_returns: [-0.03, -0.025, -0.02, -0.04, -0.015],
      volatilities: [0.25, 0.22, 0.20, 0.30, 0.18],
      volume_zscores: [2.5, 2.0, 1.8, 3.0, 1.5],
      rsi_values: [25, 28, 30, 20, 35],
      macd_values: [-0.04, -0.03, -0.025, -0.05, -0.02],
      current_weights: currentWeights,
      cash_ratio: 0.20,
      portfolio_value: 95000,
      peak_value: 100000,
    };
  }

  // Volatile
  return {
    ticker_returns: [0.05, -0.04, 0.03, -0.06, 0.02],
    volatilities: [0.35, 0.30, 0.28, 0.40, 0.25],
    volume_zscores: [3.0, -2.5, 2.8, -3.0, 1.5],
    rsi_values: [75, 25, 60, 15, 55],
    macd_values: [0.08, -0.06, 0.04, -0.09, 0.02],
    current_weights: currentWeights,
    cash_ratio: 0.10,
    portfolio_value: 98000,
    peak_value: 100000,
  };
};
