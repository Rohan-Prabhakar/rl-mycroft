'use client';

import { useState } from 'react';
import { fetchRLPrediction, getScenarioData, type MarketState, type AllocationResponse } from '@/lib/mycroft-rl';
import AllocationChart from './components/AllocationChart';
import RiskMetrics from './components/RiskMetrics';
import { Activity, AlertCircle, TrendingUp, ShieldAlert } from 'lucide-react';

export default function RLPortfolioPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AllocationResponse | null>(null);
  const [currentScenario, setCurrentScenario] = useState<'bull' | 'bear' | 'volatile' | null>(null);

  const handleScenarioClick = async (type: 'bull' | 'bear' | 'volatile') => {
    setLoading(true);
    setError(null);
    setResult(null);
    setCurrentScenario(type);

    try {
      const marketState = getScenarioData(type);
      const data = await fetchRLPrediction(marketState);
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getInsight = () => {
    if (!result || !currentScenario) return null;
    
    switch (currentScenario) {
      case 'bull':
        return "Bull Market Detected: The model favors higher allocation to high-momentum assets while maintaining diversification to capture upside potential.";
      case 'bear':
        return "Bear Market Detected: The model suggests defensive positioning, potentially increasing cash equivalents or lowering exposure to high-volatility assets.";
      case 'volatile':
        return "High Volatility Regime: The model is balancing risk by normalizing weights and avoiding extreme concentrations in unstable assets.";
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-background py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-foreground mb-4">
            RL Portfolio Optimizer
          </h1>
          <p className="text-lg text-muted-foreground">
            Simulate Reinforcement Learning asset allocation across different market regimes.
          </p>
        </div>

        {/* Scenario Buttons */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
          <button
            onClick={() => handleScenarioClick('bull')}
            disabled={loading}
            className="flex flex-col items-center justify-center p-6 bg-card rounded-xl shadow-md hover:shadow-lg transition-all border border-green-200 dark:border-green-900 group"
          >
            <TrendingUp className="w-12 h-12 text-green-500 mb-3 group-hover:scale-110 transition-transform" />
            <span className="font-semibold text-foreground">Bull Market</span>
            <span className="text-sm text-muted-foreground mt-1">High Momentum</span>
          </button>

          <button
            onClick={() => handleScenarioClick('bear')}
            disabled={loading}
            className="flex flex-col items-center justify-center p-6 bg-card rounded-xl shadow-md hover:shadow-lg transition-all border border-red-200 dark:border-red-900 group"
          >
            <ShieldAlert className="w-12 h-12 text-red-500 mb-3 group-hover:scale-110 transition-transform" />
            <span className="font-semibold text-foreground">Bear Market</span>
            <span className="text-sm text-muted-foreground mt-1">Downtrend</span>
          </button>

          <button
            onClick={() => handleScenarioClick('volatile')}
            disabled={loading}
            className="flex flex-col items-center justify-center p-6 bg-card rounded-xl shadow-md hover:shadow-lg transition-all border border-amber-200 dark:border-amber-900 group"
          >
            <Activity className="w-12 h-12 text-amber-500 mb-3 group-hover:scale-110 transition-transform" />
            <span className="font-semibold text-foreground">Volatile</span>
            <span className="text-sm text-muted-foreground mt-1">Chaotic</span>
          </button>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-primary"></div>
            <p className="mt-4 text-muted-foreground">Analyzing market state...</p>
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-destructive/10 border-l-4 border-destructive p-6 rounded-md mb-8">
            <div className="flex items-start">
              <AlertCircle className="w-6 h-6 text-destructive mr-3 mt-0.5" />
              <div>
                <h3 className="text-lg font-medium text-destructive">Connection Error</h3>
                <p className="mt-2 text-muted-foreground">{error}</p>
                <p className="mt-2 text-sm text-muted-foreground">
                  Ensure the Python backend is running at <code>NEXT_PUBLIC_RL_API_URL</code>.
                  Showing fallback equal-weight analysis.
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Results Display */}
        {(result || error) && (
          <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-700">
            
            {/* Insight Box */}
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-6">
              <h3 className="text-lg font-semibold text-blue-900 dark:text-blue-100 mb-2">
                AI Insight
              </h3>
              <p className="text-blue-800 dark:text-blue-200">
                {getInsight() || "Select a scenario to generate an allocation strategy."}
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Chart Section */}
              <div className="bg-card rounded-xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-foreground mb-6">
                  Recommended Allocation
                </h2>
                {result ? (
                  <AllocationChart 
                    weights={result.weights} 
                    tickers={result.tickers} 
                  />
                ) : (
                  <AllocationChart 
                    weights={[0.2, 0.2, 0.2, 0.2, 0.2]} 
                    tickers={['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']} 
                    isFallback={true}
                  />
                )}
              </div>

              {/* Metrics Section */}
              <div className="space-y-6">
                {result ? (
                  <RiskMetrics metrics={result.risk_metrics} confidence={result.confidence} />
                ) : (
                  <RiskMetrics 
                    metrics={{
                      max_weight: 0.2,
                      concentration_hhi: 0.2,
                      effective_n_assets: 5,
                      volatility_weighted: 0.15
                    }} 
                    confidence={0.5}
                    isFallback={true}
                  />
                )}
                
                {result?.metadata?.fallback && (
                  <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-4">
                    <p className="text-sm text-amber-800 dark:text-amber-200">
                      <strong>Note:</strong> Model unavailable. Displaying equal-weight baseline strategy.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
