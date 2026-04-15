'use client';

import { useState, useEffect } from 'react';

interface PortfolioAllocation {
  ticker: string;
  weight: number;
  companyName: string;
}

interface PortfolioMetrics {
  sharpeRatio: number;
  cumulativeReturn: number;
  maxDrawdown: number;
  portfolioValue: number;
}

interface PortfolioData {
  allocations: PortfolioAllocation[];
  metrics: PortfolioMetrics;
  timestamp: string;
}

export default function PortfolioPage() {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [usingSimulation, setUsingSimulation] = useState(false);

  useEffect(() => {
    fetchPortfolioData();
  }, []);

  async function fetchPortfolioData() {
    try {
      setLoading(true);
      setError(null);
      
      // Try Python backend first (if running)
      let response;
      try {
        // Attempt to connect to Python backend on port 8000
        response = await fetch('http://localhost:8000/api/portfolio?action=predict');
      } catch (backendError) {
        // Fallback to Next.js API route if backend is not running
        console.log('Python backend not available, using simulated data');
        setUsingSimulation(true);
        response = await fetch('/api/portfolio?action=predict');
      }
      
      if (!response.ok) {
        throw new Error('Failed to fetch portfolio data');
      }
      const data = await response.json();
      if (data.success) {
        setPortfolioData({
          allocations: data.allocations,
          metrics: data.metrics,
          timestamp: data.timestamp,
        });
      } else {
        setError(data.error || 'Unknown error');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-lg text-gray-600 dark:text-gray-300">Loading SAC Agent Predictions...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 flex items-center justify-center">
        <div className="text-center bg-white dark:bg-gray-800 p-8 rounded-lg shadow-lg">
          <h2 className="text-2xl font-bold text-red-600 mb-4">Error</h2>
          <p className="text-gray-600 dark:text-gray-300">{error}</p>
          <button
            onClick={fetchPortfolioData}
            className="mt-4 px-6 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800 py-12 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 dark:text-white mb-4">
            Mycroft Portfolio Agents
          </h1>
          <p className="text-xl text-gray-600 dark:text-gray-300">
            AI-Powered Portfolio Optimization using Soft Actor-Critic (SAC)
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-2">
            Last updated: {portfolioData?.timestamp ? new Date(portfolioData.timestamp).toLocaleString() : 'N/A'}
          </p>
          {usingSimulation && (
            <div className="mt-4 inline-flex items-center px-4 py-2 bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-300 rounded-full text-sm">
              <span className="mr-2">⚠️</span>
              Running in simulation mode - Start Python backend for live predictions
            </div>
          )}
        </div>

        {/* Metrics Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-12">
          <MetricCard
            title="Sharpe Ratio"
            value={portfolioData?.metrics.sharpeRatio.toFixed(2) || '0.00'}
            description="Risk-adjusted return"
            icon="📈"
          />
          <MetricCard
            title="Cumulative Return"
            value={`${((portfolioData?.metrics.cumulativeReturn || 0) * 100).toFixed(1)}%`}
            description="Total portfolio growth"
            icon="💰"
          />
          <MetricCard
            title="Max Drawdown"
            value={`${((portfolioData?.metrics.maxDrawdown || 0) * 100).toFixed(1)}%`}
            description="Largest peak-to-trough decline"
            icon="📉"
          />
          <MetricCard
            title="Portfolio Value"
            value={`$${(portfolioData?.metrics.portfolioValue || 0).toLocaleString()}`}
            description="Current total value"
            icon="💵"
          />
        </div>

        {/* Allocation Chart */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-6">
            Portfolio Allocation
          </h2>
          <div className="space-y-4">
            {portfolioData?.allocations.map((allocation, index) => (
              <AllocationBar
                key={allocation.ticker}
                allocation={allocation}
                rank={index + 1}
              />
            ))}
          </div>
        </div>

        {/* Info Section */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-6">
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
            About the SAC Agent
          </h2>
          <div className="prose dark:prose-invert max-w-none">
            <p className="text-gray-600 dark:text-gray-300">
              This portfolio is optimized using a <strong>Soft Actor-Critic (SAC)</strong> reinforcement learning agent.
              The SAC algorithm learns optimal portfolio allocation strategies by interacting with market data
              and receiving feedback based on risk-adjusted returns.
            </p>
            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">Key Features:</h3>
                <ul className="list-disc list-inside text-gray-600 dark:text-gray-300 space-y-1">
                  <li>Adapts to changing market conditions</li>
                  <li>Optimizes for risk-adjusted returns</li>
                  <li>Considers transaction costs</li>
                  <li>Implements drawdown protection</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white mb-2">AI Companies:</h3>
                <p className="text-gray-600 dark:text-gray-300 text-sm">
                  The portfolio focuses on leading AI companies including NVIDIA, Microsoft, Google,
                  Meta, Amazon, and other major players in artificial intelligence.
                </p>
              </div>
            </div>
            
            {/* Setup Instructions */}
            <div className="mt-6 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <h3 className="font-semibold text-blue-900 dark:text-blue-100 mb-2">🚀 Getting Started:</h3>
              <ol className="list-decimal list-inside text-sm text-blue-800 dark:text-blue-200 space-y-1">
                <li>Download S&P 500 data from <a href="https://www.kaggle.com/datasets/andrewmvd/sp-500-stocks" className="underline hover:text-blue-600" target="_blank" rel="noopener noreferrer">Kaggle</a></li>
                <li>Create pickle: <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">python rl_portfolio/envs/sp500_data_loader.py --mode convert</code></li>
                <li>Train model: <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">python rl_portfolio/training/train.py --debug</code></li>
                <li>Start backend: <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">python rl_portfolio/serve_model.py --port 8000</code></li>
                <li>Refresh this page to see live predictions!</li>
              </ol>
              <p className="mt-3 text-xs text-blue-700 dark:text-blue-300">
                📖 Full documentation: <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">rl_portfolio/INTEGRATION.md</code>
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

interface MetricCardProps {
  title: string;
  value: string;
  description: string;
  icon: string;
}

function MetricCard({ title, value, description, icon }: MetricCardProps) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow">
      <div className="flex items-center justify-between mb-4">
        <span className="text-3xl">{icon}</span>
        <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400">{title}</h3>
      </div>
      <p className="text-2xl font-bold text-gray-900 dark:text-white mb-1">{value}</p>
      <p className="text-xs text-gray-500 dark:text-gray-400">{description}</p>
    </div>
  );
}

interface AllocationBarProps {
  allocation: PortfolioAllocation;
  rank: number;
}

function AllocationBar({ allocation, rank }: AllocationBarProps) {
  const percentage = (allocation.weight * 100).toFixed(1);
  
  // Color gradient based on weight
  const getColor = (weight: number) => {
    if (weight >= 0.15) return 'bg-blue-600';
    if (weight >= 0.10) return 'bg-blue-500';
    if (weight >= 0.05) return 'bg-blue-400';
    return 'bg-blue-300';
  };

  return (
    <div className="flex items-center space-x-4">
      <div className="w-8 text-right text-sm font-medium text-gray-500 dark:text-gray-400">
        #{rank}
      </div>
      <div className="w-16 text-sm font-bold text-gray-900 dark:text-white">
        {allocation.ticker}
      </div>
      <div className="flex-1">
        <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full ${getColor(allocation.weight)} transition-all duration-500`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      </div>
      <div className="w-20 text-right text-sm font-medium text-gray-900 dark:text-white">
        {percentage}%
      </div>
      <div className="w-48 text-sm text-gray-600 dark:text-gray-300 truncate hidden lg:block">
        {allocation.companyName}
      </div>
    </div>
  );
}
