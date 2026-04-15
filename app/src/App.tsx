import { useState, useEffect } from 'react';
import axios from 'axios';
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts';
import { Activity, TrendingUp, AlertTriangle, Zap } from 'lucide-react';
import './App.css';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

interface Allocation {
  ticker: string;
  weight: number;
  companyName: string;
}

interface Metrics {
  sharpeRatio: number;
  cumulativeReturn: number;
  maxDrawdown: number;
  portfolioValue: number;
}

interface PortfolioData {
  success: boolean;
  allocations: Allocation[];
  metrics: Metrics;
  timestamp: string;
  message?: string;
}

const COLORS = [
  '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8',
  '#82CA9D', '#FFC658', '#FF6B6B', '#4ECDC4', '#45B7D1',
  '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F'
];

function App() {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isLiveMode, setIsLiveMode] = useState(false);

  useEffect(() => {
    fetchPortfolioData();
  }, []);

  const fetchPortfolioData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.get(`${API_BASE_URL}/api/portfolio`, {
        params: { action: 'predict' },
        timeout: 5000
      });
      
      setPortfolioData(response.data);
      setIsLiveMode(true);
    } catch (err: any) {
      console.error('Failed to fetch portfolio data:', err);
      
      if (err.code === 'ERR_NETWORK' || err.message.includes('Network Error')) {
        setError('Backend server is not running. Starting in simulation mode.');
        setPortfolioData({
          success: true,
          allocations: [
            { ticker: 'NVDA', weight: 0.18, companyName: 'NVIDIA Corporation' },
            { ticker: 'MSFT', weight: 0.15, companyName: 'Microsoft Corporation' },
            { ticker: 'GOOGL', weight: 0.12, companyName: 'Alphabet Inc.' },
            { ticker: 'META', weight: 0.10, companyName: 'Meta Platforms Inc.' },
            { ticker: 'AMZN', weight: 0.10, companyName: 'Amazon.com Inc.' },
            { ticker: 'TSLA', weight: 0.08, companyName: 'Tesla Inc.' },
            { ticker: 'AMD', weight: 0.07, companyName: 'Advanced Micro Devices' },
            { ticker: 'INTC', weight: 0.05, companyName: 'Intel Corporation' },
            { ticker: 'CRM', weight: 0.05, companyName: 'Salesforce Inc.' },
          ],
          metrics: {
            sharpeRatio: 1.85,
            cumulativeReturn: 0.234,
            maxDrawdown: 0.087,
            portfolioValue: 123400.00
          },
          timestamp: new Date().toISOString(),
          message: 'Simulated data (backend not connected)'
        });
        setIsLiveMode(false);
      } else {
        setError(`Failed to load data: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;
  const formatCurrency = (value: number) => `$${value.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <div className="app-container">
      <header className="header">
        <div className="header-content">
          <div className="logo-section">
            <Activity className="logo-icon" />
            <h1>Mycroft RL Portfolio</h1>
          </div>
          <div className={`mode-indicator ${isLiveMode ? 'live' : 'simulation'}`}>
            <Zap size={16} />
            {isLiveMode ? 'Live Mode' : 'Simulation Mode'}
          </div>
        </div>
        <p className="subtitle">AI-Powered Portfolio Optimization with Reinforcement Learning</p>
      </header>

      <main className="main-content">
        {loading ? (
          <div className="loading-state">
            <div className="spinner"></div>
            <p>Loading portfolio data...</p>
          </div>
        ) : error && !portfolioData ? (
          <div className="error-state">
            <AlertTriangle className="error-icon" />
            <h2>Error Loading Data</h2>
            <p>{error}</p>
            <button onClick={fetchPortfolioData} className="retry-button">Retry</button>
          </div>
        ) : portfolioData ? (
          <>
            <section className="metrics-section">
              <h2>Portfolio Performance</h2>
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-icon positive"><TrendingUp size={24} /></div>
                  <div className="metric-info">
                    <span className="metric-label">Sharpe Ratio</span>
                    <span className="metric-value">{portfolioData.metrics.sharpeRatio.toFixed(2)}</span>
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-icon positive"><TrendingUp size={24} /></div>
                  <div className="metric-info">
                    <span className="metric-label">Cumulative Return</span>
                    <span className="metric-value">{formatPercentage(portfolioData.metrics.cumulativeReturn)}</span>
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-icon negative"><AlertTriangle size={24} /></div>
                  <div className="metric-info">
                    <span className="metric-label">Max Drawdown</span>
                    <span className="metric-value">{formatPercentage(portfolioData.metrics.maxDrawdown)}</span>
                  </div>
                </div>
                <div className="metric-card highlight">
                  <div className="metric-icon"><TrendingUp size={24} /></div>
                  <div className="metric-info">
                    <span className="metric-label">Portfolio Value</span>
                    <span className="metric-value">{formatCurrency(portfolioData.metrics.portfolioValue)}</span>
                  </div>
                </div>
              </div>
            </section>

            <section className="allocation-section">
              <h2>Portfolio Allocation</h2>
              <div className="allocation-content">
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={400}>
                    <PieChart>
                      <Pie
                        data={portfolioData.allocations}
                        cx="50%"
                        cy="50%"
                        labelLine={false}
                        label={({ ticker, percent }) => `${ticker}: ${(percent * 100).toFixed(0)}%`}
                        outerRadius={120}
                        fill="#8884d8"
                        dataKey="weight"
                      >
                        {portfolioData.allocations.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value: number) => formatPercentage(value)}
                        contentStyle={{ backgroundColor: '#1a1a2e', border: '1px solid #333', borderRadius: '8px' }}
                      />
                      <Legend verticalAlign="bottom" height={36}
                        formatter={(value) => <span style={{ color: '#e0e0e0' }}>{value}</span>}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </div>

                <div className="allocation-list">
                  <h3>Top Holdings</h3>
                  {portfolioData.allocations.slice(0, 10).map((allocation, index) => (
                    <div key={allocation.ticker} className="allocation-item">
                      <div className="allocation-rank">{index + 1}</div>
                      <div className="allocation-details">
                        <div className="allocation-ticker">{allocation.ticker}</div>
                        <div className="allocation-company">{allocation.companyName}</div>
                      </div>
                      <div className="allocation-weight">{formatPercentage(allocation.weight)}</div>
                      <div className="allocation-bar" style={{ width: `${allocation.weight * 100}%`, backgroundColor: COLORS[index % COLORS.length] }} />
                    </div>
                  ))}
                </div>
              </div>
            </section>

            {portfolioData.message && (
              <section className="info-section">
                <div className="info-box">
                  <Zap size={20} />
                  <span>{portfolioData.message}</span>
                </div>
              </section>
            )}

            <section className="last-updated">
              Last updated: {new Date(portfolioData.timestamp).toLocaleString()}
            </section>
          </>
        ) : null}
      </main>

      <footer className="footer">
        <p>Powered by SAC (Soft Actor-Critic) Reinforcement Learning Agent</p>
        <p>Data refreshed every 60 seconds when backend is connected</p>
      </footer>
    </div>
  );
}

export default App;
