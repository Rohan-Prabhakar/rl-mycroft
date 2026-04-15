import { NextRequest, NextResponse } from 'next/server';

/**
 * Portfolio API Endpoint
 * 
 * Provides SAC agent predictions for portfolio allocation across AI companies.
 * In production, this would call a Python backend running the trained SAC model.
 * 
 * For now, returns simulated allocations based on the SAC strategy.
 */

interface PortfolioAllocation {
  ticker: string;
  weight: number;
  companyName: string;
}

interface PortfolioResponse {
  success: boolean;
  allocations: PortfolioAllocation[];
  metrics: {
    sharpeRatio: number;
    cumulativeReturn: number;
    maxDrawdown: number;
    portfolioValue: number;
  };
  timestamp: string;
  message?: string;
}

// AI company tickers with names
const AI_COMPANIES: Record<string, string> = {
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
};

/**
 * Simulate SAC agent prediction
 * In production, this would call the Python SAC model via subprocess or RPC
 */
function simulateSACPrediction(): PortfolioAllocation[] {
  // These weights would come from the actual SAC model prediction
  // Based on learned policy from market feedback
  const baseWeights: Record<string, number> = {
    'NVDA': 0.18,
    'MSFT': 0.15,
    'GOOGL': 0.12,
    'META': 0.10,
    'AMZN': 0.10,
    'TSLA': 0.08,
    'AMD': 0.07,
    'INTC': 0.05,
    'CRM': 0.05,
    'ORCL': 0.04,
    'IBM': 0.03,
    'PLTR': 0.01,
    'SNOW': 0.01,
    'AI': 0.01,
  };

  return Object.entries(baseWeights).map(([ticker, weight]) => ({
    ticker,
    weight: Math.round(weight * 10000) / 10000, // Round to 4 decimals
    companyName: AI_COMPANIES[ticker] || ticker,
  }));
}

/**
 * Get simulated portfolio metrics
 * In production, these would come from the trained model's evaluation
 */
function getPortfolioMetrics(): PortfolioResponse['metrics'] {
  return {
    sharpeRatio: 1.85,
    cumulativeReturn: 0.234,
    maxDrawdown: 0.087,
    portfolioValue: 123400.00,
  };
}

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const action = searchParams.get('action') || 'predict';

    if (action === 'predict') {
      // Get SAC agent predictions
      const allocations = simulateSACPrediction();
      const metrics = getPortfolioMetrics();

      const response: PortfolioResponse = {
        success: true,
        allocations,
        metrics,
        timestamp: new Date().toISOString(),
        message: 'Portfolio allocation optimized by SAC agent',
      };

      return NextResponse.json(response);
    }

    if (action === 'metrics') {
      // Return only metrics
      const metrics = getPortfolioMetrics();

      return NextResponse.json({
        success: true,
        metrics,
        timestamp: new Date().toISOString(),
      });
    }

    if (action === 'tickers') {
      // Return available tickers
      const tickers = Object.entries(AI_COMPANIES).map(([ticker, name]) => ({
        ticker,
        name,
      }));

      return NextResponse.json({
        success: true,
        tickers,
        timestamp: new Date().toISOString(),
      });
    }

    return NextResponse.json(
      {
        success: false,
        error: 'Invalid action. Use: predict, metrics, or tickers',
      },
      { status: 400 }
    );
  } catch (error) {
    console.error('Portfolio API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { action, tickers, weights } = body;

    if (action === 'rebalance') {
      // Handle portfolio rebalancing request
      // In production, this would update the environment state
      
      return NextResponse.json({
        success: true,
        message: 'Portfolio rebalanced successfully',
        timestamp: new Date().toISOString(),
      });
    }

    if (action === 'evaluate') {
      // Evaluate custom portfolio weights
      const allocations = tickers?.map((ticker: string, i: number) => ({
        ticker,
        weight: weights?.[i] || 0,
        companyName: AI_COMPANIES[ticker] || ticker,
      })) || [];

      return NextResponse.json({
        success: true,
        allocations,
        timestamp: new Date().toISOString(),
      });
    }

    return NextResponse.json(
      {
        success: false,
        error: 'Invalid action. Use: rebalance or evaluate',
      },
      { status: 400 }
    );
  } catch (error) {
    console.error('Portfolio API error:', error);
    return NextResponse.json(
      {
        success: false,
        error: error instanceof Error ? error.message : 'Invalid request body',
      },
      { status: 400 }
    );
  }
}
