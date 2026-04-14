'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface AllocationChartProps {
  weights: number[];
  tickers: string[];
  isFallback?: boolean;
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#ec4899', '#6366f1', '#14b8a6'];

export default function AllocationChart({ weights, tickers, isFallback = false }: AllocationChartProps) {
  const data = weights.map((weight, index) => ({
    name: tickers[index] || `Asset ${index + 1}`,
    value: (weight * 100).toFixed(1),
    color: COLORS[index % COLORS.length],
  }));

  return (
    <div className="h-[300px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis 
            dataKey="name" 
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
          />
          <YAxis 
            label={{ value: 'Allocation (%)', angle: -90, position: 'insideLeft', fill: 'hsl(var(--muted-foreground))' }}
            stroke="hsl(var(--muted-foreground))"
            fontSize={12}
            tickLine={false}
            axisLine={false}
            domain={[0, 100]}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'hsl(var(--card))', 
              borderRadius: '8px', 
              border: '1px solid hsl(var(--border))',
              boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)'
            }}
            formatter={(value: string) => [`${value}%`, 'Weight']}
          />
          <Bar dataKey="value" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      {isFallback && (
        <p className="text-xs text-center text-muted-foreground mt-2">
          Fallback: Equal Weight Distribution
        </p>
      )}
    </div>
  );
}
