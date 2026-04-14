'use client';

import { Gauge, PieChart, Layers, Activity } from 'lucide-react';

interface RiskMetricsProps {
  metrics: {
    max_weight: number;
    concentration_hhi: number;
    effective_n_assets: number;
    volatility_weighted: number;
  };
  confidence: number;
  isFallback?: boolean;
}

const MetricCard = ({ 
  icon: Icon, 
  title, 
  value, 
  subtext, 
  colorClass 
}: { 
  icon: any, 
  title: string, 
  value: string, 
  subtext?: string,
  colorClass: string 
}) => (
  <div className="bg-muted/50 rounded-lg p-4 border border-border">
    <div className="flex items-center justify-between mb-2">
      <span className="text-sm font-medium text-muted-foreground">{title}</span>
      <Icon className={`w-5 h-5 ${colorClass}`} />
    </div>
    <div className="text-2xl font-bold text-foreground">{value}</div>
    {subtext && <div className="text-xs text-muted-foreground mt-1">{subtext}</div>}
  </div>
);

export default function RiskMetrics({ metrics, confidence, isFallback = false }: RiskMetricsProps) {
  return (
    <div className="grid grid-cols-2 gap-4">
      <MetricCard 
        icon={Gauge}
        title="Confidence"
        value={`${(confidence * 100).toFixed(0)}%`}
        subtext={isFallback ? "Baseline Strategy" : "Model Certainty"}
        colorClass={isFallback ? "text-muted-foreground" : "text-primary"}
      />
      
      <MetricCard 
        icon={PieChart}
        title="Max Weight"
        value={`${(metrics.max_weight * 100).toFixed(1)}%`}
        subtext="Highest Single Asset"
        colorClass="text-green-500"
      />
      
      <MetricCard 
        icon={Layers}
        title="HHI Index"
        value={metrics.concentration_hhi.toFixed(3)}
        subtext="< 0.15 = Diversified"
        colorClass="text-purple-500"
      />
      
      <MetricCard 
        icon={Activity}
        title="Eff. Assets"
        value={metrics.effective_n_assets.toFixed(1)}
        subtext="True Diversification Count"
        colorClass="text-amber-500"
      />
    </div>
  );
}
