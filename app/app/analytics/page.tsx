"use client";

import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {
  analyticsKpis,
  predictionsOverTime,
  popularityDistribution,
  alignmentBreakdown,
  type KpiMetric,
} from "@/lib/mock-data";

const TREND_ICON: Record<KpiMetric["trend"], React.ElementType> = {
  up: TrendingUp,
  down: TrendingDown,
  flat: Minus,
};

function KpiCard({ label, value, delta, trend }: KpiMetric) {
  const TrendIcon = TREND_ICON[trend];
  const trendColor =
    trend === "up"
      ? "text-accent"
      : trend === "down"
        ? "text-destructive"
        : "text-muted-foreground";

  return (
    <Card>
      <CardContent className="pt-1">
        <p className="text-xs text-muted-foreground">{label}</p>
        <div className="flex items-end justify-between mt-1.5">
          <span className="text-2xl font-semibold text-primary">{value}</span>
          <span className={`flex items-center gap-1 text-xs font-medium ${trendColor}`}>
            <TrendIcon size={13} />
            {delta}
          </span>
        </div>
      </CardContent>
    </Card>
  );
}

const CHART_GRID = "var(--color-border)";
const CHART_MUTED = "var(--color-muted-foreground)";
const CHART_ACCENT = "var(--color-accent)";

const tooltipStyle = {
  backgroundColor: "var(--color-card)",
  border: "1px solid var(--color-border)",
  borderRadius: "8px",
  fontSize: "12px",
  color: "var(--color-primary)",
};

export default function AnalyticsPage() {
  return (
    <main className="flex-1 px-6 py-8 bg-background space-y-6 max-w-5xl w-full mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-primary mb-1">Analytics</h1>
        <p className="text-sm text-muted-foreground">
          Usage trends, prediction distributions, and model diagnostics.
        </p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {analyticsKpis.map((kpi) => (
          <KpiCard key={kpi.label} {...kpi} />
        ))}
      </div>

      <div className="grid lg:grid-cols-2 gap-4">
        <Card>
          <CardHeader>
            <CardTitle>Predictions over time</CardTitle>
          </CardHeader>
          <CardContent className="h-64 pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={predictionsOverTime}>
                <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="month" stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} width={32} />
                <Tooltip contentStyle={tooltipStyle} />
                <Line
                  type="monotone"
                  dataKey="predictions"
                  stroke={CHART_ACCENT}
                  strokeWidth={2.5}
                  dot={{ r: 3, fill: CHART_ACCENT }}
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Popularity score distribution</CardTitle>
          </CardHeader>
          <CardContent className="h-64 pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={popularityDistribution}>
                <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="bucket" stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} width={32} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--color-muted)" }} />
                <Bar dataKey="count" fill={CHART_ACCENT} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Average alignment feature values</CardTitle>
        </CardHeader>
        <CardContent className="h-64 pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={alignmentBreakdown} layout="vertical">
              <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" domain={[0, 1]} stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} />
              <YAxis
                type="category"
                dataKey="feature"
                stroke={CHART_MUTED}
                fontSize={12}
                tickLine={false}
                axisLine={false}
                width={90}
              />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--color-muted)" }} />
              <Bar dataKey="value" fill={CHART_ACCENT} radius={[0, 4, 4, 0]} barSize={18} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </main>
  );
}
