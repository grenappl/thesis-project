"use client";

import { useEffect, useState } from "react";
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
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { PredictionError, getAnalytics } from "@/lib/api";
import { getAlignmentSummary } from "@/lib/dataset";
import { AlignmentSummary, AnalyticsStats } from "@/lib/types";

interface Kpi {
  label: string;
  value: string;
  hint: string;
}

function signed(value: number, digits = 2) {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(digits)}`;
}

function featureLabel(name: string) {
  const spaced = name.replace(/_/g, " ");
  return spaced.charAt(0).toUpperCase() + spaced.slice(1);
}

function kpisFor(stats: AnalyticsStats): Kpi[] {
  const { model } = stats;
  return [
    {
      label: "Songs analysed",
      value: stats.total.toLocaleString(),
      hint: "saved predictions",
    },
    {
      label: "Avg predicted popularity",
      value: stats.avgPopularity === null ? "—" : stats.avgPopularity.toFixed(1),
      hint: model ? `model error ±${model.testRmse.toFixed(1)} points` : "out of 100",
    },
    {
      label: "Avg alignment gap",
      value: stats.avgAlignmentGap === null ? "—" : signed(stats.avgAlignmentGap),
      hint: "+ lyrics brighter than music",
    },
    {
      label: "Model R²",
      value: model ? model.testR2.toFixed(2) : "—",
      hint: model ? "held-out test set" : "model not exported",
    },
  ];
}

function KpiCard({ label, value, hint }: Kpi) {
  return (
    <Card>
      <CardContent className="pt-1">
        <p className="text-xs text-muted-foreground">{label}</p>
        <span className="block text-2xl font-semibold text-primary mt-1.5">{value}</span>
        <p className="text-xs text-muted-foreground mt-1">{hint}</p>
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

function ErrorBanner({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-destructive/30 bg-destructive/10 text-destructive text-sm px-4 py-3 whitespace-pre-wrap">
      {message}
    </div>
  );
}

export default function AnalyticsPage() {
  const [stats, setStats] = useState<AnalyticsStats | null>(null);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [corpus, setCorpus] = useState<AlignmentSummary | null>(null);
  const [corpusError, setCorpusError] = useState<string | null>(null);

  useEffect(() => {
    getAnalytics()
      .then(setStats)
      .catch((err) =>
        setStatsError(err instanceof PredictionError ? err.message : "Could not load analytics.")
      );
    getAlignmentSummary()
      .then(setCorpus)
      .catch(() => setCorpusError("Could not read the dataset to summarise it."));
  }, []);

  const empty = stats !== null && stats.total === 0;

  return (
    <main className="flex-1 px-6 py-8 bg-background space-y-6 max-w-5xl w-full mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-primary mb-1">Analytics</h1>
        <p className="text-sm text-muted-foreground">
          What your analysed songs look like, and how alignment relates to popularity across the dataset.
        </p>
      </div>

      {statsError && <ErrorBanner message={statsError} />}

      {stats && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
          {kpisFor(stats).map((kpi) => (
            <KpiCard key={kpi.label} {...kpi} />
          ))}
        </div>
      )}

      {empty && (
        <p className="text-sm text-muted-foreground">
          No songs analysed yet. Predict a song on the home page and these charts will fill in.
        </p>
      )}

      {stats && !empty && (
        <>
          <div className="grid lg:grid-cols-2 gap-4">
            <Card>
              <CardHeader>
                <CardTitle>Songs analysed per day</CardTitle>
              </CardHeader>
              <CardContent className="h-64 pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stats.overTime}>
                    <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="date" stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis allowDecimals={false} stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} width={32} />
                    <Tooltip contentStyle={tooltipStyle} />
                    <Line
                      type="monotone"
                      dataKey="predictions"
                      name="Songs"
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
                <CardTitle>Predicted popularity distribution</CardTitle>
              </CardHeader>
              <CardContent className="h-64 pt-2">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={stats.distribution}>
                    <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="bucket" stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} />
                    <YAxis allowDecimals={false} stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} width={32} />
                    <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--color-muted)" }} />
                    <Bar dataKey="count" name="Songs" fill={CHART_ACCENT} radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader>
              <CardTitle>What moves your predictions</CardTitle>
              <p className="text-xs text-muted-foreground">
                Average popularity points each feature added or removed, in either direction (SHAP).
              </p>
            </CardHeader>
            <CardContent className="h-96 pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={stats.featureInfluence.map((f) => ({
                    feature: featureLabel(f.feature),
                    value: Number(f.meanAbsContribution.toFixed(2)),
                  }))}
                  layout="vertical"
                >
                  <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" horizontal={false} />
                  <XAxis type="number" stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} />
                  <YAxis type="category" dataKey="feature" stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} width={110} interval={0} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--color-muted)" }} />
                  <Bar dataKey="value" name="Avg points" fill={CHART_ACCENT} radius={[0, 4, 4, 0]} barSize={14} />
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </>
      )}

      {corpusError && <ErrorBanner message={corpusError} />}

      {corpus && corpus.bins.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Popularity by alignment gap, across the dataset</CardTitle>
            <p className="text-xs text-muted-foreground">
              Mean Spotify popularity of {corpus.tracks.toLocaleString()} tracks, grouped by
              lyric sentiment minus normalised valence. Negative: music sounds brighter than the
              lyrics read.
              {corpus.correlationGap !== null && corpus.correlationAbsGap !== null && (
                <>
                  {" "}Correlation with popularity: signed gap r = {corpus.correlationGap.toFixed(3)},
                  size of mismatch r = {corpus.correlationAbsGap.toFixed(3)}.
                </>
              )}
            </p>
          </CardHeader>
          <CardContent className="h-72 pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={corpus.bins.map((b) => ({ ...b, meanPopularity: Number(b.meanPopularity.toFixed(1)) }))}>
                <CartesianGrid stroke={CHART_GRID} strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="label" stroke={CHART_MUTED} fontSize={11} tickLine={false} axisLine={false} />
                <YAxis stroke={CHART_MUTED} fontSize={12} tickLine={false} axisLine={false} width={32} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "var(--color-muted)" }} />
                <Bar dataKey="meanPopularity" name="Mean popularity" fill={CHART_ACCENT} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      )}
    </main>
  );
}
