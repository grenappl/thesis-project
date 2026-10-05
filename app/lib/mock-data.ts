import { PredictionResult } from "./types";

// ---------------------------------------------------------------------------
// Placeholder data for pages not yet wired to a live backend endpoint.
// Replace each section with a real fetch once the corresponding FastAPI
// ---------------------------------------------------------------------------

export const MOCK_PREDICTION: PredictionResult = {
  songTitle: "Wuwa Song",
  popularity: 70,
  alignmentScore: 30,
  features: {
    danceability: 0.5,
    energy: 0.5,
    key: 2,
    loudness: 0.5,
    mode: 1,
    speechiness: 0.5,
    acousticness: 0.5,
    instrumentalness: 0.5,
    liveness: 0.5,
    tempo: 110,
    duration_ms: 0.5,
    valence: 0.5,
    sentiment: 0.5,
  },
};

export interface KpiMetric {
  label: string;
  value: string;
  delta: string;
  trend: "up" | "down" | "flat";
}

export const analyticsKpis: KpiMetric[] = [
  { label: "Total Predictions", value: "1,284", delta: "+12.4%", trend: "up" },
  { label: "Avg Popularity Score", value: "61.3", delta: "+3.1%", trend: "up" },
  { label: "Avg Alignment Score", value: "54.8%", delta: "-1.8%", trend: "down" },
  { label: "Model R²", value: "0.27", delta: "+0.02", trend: "up" },
];

export const predictionsOverTime = [
  { month: "Apr", predictions: 62 },
  { month: "May", predictions: 98 },
  { month: "Jun", predictions: 141 },
  { month: "Jul", predictions: 176 },
  { month: "Aug", predictions: 210 },
  { month: "Sep", predictions: 245 },
];

export const popularityDistribution = [
  { bucket: "0-20", count: 48 },
  { bucket: "21-40", count: 132 },
  { bucket: "41-60", count: 310 },
  { bucket: "61-80", count: 418 },
  { bucket: "81-100", count: 176 },
];

export const alignmentBreakdown = [
  { feature: "Danceability", value: 0.71 },
  { feature: "Energy", value: 0.64 },
  { feature: "Key", value: 0.58 },
  { feature: "Loudness", value: 0.8 },
  { feature: "Mode", value: 0.71 },
  { feature: "Speechiness", value: 0.64 },
  { feature: "Acousticness", value: 0.58 },
  { feature: "Instrumentalness", value: 0.8 },
  { feature: "Liveness", value: 0.71 },
  { feature: "Tempo", value: 0.64 },
  { feature: "Duration Ms", value: 0.58 },
  { feature: "Valence", value: 0.8 },
  { feature: "Sentiment", value: 0.71 }
];

export interface HistoryEntry {
  id: string;
  track: string;
  artist: string;
  popularity: number | null;
  timestamp: string; // ISO date
}

export const historyEntries: HistoryEntry[] = [
  { id: "h1", track: "Neon Static", artist: "Rue Haven", popularity: 78, timestamp: "2026-10-02T14:32:00Z" },
  { id: "h2", track: "Glass Halo", artist: "Marin Faye", popularity: 54, timestamp: "2026-10-02T09:10:00Z" },
  { id: "h3", track: "Lowtide", artist: "Oska Vale", popularity: 49,  timestamp: "2026-10-01T22:47:00Z" },
  { id: "h4", track: "Paper Moons", artist: "Isla Kerr", popularity: 31, timestamp: "2026-10-01T18:02:00Z" },
  { id: "h5", track: "Static Bloom", artist: "Nova Reign", popularity: 56, timestamp: "2026-09-30T11:15:00Z" },
  { id: "h6", track: "Amber Line", artist: "Theo Wren", popularity: 89, timestamp: "2026-09-29T20:58:00Z" },
  { id: "h7", track: "Hollow Coast", artist: "Dove Castillo", popularity: 46, timestamp: "2026-09-28T08:21:00Z" },
];