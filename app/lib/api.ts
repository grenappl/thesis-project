import {
  AnalyticsStats,
  HistoryPage,
  HistoryQuery,
  PredictionInput,
  PredictionResult,
} from "./types";

// 8010, not uvicorn's default 8000 -- see CLAUDE.md and run-demo.ps1.
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010";

export class PredictionError extends Error {}

/** Thrown when a saved analysis doesn't exist (deleted, or a bad link). */
export class AnalysisNotFoundError extends PredictionError {}

/** Extracted feature record from POST /demo/extract-features(/stream). */
type ExtractedFeatures = Record<string, number | null>;

type StreamEvent =
  | { type: "log"; stream: "stdout" | "stderr"; line: string }
  | { type: "result"; data: ExtractedFeatures }
  | { type: "error"; message: string };

/** POST /predict response (api/schemas/prediction.py PredictionRead). */
interface PredictResponse {
  predicted_popularity: number;
  base_value: number;
  lyric_sentiment: number;
  valence_normalized: number;
  alignment_gap: number;
  features: { feature: string; value: number; contribution: number }[];
  model: { test_rmse: number };
}

/** POST/GET /analyses (api/schemas/analysis.py AnalysisRead). */
interface AnalysisResponse {
  id: number;
  song_title: string;
  artist: string | null;
  created_at: string;
  prediction: PredictResponse;
}

/** One item of GET /analyses (api/schemas/analysis.py AnalysisSummary). */
interface AnalysisSummaryResponse {
  id: number;
  song_title: string;
  artist: string | null;
  created_at: string;
  predicted_popularity: number;
  alignment_gap: number;
}

/** GET /analyses/stats (api/schemas/analysis.py AnalyticsRead). */
interface AnalyticsResponse {
  total: number;
  avg_popularity: number | null;
  avg_alignment_gap: number | null;
  over_time: { date: string; predictions: number }[];
  distribution: { bucket: string; count: number }[];
  feature_influence: { feature: string; mean_abs_contribution: number }[];
  model: { name: string; test_r2: number; test_rmse: number } | null;
}

async function errorDetail(res: Response): Promise<string> {
  const text = await res.text().catch(() => "");
  try {
    const body = JSON.parse(text);
    if (typeof body.detail === "string") return body.detail;
  } catch {
    // not JSON -- fall through to the raw text
  }
  return text || `Request failed (${res.status}).`;
}

async function request(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch {
    throw new PredictionError(
      `Could not reach the prediction server at ${API_BASE_URL}. Is the backend running?`
    );
  }
}

/**
 * Step 1: run Pipeline 2 on the upload. The stream endpoint reports progress
 * as it goes; `onProgress` receives each log line so the UI can show which
 * stage is running (extraction takes a minute or two).
 */
async function extractFeatures(
  audioFile: File,
  lyrics: string,
  onProgress?: (line: string) => void
): Promise<ExtractedFeatures> {
  const formData = new FormData();
  formData.append("audio", audioFile);
  formData.append("lyrics", lyrics);

  const res = await request(`${API_BASE_URL}/demo/extract-features/stream`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) throw new PredictionError(await errorDetail(res));
  if (!res.body) throw new PredictionError("This browser doesn't support streaming responses.");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let streamDone = false;

  while (!streamDone) {
    const { done, value } = await reader.read();
    streamDone = done;
    buffer += done ? decoder.decode() : decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    // keep a trailing partial line for the next chunk, unless the stream has ended
    buffer = done ? "" : (lines.pop() ?? "");

    for (const line of lines) {
      if (!line.trim()) continue;
      const event = JSON.parse(line) as StreamEvent;
      if (event.type === "log") onProgress?.(event.line);
      else if (event.type === "error") throw new PredictionError(event.message);
      else if (event.type === "result") return event.data;
    }
  }
  throw new PredictionError("Feature extraction finished without returning features.");
}

function toResult(data: AnalysisResponse): PredictionResult {
  const { prediction } = data;
  return {
    id: data.id,
    songTitle: data.song_title,
    artist: data.artist,
    createdAt: data.created_at,
    popularity: prediction.predicted_popularity,
    baseValue: prediction.base_value,
    lyricSentiment: prediction.lyric_sentiment,
    valenceNormalized: prediction.valence_normalized,
    alignmentGap: prediction.alignment_gap,
    features: prediction.features,
    modelRmse: prediction.model.test_rmse,
  };
}

/**
 * Step 2: score the extracted features with the trained model and save the
 * analysis. POST /analyses runs the same model as /predict (recomputing the
 * alignment gap itself) and keeps the result for History and Analytics.
 */
async function saveAnalysis(
  features: ExtractedFeatures,
  songTitle: string,
  artist: string | undefined
): Promise<PredictionResult> {
  const res = await request(`${API_BASE_URL}/analyses`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ song_title: songTitle, artist: artist || null, features }),
  });
  if (!res.ok) throw new PredictionError(await errorDetail(res));
  return toResult((await res.json()) as AnalysisResponse);
}

/** Song title from the file name: "My Song.mp3" -> "My Song". */
function titleFromFile(file: File): string {
  return file.name.replace(/\.[^.]+$/, "") || "Uploaded track";
}

/**
 * Audio file + lyrics -> extracted features -> trained model -> saved
 * prediction. The full SRS flow (Section 4.5): submit, extract, compute
 * alignment, predict, display.
 */
export async function predictPopularity(
  input: PredictionInput,
  onProgress?: (line: string) => void
): Promise<PredictionResult> {
  if (!input.audioFile || !input.lyrics.trim()) {
    throw new PredictionError("Both an audio file and lyrics are required.");
  }

  const features = await extractFeatures(input.audioFile, input.lyrics, onProgress);
  onProgress?.("[app] predicting popularity");
  const title = input.songTitle?.trim() || titleFromFile(input.audioFile);
  return saveAnalysis(features, title, input.artist?.trim());
}

/** A saved analysis, for the Song page. Safe to call from a server component. */
export async function getAnalysis(id: number): Promise<PredictionResult> {
  const res = await request(`${API_BASE_URL}/analyses/${id}`, { cache: "no-store" });
  if (res.status === 404) throw new AnalysisNotFoundError(await errorDetail(res));
  if (!res.ok) throw new PredictionError(await errorDetail(res));
  return toResult((await res.json()) as AnalysisResponse);
}

/** One page of saved analyses, newest first, for the History page. */
export async function listAnalyses(query: HistoryQuery): Promise<HistoryPage> {
  const params = new URLSearchParams({ limit: String(query.limit), offset: String(query.offset) });
  if (query.search.trim()) params.set("search", query.search.trim());
  if (query.days !== null) params.set("days", String(query.days));

  const res = await request(`${API_BASE_URL}/analyses?${params}`, { cache: "no-store" });
  if (!res.ok) throw new PredictionError(await errorDetail(res));
  const data = (await res.json()) as { items: AnalysisSummaryResponse[]; total: number };
  return {
    total: data.total,
    items: data.items.map((item) => ({
      id: item.id,
      songTitle: item.song_title,
      artist: item.artist,
      createdAt: item.created_at,
      popularity: item.predicted_popularity,
      alignmentGap: item.alignment_gap,
    })),
  };
}

export async function deleteAnalysis(id: number): Promise<void> {
  const res = await request(`${API_BASE_URL}/analyses/${id}`, { method: "DELETE" });
  // Already gone is the outcome the caller wanted.
  if (!res.ok && res.status !== 404) throw new PredictionError(await errorDetail(res));
}

/** Aggregates over every saved analysis, for the Analytics page. */
export async function getAnalytics(): Promise<AnalyticsStats> {
  const res = await request(`${API_BASE_URL}/analyses/stats`, { cache: "no-store" });
  if (!res.ok) throw new PredictionError(await errorDetail(res));
  const data = (await res.json()) as AnalyticsResponse;
  return {
    total: data.total,
    avgPopularity: data.avg_popularity,
    avgAlignmentGap: data.avg_alignment_gap,
    overTime: data.over_time,
    distribution: data.distribution,
    featureInfluence: data.feature_influence.map((f) => ({
      feature: f.feature,
      meanAbsContribution: f.mean_abs_contribution,
    })),
    model: data.model && {
      name: data.model.name,
      testR2: data.model.test_r2,
      testRmse: data.model.test_rmse,
    },
  };
}
