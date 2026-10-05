import { PredictionInput, PredictionResult } from "./types";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class PredictionError extends Error {}

/**
 * Sends the audio file + lyrics to the FastAPI backend's /predict endpoint
 * and returns the parsed prediction result.
 */
export async function predictPopularity(
  input: PredictionInput
): Promise<PredictionResult> {
  if (!input.audioFile || !input.lyrics.trim()) {
    throw new PredictionError("Both an audio file and lyrics are required.");
  }

  const formData = new FormData();
  formData.append("audio", input.audioFile);
  formData.append("lyrics", input.lyrics);

  const response = await fetch(`${API_BASE_URL}/predict`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const message = await response.text().catch(() => "");
    throw new PredictionError(
      message || `Prediction request failed (${response.status}).`
    );
  }

  const data = await response.json();

  // Adapt the FastAPI response shape into the frontend's PredictionResult.
  // The endpoint reports the raw audio valence; rescale it onto [-1, 1] to
  // match FeatureBreakdown.valence.
  const valence = Number(data.features?.valence ?? 0);

  return {
    songTitle: input.songTitle ?? "Uploaded track",
    popularity: Number(data.predicted_popularity ?? 0),
    alignmentScore: Number(data.alignment_score ?? 0),
    features: {
      danceability: Number(data.features?.danceability ?? 0),
      energy: Number(data.features?.energy ?? 0),
      key: Number(data.features?.key ?? 0),
      loudness: Number(data.features?.loudness ?? 0),
      mode: Number(data.features?.mode ?? 0),
      speechiness: Number(data.features?.speechiness ?? 0),
      acousticness: Number(data.features?.acousticness ?? 0),
      instrumentalness: Number(data.features?.instrumentalness ?? 0),
      liveness: Number(data.features?.liveness ?? 0),
      tempo: Number(data.features?.tempo ?? 0),
      duration_ms: Number(data.features?.duration_ms ?? 0),
      valence: valence * 2 - 1,
      sentiment: Number(data.features?.sentiment ?? 0),
    },
  };
}
