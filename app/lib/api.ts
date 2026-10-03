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
  return {
    popularity: data.popularity,
    alignmentScore: data.alignment_score,
    features: {
      valence: data.features.valence,
      sentiment: data.features.sentiment,
      tempo: data.features.tempo,
      energy: data.features.energy,
    },
  };
}
