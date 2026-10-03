export interface FeatureBreakdown {
  danceability: number;
  energy: number;
  key: number;
  loudness: number;
  mode: number;
  speechniness: number;
  acousticness: number;
  instrumentalness: number;
  liveness: number;
  tempo: number;
  duration_ms: number;
  valence: number;
  sentiment: number;
}

export interface PredictionResult {
  songTitle: string;
  popularity: number; // 0-100
  alignmentScore: number; // 0-100 (%)
  features: FeatureBreakdown;
}

export interface PredictionInput {
  audioFile: File | null;
  lyrics: string;
}
