export interface FeatureBreakdown {
  danceability: number;
  energy: number;
  key: number;
  loudness: number;
  mode: number;
  speechiness: number;
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
  songTitle?: string;
}

// ---------------------------------------------------------------------------
// Dataset -- notebooks/data_filtered/songs_{year}.csv
// ---------------------------------------------------------------------------

/**
 * One row of the cleaned, sentiment-scored corpus.
 *
 * The source CSVs spell the track title `name` and carry the audio valence
 * rescaled onto the VADER range in `valence_norm`; both are renamed here so
 * the UI can read them consistently.
 */
export interface TrackData {
  track_name: string;
  artists: string;
  year: number;
  /** Valence rescaled to [-1, 1] (was `valence_norm` in the CSVs). */
  valence: number;
  /** VADER compound lyric sentiment in [-1, 1]. */
  lyric_sentiment: number;
  /** `lyric_sentiment - valence`, in [-2, 2]. */
  alignment_gap: number;
  /** Spotify popularity, 0-100. */
  popularity: number;
}

/** Inclusive lower/upper bound. `null` means unbounded on that side. */
export interface NumericRange {
  min: number | null;
  max: number | null;
}

/** Search + facet state for the dataset page, mirrored in the query string. */
export interface DatasetFilters {
  search: string;
  year: number | null;
  popularity: NumericRange;
  alignmentGap: NumericRange;
}

/** Distinct values and observed bounds, used to drive the filter controls. */
export interface DatasetFacets {
  years: number[];
  popularity: { min: number; max: number };
  alignmentGap: { min: number; max: number };
}

/** One page of results returned by the `queryDataset` server action. */
export interface DatasetQueryResult {
  rows: TrackData[];
  /** Rows matching the current filters. */
  total: number;
  /** All rows in the corpus. */
  totalRecords: number;
  page: number;
  pageSize: number;
  totalPages: number;
  facets: DatasetFacets;
}