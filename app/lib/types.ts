/** One model input, its value, and how far it moved the prediction. */
export interface FeatureContribution {
  /** Backend name, e.g. "tempo", "lyric_sentiment", "alignment_gap", "key". */
  feature: string;
  value: number;
  /** Popularity points this feature added (+) or subtracted (-), from TreeSHAP. */
  contribution: number;
}

/** A prediction from POST /predict, in the shape the results screen uses. */
export interface PredictionResult {
  /** Saved-analysis id; the result is viewable at /song/{id}. */
  id?: number;
  songTitle: string;
  artist?: string | null;
  /** ISO timestamp the analysis was saved. */
  createdAt?: string;
  /** Predicted Spotify popularity, 0-100. */
  popularity: number;
  /** The model's average prediction, before it sees this song's features. */
  baseValue: number;
  /** VADER compound score of the lyrics, -1 to 1. */
  lyricSentiment: number;
  /** Audio valence rescaled to -1 to 1, so it shares the lyric scale. */
  valenceNormalized: number;
  /** lyricSentiment - valenceNormalized, -2 to 2. Positive: lyrics read brighter than the music sounds. */
  alignmentGap: number;
  /** Every input, largest absolute contribution first. */
  features: FeatureContribution[];
  /** The model's typical error on the held-out test set, in popularity points. */
  modelRmse: number;
}

export interface PredictionInput {
  audioFile: File | null;
  lyrics: string;
  songTitle?: string;
  artist?: string;
}

// ---------------------------------------------------------------------------
// Saved analyses -- GET /analyses, GET /analyses/stats
// ---------------------------------------------------------------------------

/** One row of the History list. */
export interface HistoryEntry {
  id: number;
  songTitle: string;
  artist: string | null;
  /** ISO timestamp (UTC). */
  createdAt: string;
  popularity: number;
  alignmentGap: number;
}

export interface HistoryPage {
  items: HistoryEntry[];
  /** Entries matching the filters, across all pages. */
  total: number;
}

export interface HistoryQuery {
  search: string;
  /** Only the last N days; null for all time. */
  days: number | null;
  limit: number;
  offset: number;
}

/** Aggregates over every saved analysis, for the Analytics page. */
export interface AnalyticsStats {
  total: number;
  avgPopularity: number | null;
  /** Signed mean of lyric_sentiment - valence_normalized. */
  avgAlignmentGap: number | null;
  /** Analyses per UTC day, oldest first, days with none omitted. */
  overTime: { date: string; predictions: number }[];
  distribution: { bucket: string; count: number }[];
  /** Mean absolute popularity points each feature moved a prediction. */
  featureInfluence: { feature: string; meanAbsContribution: number }[];
  model: { name: string; testR2: number; testRmse: number } | null;
}

/**
 * How popularity varies with the alignment gap across the whole cleaned
 * corpus -- the thesis's research question, answered over dataset rows rather
 * than the user's own analyses.
 */
export interface AlignmentSummary {
  tracks: number;
  /** Mean popularity per alignment-gap bin, lowest gap first; empty bins omitted. */
  bins: { label: string; tracks: number; meanPopularity: number }[];
  /** Pearson r between the signed gap and popularity. */
  correlationGap: number | null;
  /** Pearson r between the gap's magnitude (mismatch either way) and popularity. */
  correlationAbsGap: number | null;
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