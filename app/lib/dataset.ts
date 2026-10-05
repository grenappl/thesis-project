"use server";

import { createReadStream } from "node:fs";
import { readdir } from "node:fs/promises";
import path from "node:path";

import Papa from "papaparse";

import type {
  DatasetFacets,
  DatasetFilters,
  DatasetQueryResult,
  NumericRange,
  TrackData,
} from "./types";

// ---------------------------------------------------------------------------
// Server-only dataset access + filtering.
//
// This is a "use server" module: every *exported* function is a Server Action
// callable from the client page, and everything else stays private to the
// server. Only async functions may be exported from here.
// ---------------------------------------------------------------------------

/** Rows rendered per page. */
const PAGE_SIZE = 10;

/**
 * Where the corpus might live. `DATASET_DIR` wins; the rest cover running from
 * the app root, repo root, or a monorepo workspace.
 */
const DATA_DIR_CANDIDATES = [
  process.env.DATASET_DIR,
  path.resolve(process.cwd(), "dataset"),
  path.resolve(process.cwd(), "..", "notebooks", "data_filtered"),
  path.resolve(process.cwd(), "notebooks", "data_filtered"),
  path.resolve(process.cwd(), "../../notebooks/data_filtered"),
].filter((dir): dir is string => typeof dir === "string" && dir.length > 0);

/** The columns we read out of each CSV, as they are spelled on disk. */
interface RawCsvRow {
  name?: string;
  artists?: string;
  year?: string;
  valence_norm?: string;
  lyric_sentiment?: string;
  alignment_gap?: string;
  popularity?: string;
}

interface DatasetIndex {
  tracks: TrackData[];
  facets: DatasetFacets;
  /**
   * Lowercased "track_name artists" per track, precomputed so search is a
   * cheap substring test. Index-aligned with `tracks`.
   */
  searchKeys: string[];
}

/** Parsed once per server process, since the corpus is static on disk. */
let indexPromise: Promise<DatasetIndex> | null = null;

// ----------------------------- CSV parsing --------------------------------

/** Coerce a CSV cell to a number, falling back to 0 for blanks/garbage. */
function toNumber(value: string | undefined): number {
  if (value === undefined || value === null) return 0;
  const trimmed = value.trim();
  if (trimmed === "") return 0;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : 0;
}

/** `artists` is a JSON array string (e.g. `["Linkin Park"]`); flatten it. */
function parseArtists(raw: string | undefined): string {
  const value = (raw ?? "").trim();
  if (value === "") return "";
  if (value.startsWith("[")) {
    try {
      const parsed: unknown = JSON.parse(value);
      if (Array.isArray(parsed)) {
        return parsed
          .filter((name): name is string => typeof name === "string")
          .map((name) => name.trim())
          .filter((name) => name !== "")
          .join(", ");
      }
    } catch {
      // Not valid JSON -- fall through and use the raw string.
    }
  }
  return value;
}

/** Map one parsed CSV row onto `TrackData`, renaming the source columns. */
function toTrackData(row: RawCsvRow): TrackData | null {
  const trackName = (row.name ?? "").trim();
  if (trackName === "") return null;

  return {
    track_name: trackName,
    artists: parseArtists(row.artists),
    year: toNumber(row.year),
    valence: toNumber(row.valence_norm),
    lyric_sentiment: toNumber(row.lyric_sentiment),
    alignment_gap: toNumber(row.alignment_gap),
    popularity: toNumber(row.popularity),
  };
}

/** Stream one CSV and push every usable row onto `sink`. */
function parseFile(filePath: string, sink: TrackData[]): Promise<void> {
  return new Promise((resolve, reject) => {
    Papa.parse<RawCsvRow>(createReadStream(filePath, { encoding: "utf8" }), {
      header: true,
      skipEmptyLines: true,
      // Row-by-row so the bulky `lyrics` column is never retained.
      step: (results) => {
        const track = toTrackData(results.data);
        if (track) sink.push(track);
      },
      complete: () => resolve(),
      error: (error: Error) => reject(error),
    });
  });
}

/** Distinct years and the observed bounds of the two range-filtered columns. */
function computeFacets(tracks: TrackData[]): DatasetFacets {
  if (tracks.length === 0) {
    return {
      years: [],
      popularity: { min: 0, max: 100 },
      alignmentGap: { min: -2, max: 2 },
    };
  }

  const years = new Set<number>();
  let popularityMin = Number.POSITIVE_INFINITY;
  let popularityMax = Number.NEGATIVE_INFINITY;
  let gapMin = Number.POSITIVE_INFINITY;
  let gapMax = Number.NEGATIVE_INFINITY;

  for (const track of tracks) {
    years.add(track.year);
    if (track.popularity < popularityMin) popularityMin = track.popularity;
    if (track.popularity > popularityMax) popularityMax = track.popularity;
    if (track.alignment_gap < gapMin) gapMin = track.alignment_gap;
    if (track.alignment_gap > gapMax) gapMax = track.alignment_gap;
  }

  return {
    years: [...years].sort((a, b) => a - b),
    popularity: { min: popularityMin, max: popularityMax },
    alignmentGap: {
      min: Math.floor(gapMin * 100) / 100,
      max: Math.ceil(gapMax * 100) / 100,
    },
  };
}

/** Find the first candidate directory that actually contains CSV files. */
async function resolveDataFiles(): Promise<{ dir: string; files: string[] }> {
  for (const dir of DATA_DIR_CANDIDATES) {
    try {
      const files = (await readdir(dir))
        .filter((file) => file.toLowerCase().endsWith(".csv"))
        .sort();
      if (files.length > 0) return { dir, files };
    } catch {
      // Directory doesn't exist -- try the next candidate.
    }
  }
  throw new Error(
    `No dataset CSVs found. Set DATASET_DIR or place them in one of: ${DATA_DIR_CANDIDATES.join(", ")}`,
  );
}

async function buildIndex(): Promise<DatasetIndex> {
  const { dir, files } = await resolveDataFiles();

  const tracks: TrackData[] = [];
  for (const file of files) {
    await parseFile(path.join(dir, file), tracks);
  }

  return {
    tracks,
    facets: computeFacets(tracks),
    searchKeys: tracks.map((track) =>
      `${track.track_name} ${track.artists}`.toLowerCase(),
    ),
  };
}

function getDatasetIndex(): Promise<DatasetIndex> {
  if (!indexPromise) {
    indexPromise = buildIndex().catch((error) => {
      indexPromise = null; // Don't cache a failure; allow a retry.
      throw error;
    });
  }
  return indexPromise;
}

// ------------------------------ Filtering ---------------------------------

/** Server Action arguments come from the client, so never trust their shape. */
function sanitizeNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function sanitizeRange(value: Partial<NumericRange> | undefined): NumericRange {
  return { min: sanitizeNumber(value?.min), max: sanitizeNumber(value?.max) };
}

function sanitizeFilters(input: Partial<DatasetFilters> | undefined): DatasetFilters {
  const year = sanitizeNumber(input?.year);
  return {
    search: typeof input?.search === "string" ? input.search.trim() : "",
    year: year !== null && Number.isInteger(year) ? year : null,
    popularity: sanitizeRange(input?.popularity),
    alignmentGap: sanitizeRange(input?.alignmentGap),
  };
}

function withinRange(value: number, range: NumericRange): boolean {
  if (range.min !== null && value < range.min) return false;
  if (range.max !== null && value > range.max) return false;
  return true;
}

function filterTracks(
  tracks: TrackData[],
  searchKeys: string[],
  filters: DatasetFilters,
): TrackData[] {
  const needle = filters.search.toLowerCase();

  return tracks.filter((track, i) => {
    if (needle !== "" && !searchKeys[i].includes(needle)) return false;
    if (filters.year !== null && track.year !== filters.year) return false;
    if (!withinRange(track.popularity, filters.popularity)) return false;
    if (!withinRange(track.alignment_gap, filters.alignmentGap)) return false;
    return true;
  });
}

// ----------------------------- Server Action ------------------------------

/**
 * Filter the corpus and return one page of rows. Called from the client page;
 * the filters and page live in client state, so the URL never changes.
 */
export async function queryDataset(
  rawFilters: DatasetFilters,
  rawPage: number,
): Promise<DatasetQueryResult> {
  const { tracks, searchKeys, facets } = await getDatasetIndex();
  const filters = sanitizeFilters(rawFilters);

  const filtered = filterTracks(tracks, searchKeys, filters);
  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const requested =
    Number.isInteger(rawPage) && rawPage >= 1 ? rawPage : 1;
  const page = Math.min(requested, totalPages);
  const start = (page - 1) * PAGE_SIZE;

  return {
    rows: filtered.slice(start, start + PAGE_SIZE),
    total: filtered.length,
    totalRecords: tracks.length,
    page,
    pageSize: PAGE_SIZE,
    totalPages,
    facets,
  };
}