"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { ChevronLeft, ChevronRight, Loader2, RotateCcw, Search } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { queryDataset } from "@/lib/dataset";
import type { DatasetFilters, DatasetQueryResult, TrackData } from "@/lib/types";

/** Editable draft state for the filter inputs. */
interface FilterFormState {
  search: string;
  year: string;
  popularityMin: string;
  popularityMax: string;
  alignmentGapMin: string;
  alignmentGapMax: string;
}

const EMPTY_FORM: FilterFormState = {
  search: "",
  year: "",
  popularityMin: "",
  popularityMax: "",
  alignmentGapMin: "",
  alignmentGapMax: "",
};

/** Parse a form field into an optional number (blank/invalid -> null). */
function toOptionalNumber(raw: string): number | null {
  const trimmed = raw.trim();
  if (trimmed === "") return null;
  const parsed = Number(trimmed);
  return Number.isFinite(parsed) ? parsed : null;
}

function toFilters(form: FilterFormState): DatasetFilters {
  const year = toOptionalNumber(form.year);
  return {
    search: form.search.trim(),
    year: year !== null && Number.isInteger(year) ? year : null,
    popularity: {
      min: toOptionalNumber(form.popularityMin),
      max: toOptionalNumber(form.popularityMax),
    },
    alignmentGap: {
      min: toOptionalNumber(form.alignmentGapMin),
      max: toOptionalNumber(form.alignmentGapMax),
    },
  };
}

function hasActiveFilters(filters: DatasetFilters): boolean {
  return (
    filters.search !== "" ||
    filters.year !== null ||
    filters.popularity.min !== null ||
    filters.popularity.max !== null ||
    filters.alignmentGap.min !== null ||
    filters.alignmentGap.max !== null
  );
}

/** Round a float column to a fixed number of decimals for display. */
function formatScore(value: number): string {
  return value.toFixed(3);
}

const COLUMNS = [
  "Track Name",
  "Artists",
  "Year",
  "Valence",
  "Lyric Sentiment",
  "Alignment Gap",
  "Popularity",
] as const;

const SELECT_CLASS =
  "h-8 rounded-lg border border-input bg-background px-2.5 text-sm text-primary outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50";

const NUMBER_INPUT_CLASS =
  "h-8 w-full min-w-0 [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none";

export default function DatasetPage() {
  const [form, setForm] = useState<FilterFormState>(EMPTY_FORM);
  const [debouncedSearch, setDebouncedSearch] = useState("");
  const [page, setPage] = useState(1);

  const [result, setResult] = useState<DatasetQueryResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Used to discard responses that arrive after a newer request was sent.
  const requestId = useRef(0);

  // Only the search text is debounced; selects and number boxes apply at once.
  useEffect(() => {
    if (form.search === debouncedSearch) return;
    const timer = setTimeout(() => {
      setDebouncedSearch(form.search);
      setPage(1);
    }, 300);
    return () => clearTimeout(timer);
  }, [form.search, debouncedSearch]);

  const committed = useMemo(
    () => toFilters({ ...form, search: debouncedSearch }),
    [form, debouncedSearch],
  );
  // Stable string key so the fetch effect only re-runs on real changes.
  const committedKey = JSON.stringify(committed);

  useEffect(() => {
    const id = ++requestId.current;
    setIsLoading(true);

    queryDataset(JSON.parse(committedKey) as DatasetFilters, page)
      .then((data) => {
        if (id !== requestId.current) return; // Stale response.
        setResult(data);
        setError(null);
        // The server clamps out-of-range pages; stay in sync with it.
        if (data.page !== page) setPage(data.page);
      })
      .catch((err: unknown) => {
        if (id !== requestId.current) return;
        setError(err instanceof Error ? err.message : "Failed to load dataset.");
      })
      .finally(() => {
        if (id === requestId.current) setIsLoading(false);
      });
  }, [committedKey, page]);

  const update = <K extends keyof FilterFormState>(
    key: K,
    value: FilterFormState[K],
  ) => {
    setForm((prev) => ({ ...prev, [key]: value }));
    // Search resets the page when its debounce settles (above).
    if (key !== "search") setPage(1);
  };

  const reset = () => {
    setForm(EMPTY_FORM);
    setDebouncedSearch("");
    setPage(1);
  };

  const facets = result?.facets;
  const rows: TrackData[] = result?.rows ?? [];
  const totalPages = result?.totalPages ?? 1;
  const currentPage = result?.page ?? page;
  const total = result?.total ?? 0;
  const pageStart = ((result?.page ?? 1) - 1) * (result?.pageSize ?? 0);

  return (
    <main className="flex-1 px-6 py-8 bg-background space-y-6 max-w-5xl w-full mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-primary mb-1">Dataset</h1>
        <p className="text-sm text-muted-foreground">
          Total number of track records feeding the ablation study and demo app.
        </p>
      </div>

      {/* ------------------------------ Filters ------------------------------ */}
      <div className="card p-4 space-y-3">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          <div className="relative flex-1">
            <Search
              size={15}
              className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground"
            />
            <Input
              value={form.search}
              onChange={(e) => update("search", e.target.value)}
              placeholder="Search track name or artist..."
              aria-label="Search tracks by name or artist"
              className="pl-8"
            />
          </div>

          <select
            value={form.year}
            onChange={(e) => update("year", e.target.value)}
            aria-label="Filter by year"
            className={SELECT_CLASS}
          >
            <option value="">All years</option>
            {facets?.years.map((year) => (
              <option key={year} value={year}>
                {year}
              </option>
            ))}
          </select>

          <Button
            variant="outline"
            onClick={reset}
            disabled={!hasActiveFilters(committed) && form.search === ""}
            aria-label="Clear all filters"
          >
            <RotateCcw />
            Reset
          </Button>
        </div>

        <div className="grid gap-3 sm:grid-cols-2">
          <fieldset className="space-y-1.5">
            <legend className="text-xs font-medium text-muted-foreground">
              Popularity (0-100)
            </legend>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                inputMode="numeric"
                value={form.popularityMin}
                onChange={(e) => update("popularityMin", e.target.value)}
                placeholder="Min"
                aria-label="Minimum popularity"
                className={NUMBER_INPUT_CLASS}
              />
              <span className="text-xs text-muted-foreground">to</span>
              <Input
                type="number"
                inputMode="numeric"
                value={form.popularityMax}
                onChange={(e) => update("popularityMax", e.target.value)}
                placeholder="Max"
                aria-label="Maximum popularity"
                className={NUMBER_INPUT_CLASS}
              />
            </div>
          </fieldset>

          <fieldset className="space-y-1.5">
            <legend className="text-xs font-medium text-muted-foreground">
              Alignment gap (-2 to 2)
            </legend>
            <div className="flex items-center gap-2">
              <Input
                type="number"
                inputMode="decimal"
                step="0.01"
                value={form.alignmentGapMin}
                onChange={(e) => update("alignmentGapMin", e.target.value)}
                placeholder="Min"
                aria-label="Minimum alignment gap"
                className={NUMBER_INPUT_CLASS}
              />
              <span className="text-xs text-muted-foreground">to</span>
              <Input
                type="number"
                inputMode="decimal"
                step="0.01"
                value={form.alignmentGapMax}
                onChange={(e) => update("alignmentGapMax", e.target.value)}
                placeholder="Max"
                aria-label="Maximum alignment gap"
                className={NUMBER_INPUT_CLASS}
              />
            </div>
          </fieldset>
        </div>

        <p
          className="flex items-center gap-1.5 text-xs text-muted-foreground"
          role="status"
          aria-live="polite"
        >
          {isLoading && <Loader2 size={12} className="animate-spin" />}
          {isLoading
            ? result
              ? "Filtering…"
              : "Loading dataset (first load parses the CSVs)…"
            : `${total.toLocaleString()} tracks match`}
        </p>
      </div>

      {error && (
        <p className="text-sm text-destructive" role="alert">
          {error}
        </p>
      )}

      {/* ------------------------------- Table ------------------------------- */}
      <div
        className={`card overflow-hidden transition-opacity ${
          isLoading && result ? "opacity-60" : ""
        }`}
      >
        <Table>
          <TableHeader>
            <TableRow>
              {COLUMNS.map((column) => (
                <TableHead key={column}>{column}</TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {result && rows.length === 0 && (
              <TableRow>
                <TableCell
                  colSpan={COLUMNS.length}
                  className="text-center text-muted-foreground py-10"
                >
                  No tracks match your filters.
                </TableCell>
              </TableRow>
            )}
            {rows.map((track, i) => (
              <TableRow key={`${track.track_name}-${track.year}-${pageStart + i}`}>
                <TableCell className="max-w-[16rem] truncate text-sm font-medium text-primary">
                  {track.track_name}
                </TableCell>
                <TableCell className="max-w-56 truncate text-sm text-muted-foreground">
                  {track.artists}
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {track.year}
                </TableCell>
                <TableCell className="text-sm tabular-nums text-muted-foreground">
                  {formatScore(track.valence)}
                </TableCell>
                <TableCell className="text-sm tabular-nums text-muted-foreground">
                  {formatScore(track.lyric_sentiment)}
                </TableCell>
                <TableCell className="text-sm tabular-nums text-muted-foreground">
                  {formatScore(track.alignment_gap)}
                </TableCell>
                <TableCell className="text-sm tabular-nums text-muted-foreground">
                  {track.popularity}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {/* ----------------------------- Pagination ---------------------------- */}
      <nav className="flex items-center justify-between" aria-label="Pagination">
        <p className="text-xs text-muted-foreground">
          Page {currentPage} of {totalPages} — {total.toLocaleString()} track
          {total === 1 ? "" : "s"}
        </p>
        <div className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Previous page"
            disabled={currentPage <= 1 || isLoading}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
          >
            <ChevronLeft size={14} />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            aria-label="Next page"
            disabled={currentPage >= totalPages || isLoading}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
          >
            <ChevronRight size={14} />
          </Button>
        </div>
      </nav>
    </main>
  );
}