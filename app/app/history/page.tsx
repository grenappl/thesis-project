"use client";

import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { Search, ChevronLeft, ChevronRight, Trash2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import { deleteAnalysis, listAnalyses, PredictionError } from "@/lib/api";
import { HistoryPage } from "@/lib/types";

const DATE_OPTIONS = [
  { label: "All time", days: null },
  { label: "Last 7 days", days: 7 },
  { label: "Last 30 days", days: 30 },
] as const;
const PAGE_SIZE = 10;
const SEARCH_DEBOUNCE_MS = 300;

function formatDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function signed(value: number) {
  return `${value >= 0 ? "+" : "−"}${Math.abs(value).toFixed(2)}`;
}

export default function HistoryPageView() {
  const router = useRouter();
  const [searchInput, setSearchInput] = useState("");
  const [search, setSearch] = useState("");
  const [dayOption, setDayOption] = useState(0);
  const [page, setPage] = useState(1);
  const [data, setData] = useState<HistoryPage | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  // Typing shouldn't fire a request per keystroke.
  useEffect(() => {
    const id = setTimeout(() => {
      setSearch(searchInput);
      setPage(1);
    }, SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(id);
  }, [searchInput]);

  const days = DATE_OPTIONS[dayOption].days;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      setData(await listAnalyses({ search, days, limit: PAGE_SIZE, offset: (page - 1) * PAGE_SIZE }));
      setError(null);
    } catch (err) {
      setError(err instanceof PredictionError ? err.message : "Could not load history.");
    } finally {
      setLoading(false);
    }
  }, [search, days, page]);

  useEffect(() => {
    // Fetch-on-change: `load` sets state when the request settles.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    load();
  }, [load]);

  async function handleDelete(id: number, title: string) {
    if (!window.confirm(`Delete "${title}" from your history?`)) return;
    try {
      await deleteAnalysis(id);
      // Deleting the last row of a page would leave it empty; step back one.
      if (data && data.items.length === 1 && page > 1) setPage(page - 1);
      else await load();
    } catch (err) {
      setError(err instanceof PredictionError ? err.message : "Could not delete that entry.");
    }
  }

  const total = data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  const items = data?.items ?? [];

  return (
    <main className="flex-1 px-6 py-8 bg-background space-y-6 max-w-5xl w-full mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-primary mb-1">History</h1>
        <p className="text-sm text-muted-foreground">
          Every song you have analysed. Select one to see its full result.
        </p>
      </div>

      <div className="card p-4 flex flex-col sm:flex-row gap-3 sm:items-center">
        <div className="relative flex-1">
          <Search
            size={15}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            placeholder="Search by track or artist..."
            className="pl-8"
          />
        </div>

        <select
          value={dayOption}
          onChange={(e) => {
            setDayOption(Number(e.target.value));
            setPage(1);
          }}
          className="h-8 rounded-lg border border-input bg-background px-2.5 text-sm text-primary outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          {DATE_OPTIONS.map((opt, i) => (
            <option key={opt.label} value={i}>
              {opt.label}
            </option>
          ))}
        </select>
      </div>

      {error && (
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 text-destructive text-sm px-4 py-3 whitespace-pre-wrap">
          {error}
        </div>
      )}

      <div className={`card overflow-hidden transition-opacity ${loading ? "opacity-60" : ""}`}>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Track</TableHead>
              <TableHead>Popularity</TableHead>
              <TableHead>Alignment gap</TableHead>
              <TableHead>Date</TableHead>
              <TableHead className="w-10" />
            </TableRow>
          </TableHeader>
          <TableBody>
            {!loading && items.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted-foreground py-10">
                  {search || days !== null
                    ? "No entries match your filters."
                    : "No analyses yet. Predict a song on the home page and it will show up here."}
                </TableCell>
              </TableRow>
            )}
            {items.map((entry) => (
              <TableRow
                key={entry.id}
                className="cursor-pointer"
                onClick={() => router.push(`/song/${entry.id}`)}
              >
                <TableCell>
                  <p className="text-sm font-medium text-primary">{entry.songTitle}</p>
                  {entry.artist && <p className="text-xs text-muted-foreground">{entry.artist}</p>}
                </TableCell>
                <TableCell className="text-sm text-primary tabular-nums">
                  {Math.round(entry.popularity)}
                </TableCell>
                <TableCell className="text-sm text-primary tabular-nums">
                  {signed(entry.alignmentGap)}
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {formatDate(entry.createdAt)}
                </TableCell>
                <TableCell>
                  <Button
                    variant="ghost"
                    size="icon-sm"
                    aria-label={`Delete ${entry.songTitle}`}
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(entry.id, entry.songTitle);
                    }}
                  >
                    <Trash2 size={14} />
                  </Button>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          Page {Math.min(page, totalPages)} of {totalPages} — {total} entr{total === 1 ? "y" : "ies"}
        </p>
        <div className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="icon-sm"
            disabled={page <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            aria-label="Previous page"
          >
            <ChevronLeft size={14} />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            disabled={page >= totalPages}
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            aria-label="Next page"
          >
            <ChevronRight size={14} />
          </Button>
        </div>
      </div>
    </main>
  );
}
