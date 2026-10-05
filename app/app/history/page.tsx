"use client";

import { useMemo, useState } from "react";
import { Search, ChevronLeft, ChevronRight } from "lucide-react";
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
import { historyEntries } from "@/lib/mock-data";
import { useRouter } from "next/navigation";

const DATE_OPTIONS = ["All time", "Last 7 days", "Last 30 days"] as const;
const PAGE_SIZE = 5;

function formatDate(iso: string) {
  return new Date(iso).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function HistoryPage() {
  const [search, setSearch] = useState("");
  const [dateRange, setDateRange] = useState<(typeof DATE_OPTIONS)[number]>("All time");
  const [page, setPage] = useState(1);
  const router = useRouter()

  const filtered = useMemo(() => {
    const now = Date.now();
    const rangeMs =
      dateRange === "Last 7 days"
        ? 7 * 24 * 60 * 60 * 1000
        : dateRange === "Last 30 days"
          ? 30 * 24 * 60 * 60 * 1000
          : Infinity;

    return historyEntries.filter((entry) => {
      const matchesSearch =
        entry.track.toLowerCase().includes(search.toLowerCase()) ||
        entry.artist.toLowerCase().includes(search.toLowerCase());
      const matchesDate = now - new Date(entry.timestamp).getTime() <= rangeMs;
      return matchesSearch && matchesDate;
    });
  }, [search, dateRange]);

  const viewSong = (id: number) => {
    router.push(`/song/${id}`);
  }

  const totalPages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const pageSafe = Math.min(page, totalPages);
  const pageItems = filtered.slice((pageSafe - 1) * PAGE_SIZE, pageSafe * PAGE_SIZE);

  return (
    <main className="flex-1 px-6 py-8 bg-background space-y-6 max-w-5xl w-full mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-primary mb-1">History</h1>
        <p className="text-sm text-muted-foreground">
          Audit trail of predictions, uploads, and model runs.
        </p>
      </div>
      
      <div className="card p-4 flex flex-col sm:flex-row gap-3 sm:items-center">
        <div className="relative flex-1">
          <Search
            size={15}
            className="absolute left-2.5 top-1/2 -translate-y-1/2 text-muted-foreground"
          />
          <Input
            value={search}
            onChange={(e) => {
              setSearch(e.target.value);
              setPage(1);
            }}
            placeholder="Search by track or artist..."
            className="pl-8"
          />
        </div>

        <select
          value={dateRange}
          onChange={(e) => {
            setDateRange(e.target.value as typeof dateRange);
            setPage(1);
          }}
          className="h-8 rounded-lg border border-input bg-background px-2.5 text-sm text-primary outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          {DATE_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>
      </div>

      <div className="card overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Track</TableHead>
              <TableHead>Popularity</TableHead>
              <TableHead>Date</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {pageItems.length === 0 && (
              <TableRow>
                <TableCell colSpan={5} className="text-center text-muted-foreground py-10">
                  No entries match your filters.
                </TableCell>
              </TableRow>
            )}
            {pageItems.map((entry) => (
              <TableRow key={entry.id} className='cursor-pointer' onClick={() => viewSong(1)}>
                <TableCell>
                  <p className="text-sm font-medium text-primary">{entry.track}</p>
                  <p className="text-xs text-muted-foreground">{entry.artist}</p>
                </TableCell>
                <TableCell className="text-sm text-primary">
                  {entry.popularity ?? "—"}
                </TableCell>
                <TableCell className="text-sm text-muted-foreground">
                  {formatDate(entry.timestamp)}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <div className="flex items-center justify-between">
        <p className="text-xs text-muted-foreground">
          Page {pageSafe} of {totalPages} — {filtered.length} entr
          {filtered.length === 1 ? "y" : "ies"}
        </p>
        <div className="flex items-center gap-1.5">
          <Button
            variant="outline"
            size="icon-sm"
            disabled={pageSafe <= 1}
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            aria-label="Previous page"
          >
            <ChevronLeft size={14} />
          </Button>
          <Button
            variant="outline"
            size="icon-sm"
            disabled={pageSafe >= totalPages}
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
