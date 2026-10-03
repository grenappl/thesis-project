"use client";

import { useEffect, useMemo, useState } from "react";
import { Search, Upload, Link2, FileX2 } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  Table,
  TableHeader,
  TableBody,
  TableRow,
  TableHead,
  TableCell,
} from "@/components/ui/table";
import { datasetEntries, type DatasetStatus } from "@/lib/mock-data";

const FORMAT_OPTIONS = ["All Genres", "Pop", "Hip-Hop", "Rock"] as const;

function formatSize(mb: number) {
  return mb >= 1000 ? `${(mb / 1000).toFixed(1)} GB` : `${mb.toFixed(1)} MB`;
}

export default function DatasetPage() {
  const [isLoading, setIsLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [genre, setGenre] = useState<(typeof FORMAT_OPTIONS)[number]>("All Genres");

  useEffect(() => {
    // Simulated fetch latency — replace with GET /datasets once it exists.
    const timer = setTimeout(() => setIsLoading(false), 900);
    return () => clearTimeout(timer);
  }, []);

  const filtered = useMemo(() => {
    return datasetEntries.filter((entry) => {
      const matchesSearch = entry.name.toLowerCase().includes(search.toLowerCase());
      const matchesFormat = genre === "All Genres" || entry.genre === genre;
      return matchesSearch && matchesFormat;
    });
  }, [search, genre]);

  return (
    <main className="flex-1 px-6 py-8 bg-background space-y-6 max-w-5xl w-full mx-auto">
      <div>
        <h1 className="text-2xl font-bold text-primary mb-1">Dataset</h1>
        <p className="text-sm text-muted-foreground">
          Source files feeding the ablation study and demo app.
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
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search datasets..."
            className="pl-8"
          />
        </div>

        <select
          value={genre}
          onChange={(e) => setGenre(e.target.value as typeof genre)}
          className="h-8 rounded-lg border border-input bg-background px-2.5 text-sm text-primary outline-none focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50"
        >
          {FORMAT_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt}
            </option>
          ))}
        </select>

        <div className="flex gap-2">
          <Button variant="outline" size="sm">
            <Link2 size={14} />
            Connect Source
          </Button>
          <Button
            size="sm"
            className="bg-accent hover:bg-accent-hover text-accent-foreground"
          >
            <Upload size={14} />
            Upload Dataset
          </Button>
        </div>
      </div>

      <div className="card overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Name</TableHead>
              <TableHead>Year</TableHead>
              <TableHead>Genre</TableHead>
              <TableHead>Alignment Score</TableHead>
              <TableHead>Popularity</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {isLoading &&
              Array.from({ length: 4 }).map((_, i) => (
                <TableRow key={`skeleton-${i}`}>
                  <TableCell>
                    <Skeleton className="h-4 w-36" />
                  </TableCell>
                  <TableCell>
                    <Skeleton className="h-4 w-12" />
                  </TableCell>
                  <TableCell>
                    <Skeleton className="h-4 w-16" />
                  </TableCell>
                  <TableCell>
                    <Skeleton className="h-4 w-14" />
                  </TableCell>
                  <TableCell>
                    <Skeleton className="h-4 w-16" />
                  </TableCell>
                </TableRow>
              ))}

            {!isLoading && filtered.length === 0 && (
              <TableRow>
                <TableCell colSpan={6} className="py-14">
                  <div className="flex flex-col items-center gap-2 text-center">
                    <FileX2 size={28} className="text-muted-foreground" />
                    <p className="text-sm font-medium text-primary">
                      No datasets found
                    </p>
                    <p className="text-xs text-muted-foreground max-w-xs">
                      Try a different search term or filter, or upload a new
                      dataset to get started.
                    </p>
                  </div>
                </TableCell>
              </TableRow>
            )}

            {!isLoading &&
              filtered.map((entry) => (
                <TableRow key={entry.id}>
                  <TableCell className="text-sm font-medium text-primary">
                    {entry.name}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {entry.year}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {entry.genre}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {entry.alignmentScore}
                  </TableCell>
                  <TableCell className="text-sm text-muted-foreground">
                    {entry.popularity}
                  </TableCell>
                </TableRow>
              ))}
          </TableBody>
        </Table>
      </div>
    </main>
  );
}
