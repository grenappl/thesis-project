import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import SongResult from "@/components/SongResult";
import { AnalysisNotFoundError, PredictionError, getAnalysis } from "@/lib/api";
import { PredictionResult } from "@/lib/types";

interface SongProps {
  params: Promise<{ id: string }>;
}

export default async function SongPage({ params }: SongProps) {
  const { id } = await params;
  // Number("1abc") is NaN and Number("") is 0; only plain positive integers are ids.
  if (!/^[1-9]\d*$/.test(id)) notFound();

  let result: PredictionResult;
  try {
    result = await getAnalysis(Number(id));
  } catch (err) {
    if (err instanceof AnalysisNotFoundError) notFound();
    const message =
      err instanceof PredictionError ? err.message : "Something went wrong loading this analysis.";
    return (
      <div className="w-full max-w-xl mx-auto px-4 py-12">
        <div className="rounded-lg border border-destructive/30 bg-destructive/10 text-destructive text-sm px-4 py-3 whitespace-pre-wrap">
          {message}
        </div>
      </div>
    );
  }

  return (
    <div className="w-full max-w-xl mx-auto px-4 py-8">
      <Link
        href="/history"
        className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-primary transition-colors"
      >
        <ArrowLeft size={14} />
        Back to history
      </Link>
      <SongResult result={result} />
    </div>
  );
}
