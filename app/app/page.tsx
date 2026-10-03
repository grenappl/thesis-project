"use client";

import { useState } from "react";
import PredictorForm from "@/components/PredictorForm";
import PredictionResults from "@/components/PredictionResults";
import { PredictionError } from "@/lib/api";
import { PredictionResult } from "@/lib/types";
import { Loader2 } from "lucide-react";
import { MOCK_PREDICTION } from "@/lib/mock-data";

type ViewState = "form" | "loading" | "results";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function HomePage() {
  const [view, setView] = useState<ViewState>("form");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(audioFile: File, lyrics: string) {
    setError(null);
    setView("loading");
    try {
      // temp — replace with: await predictPopularity({ audioFile, lyrics })
      await sleep(2000);
      setResult(MOCK_PREDICTION);
      setView("results");
    } catch (err) {
      setError(
        err instanceof PredictionError
          ? err.message
          : "Something went wrong while predicting popularity. Please try again."
      );
      setView("form");
    }
  }

  function handleReset() {
    setResult(null);
    setError(null);
    setView("form");
  }

  return (
    <main className="flex-1 px-4 py-12 bg-background">
      {error && (
        <div className="w-full max-w-xl mx-auto mb-6 rounded-lg border border-destructive/30 bg-destructive/10 text-destructive text-sm px-4 py-3">
          {error}
        </div>
      )}

      {view === "loading" && (
        <div className="w-full h-full max-w-xl mx-auto flex flex-col items-center justify-center py-16">
          <Loader2 className="animate-spin text-accent mb-2" size={48} />
          <p className="text-sm text-muted-foreground">
            Analyzing audio and lyrics...
          </p>
        </div>
      )}

      {view === "form" && (
        <PredictorForm onSubmit={handleSubmit} isLoading={false} />
      )}

      {view === "results" && result && (
        <PredictionResults result={result} onReset={handleReset} />
      )}
    </main>
  );
}
