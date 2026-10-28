"use client";

import { useEffect, useState } from "react";
import PredictorForm, { type SongDetails } from "@/components/PredictorForm";
import PredictionResults from "@/components/PredictionResults";
import { PredictionError, predictPopularity } from "@/lib/api";
import { PredictionResult } from "@/lib/types";
import { Check, Loader2 } from "lucide-react";

type ViewState = "form" | "loading" | "results";

// The pipeline's own stage markers (api/pipelines/demo/pipeline.py `_log`),
// matched against the streamed log lines to show which step is running.
const STAGES: { match: string; label: string }[] = [
  { match: "loading audio", label: "Loading audio" },
  { match: "extracting Librosa features", label: "Detecting tempo, key and mode" },
  { match: "PANNs CNN14 embedding", label: "Analyzing the sound (PANNs)" },
  { match: "VGGish embedding", label: "Analyzing the sound (VGGish)" },
  { match: "predicting valence", label: "Estimating mood, energy and other audio features" },
  { match: "computing lyric sentiment", label: "Scoring lyric sentiment" },
  { match: "computing alignment gap", label: "Computing lyric–audio alignment" },
  { match: "predicting popularity", label: "Predicting popularity" },
];

function stageOf(line: string): number {
  return STAGES.findIndex((s) => line.includes(s.match));
}

export default function HomePage() {
  const [view, setView] = useState<ViewState>("form");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [stage, setStage] = useState(-1);
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (view !== "loading") return;
    const start = Date.now();
    const id = setInterval(() => setElapsed(Math.floor((Date.now() - start) / 1000)), 1000);
    return () => clearInterval(id);
  }, [view]);

  async function handleSubmit(audioFile: File, lyrics: string, details: SongDetails) {
    setError(null);
    setStage(-1);
    setElapsed(0);
    setView("loading");
    try {
      const prediction = await predictPopularity({ audioFile, lyrics, ...details }, (line) => {
        const index = stageOf(line);
        if (index >= 0) setStage((current) => Math.max(current, index));
      });
      setResult(prediction);
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
        <div className="w-full max-w-xl mx-auto mb-6 rounded-lg border border-destructive/30 bg-destructive/10 text-destructive text-sm px-4 py-3 whitespace-pre-wrap">
          {error}
        </div>
      )}

      {view === "loading" && (
        <div className="w-full max-w-md mx-auto flex flex-col items-center py-12">
          <Loader2 className="animate-spin text-accent mb-3" size={40} />
          <p className="text-sm font-medium text-primary">Analyzing audio and lyrics</p>
          <p className="text-xs text-muted-foreground mb-6">
            {elapsed}s elapsed. This usually takes one to two minutes.
          </p>
          <ol className="w-full space-y-2">
            {STAGES.map((s, i) => {
              const done = i < stage;
              const current = i === stage;
              return (
                <li key={s.match} className="flex items-center gap-2.5 text-sm">
                  <span className="flex h-5 w-5 shrink-0 items-center justify-center">
                    {done ? (
                      <Check size={15} className="text-accent" />
                    ) : current ? (
                      <Loader2 size={15} className="animate-spin text-accent" />
                    ) : (
                      <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground/40" />
                    )}
                  </span>
                  <span
                    className={
                      done || current ? "text-primary" : "text-muted-foreground"
                    }
                  >
                    {s.label}
                  </span>
                </li>
              );
            })}
          </ol>
        </div>
      )}

      {view === "form" && <PredictorForm onSubmit={handleSubmit} isLoading={false} />}

      {view === "results" && result && (
        <PredictionResults result={result} onReset={handleReset} />
      )}
    </main>
  );
}
