"use client";

import { ArrowLeft, CheckCircle2 } from "lucide-react";
import { PredictionResult } from "@/lib/types";
import FeatureIcon from "./ui/feature-icon";
import SongResult from "./SongResult";

interface PredictionResultsProps {
  result: PredictionResult;
  onReset: () => void;
}

export default function PredictionResults({
  result,
  onReset,
}: PredictionResultsProps) {
  return (
    <div className="w-full max-w-xl mx-auto">
      <div className="flex justify-center mb-4">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-accent-light text-accent text-xs font-medium px-3 py-1">
          <CheckCircle2 size={12} />
          Analysis Complete
        </span>
      </div>

      <h1 className="text-3xl font-bold text-center text-primary mb-2">
        Prediction Results
      </h1>
      <p className="text-center text-muted-foreground text-sm mb-8">
        ML prediction based on extracted audio features and lyric sentiment
        analysis.
      </p>

      <hr/>

      <SongResult result={result} />

      <div className="flex justify-center">
        <button
          onClick={onReset}
          className="inline-flex items-center gap-1.5 text-sm text-muted-foreground hover:text-primary transition-colors cursor-pointer"
        >
          <ArrowLeft size={14} />
          Try Another Song
        </button>
      </div>
    </div>
  );
}
