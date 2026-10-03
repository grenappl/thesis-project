import { PredictionResult } from "@/lib/types";
import { capitalize } from "@/lib/utils";
import FeatureIcon from "./ui/feature-icon";

interface PredictionResultsProps {
  result: PredictionResult;
}

function popularityLabel(score: number): string {
  if (score >= 70) return "High commercial potential";
  if (score >= 40) return "Moderate commercial potential";
  return "Limited commercial potential";
}

function FeatureRow({
  label,
  value,
  display,
  max,
}: {
  label: string;
  value: number;
  display: string;
  max: number;
}) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div className="flex items-center gap-3 py-2">
      <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-light text-accent shrink-0">
        <FeatureIcon size={15} />
      </span>
      <span className="text-xs text-primary w-24 shrink-0">{label}</span>
      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
        <div
          className="h-full rounded-full bg-accent"
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="text-sm text-primary w-16 text-right shrink-0">
        {display}
      </span>
    </div>
  );
}

export default function SongResult({ result }: PredictionResultsProps) {
  const { popularity, alignmentScore, features } = result;
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - (popularity / 100) * circumference;

  return (
    <>
      <h2 className="text-2xl font-semibold text-center text-primary my-6">
        {result.songTitle}
      </h2>

      <div className="card p-8 flex flex-col items-center mb-5">
        <div className="relative h-40 w-40">
          <svg viewBox="0 0 120 120" className="h-40 w-40 -rotate-90">
            <circle
              cx="60"
              cy="60"
              r="54"
              fill="none"
              stroke="var(--color-muted)"
              strokeWidth="10"
            />
            <circle
              cx="60"
              cy="60"
              r="54"
              fill="none"
              stroke="var(--color-accent)"
              strokeWidth="10"
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={offset}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className="text-3xl font-bold text-primary">
              {popularity}
            </span>
            <span className="text-xs text-muted-foreground">/ 100</span>
          </div>
        </div>
        <p className="text-sm font-medium text-primary mt-4">
          Predicted Popularity
        </p>
        <p className="text-xs text-muted-foreground text-center mt-1 max-w-xs">
          Score out of 100 based on Spotify&apos;s popularity index, predicted
          from audio and lyric features.
        </p>
        <span className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-accent-light text-accent text-xs font-medium px-3 py-1">
          {popularityLabel(popularity)}
        </span>
      </div>

      <div className="card p-5 mb-5">
        <div className="flex items-center justify-between mb-1">
          <div>
            <p className="text-sm font-medium text-primary">
              Alignment Score
            </p>
            <p className="text-xs text-muted-foreground">
              Measures how closely the musical mood matches the emotional
              tone of the lyrics.
            </p>
          </div>
          <span className="text-lg font-semibold text-primary">
            {alignmentScore}%
          </span>
        </div>
        <div className="h-2 rounded-full bg-muted overflow-hidden mt-3">
          <div
            className="h-full rounded-full bg-accent"
            style={{ width: `${alignmentScore}%` }}
          />
        </div>
        <div className="flex justify-between text-[11px] text-muted-foreground mt-1">
          <span>Mood-lyric divergence</span>
          <span>Strong coherence</span>
        </div>
      </div>

      <div className="card p-5 mb-8">
        <p className="text-sm font-medium text-primary">Feature Breakdown</p>
        <p className="text-xs text-muted-foreground mb-2">
          Extracted audio and lyric metrics used in the prediction model.
        </p>

        {Object.entries(result.features).map(([key, value]) =>
          <FeatureRow
            key={key}
            label={capitalize(key)}
            value={value}
            display={value.toFixed(2)}
            max={1}
          />)
        }
      </div>
    </>
  )
}
