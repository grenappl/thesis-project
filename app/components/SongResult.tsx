import { FeatureContribution, PredictionResult } from "@/lib/types";
import FeatureIcon from "./ui/feature-icon";

interface SongResultProps {
  result: PredictionResult;
}

const KEY_NAMES = ["C", "C♯/D♭", "D", "D♯/E♭", "E", "F", "F♯/G♭", "G", "G♯/A♭", "A", "A♯/B♭", "B"];

const LABELS: Record<string, string> = {
  tempo: "Tempo",
  loudness: "Loudness",
  key: "Key",
  mode: "Mode",
  energy: "Energy",
  danceability: "Danceability",
  speechiness: "Speechiness",
  acousticness: "Acousticness",
  instrumentalness: "Instrumentalness",
  liveness: "Liveness",
  duration_ms: "Duration",
  valence: "Valence",
  lyric_sentiment: "Lyric sentiment",
  alignment_gap: "Alignment gap",
};

const signed = (v: number, digits = 2) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(digits)}`;

/** Each feature in its own units, not forced onto a 0-1 scale. */
function formatValue({ feature, value }: FeatureContribution): string {
  switch (feature) {
    case "tempo":
      return `${Math.round(value)} BPM`;
    case "loudness":
      return `${value.toFixed(1)} dB`;
    case "key":
      return KEY_NAMES[value] ?? String(value);
    case "mode":
      return value === 1 ? "Major" : "Minor";
    case "duration_ms": {
      const total = Math.round(value / 1000);
      return `${Math.floor(total / 60)}:${String(total % 60).padStart(2, "0")}`;
    }
    case "lyric_sentiment":
    case "alignment_gap":
      return signed(value);
    default:
      return value.toFixed(2);
  }
}

function popularityLabel(score: number): string {
  if (score >= 70) return "High commercial potential";
  if (score >= 40) return "Moderate commercial potential";
  return "Limited commercial potential";
}

function alignmentSentence(gap: number): string {
  if (Math.abs(gap) < 0.1) return "The lyrics and the music carry almost exactly the same emotional tone.";
  return gap > 0
    ? "The lyrics read more positive than the music sounds."
    : "The music sounds more positive than the lyrics read.";
}

/** Lyric sentiment and music valence as two markers on one -1..+1 axis (thesis Figure 2). */
function AlignmentAxis({ lyric, music }: { lyric: number; music: number }) {
  const at = (v: number) => `${((Math.max(-1, Math.min(1, v)) + 1) / 2) * 100}%`;
  return (
    <div className="mt-5 mb-2">
      <div className="relative h-8">
        <div className="absolute top-1/2 left-0 right-0 h-px bg-border" />
        <div className="absolute top-1/2 left-1/2 h-3 w-px -translate-y-1/2 bg-border" />
        <span
          className="absolute top-0 -translate-x-1/2 text-accent text-sm leading-none"
          style={{ left: at(lyric) }}
          title={`Lyric sentiment ${signed(lyric)}`}
        >
          ▼
        </span>
        <span
          className="absolute top-1/2 h-3.5 w-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-primary bg-background"
          style={{ left: at(music) }}
          title={`Music valence ${signed(music)}`}
        />
      </div>
      <div className="flex justify-between text-[11px] text-muted-foreground">
        <span>−1 negative</span>
        <span>0</span>
        <span>positive +1</span>
      </div>
      <div className="flex gap-4 mt-2 text-xs text-primary">
        <span>
          <span className="text-accent">▼</span> Lyric sentiment {signed(lyric)}
        </span>
        <span>
          <span className="inline-block h-2.5 w-2.5 rounded-full border-2 border-primary align-middle" />{" "}
          Music valence {signed(music)}
        </span>
      </div>
    </div>
  );
}

/** One feature: its value, and a bar showing how far it pushed the prediction up or down. */
function ContributionRow({ item, maxAbs }: { item: FeatureContribution; maxAbs: number }) {
  const pct = maxAbs > 0 ? (Math.abs(item.contribution) / maxAbs) * 50 : 0;
  const up = item.contribution >= 0;
  return (
    <div className="flex items-center gap-3 py-2">
      <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-light text-accent shrink-0">
        <FeatureIcon size={15} />
      </span>
      <span className="text-xs text-primary w-28 shrink-0">{LABELS[item.feature] ?? item.feature}</span>
      <span className="text-xs text-muted-foreground w-16 shrink-0 tabular-nums">{formatValue(item)}</span>
      <div className="relative flex-1 h-2 rounded-full bg-muted overflow-hidden">
        <div className="absolute inset-y-0 left-1/2 w-px bg-border" />
        <div
          className={`absolute inset-y-0 ${up ? "left-1/2 bg-accent" : "right-1/2 bg-destructive/70"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className={`text-xs w-12 text-right shrink-0 tabular-nums ${up ? "text-primary" : "text-destructive"}`}>
        {signed(item.contribution, 1)}
      </span>
    </div>
  );
}

export default function SongResult({ result }: SongResultProps) {
  const { features } = result;
  const popularity = Math.round(result.popularity);
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - (popularity / 100) * circumference;
  // Display-only summary of the gap for the progress bar: 100% when lyrics and
  // music agree exactly (gap 0), 0% at the largest possible mismatch (|gap| 2).
  // The model itself uses the signed gap, not this percentage.
  const coherence = Math.round((1 - Math.abs(result.alignmentGap) / 2) * 100);
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.contribution)), 0);

  return (
    <>
      <div className="text-center my-6">
        <h2 className="text-2xl font-semibold text-primary">{result.songTitle}</h2>
        {(result.artist || result.createdAt) && (
          <p className="text-sm text-muted-foreground mt-1">
            {[result.artist, result.createdAt && new Date(result.createdAt).toLocaleString(undefined, {
              dateStyle: "medium",
              timeStyle: "short",
            })].filter(Boolean).join(" · ")}
          </p>
        )}
      </div>

      <div className="card p-8 flex flex-col items-center mb-5">
        <div className="relative h-40 w-40">
          <svg viewBox="0 0 120 120" className="h-40 w-40 -rotate-90">
            <circle cx="60" cy="60" r="54" fill="none" stroke="var(--color-muted)" strokeWidth="10" />
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
            <span className="text-3xl font-bold text-primary">{popularity}</span>
            <span className="text-xs text-muted-foreground">/ 100</span>
          </div>
        </div>
        <p className="text-sm font-medium text-primary mt-4">Predicted Popularity</p>
        <p className="text-xs text-muted-foreground text-center mt-1 max-w-xs">
          On Spotify&apos;s 0–100 popularity scale. The model&apos;s average prediction is{" "}
          {result.baseValue.toFixed(1)}.
        </p>
        <span className="mt-3 inline-flex items-center gap-1.5 rounded-full bg-accent-light text-accent text-xs font-medium px-3 py-1">
          {popularityLabel(popularity)}
        </span>
      </div>

      <div className="card p-5 mb-5">
        <div className="flex items-start justify-between gap-4 mb-1">
          <div>
            <p className="text-sm font-medium text-primary">Lyric–Audio Alignment</p>
            <p className="text-xs text-muted-foreground">
              How closely the emotional tone of the lyrics matches the mood of the music.
            </p>
          </div>
          <span className="text-lg font-semibold text-primary tabular-nums">{coherence}%</span>
        </div>
        <div className="h-2 rounded-full bg-muted overflow-hidden mt-3">
          <div className="h-full rounded-full bg-accent" style={{ width: `${coherence}%` }} />
        </div>
        <div className="flex justify-between text-[11px] text-muted-foreground mt-1">
          <span>Mood–lyric divergence</span>
          <span>Strong coherence</span>
        </div>

        <AlignmentAxis lyric={result.lyricSentiment} music={result.valenceNormalized} />

        <p className="text-xs text-primary mt-3">
          Alignment gap <span className="font-semibold tabular-nums">{signed(result.alignmentGap)}</span>.{" "}
          {alignmentSentence(result.alignmentGap)}
        </p>
      </div>

      <div className="card p-5 mb-5">
        <p className="text-sm font-medium text-primary">Feature Breakdown</p>
        <p className="text-xs text-muted-foreground mb-2">
          Each audio and lyric feature, and how many popularity points it added (+) or subtracted (−)
          from the model&apos;s average. Largest effects first.
        </p>
        {features.map((item) => (
          <ContributionRow key={item.feature} item={item} maxAbs={maxAbs} />
        ))}
      </div>

      <p className="text-[11px] text-muted-foreground text-center mb-8 max-w-md mx-auto">
        The audio features above are estimated from your file, not computed by Spotify, so this
        prediction is illustrative. The model&apos;s typical error is about ±
        {Math.round(result.modelRmse)} popularity points.
      </p>
    </>
  );
}
