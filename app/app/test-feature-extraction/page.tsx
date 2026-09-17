"use client";

import { useEffect, useRef, useState, type FormEvent } from "react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import { Separator } from "@/components/ui/separator";
import { Textarea } from "@/components/ui/textarea";
import { buttonVariants } from "@/components/ui/button";

type DemoFeatures = {
  energy: number;
  tempo: number;
  spectral_centroid: number;
  speechiness: number;
  liveness: number;
  danceability: number;
  valence: number;
  loudness: number;
  acousticness: number;
  instrumentalness: number;
  lyric_sentiment: number | null;
  alignment_gap: number | null;
  duration_ms: number;
  key: number;
  mode: number;
};

const PITCH_CLASS_NAMES = [
  "C", "C#/Db", "D", "D#/Eb", "E", "F",
  "F#/Gb", "G", "G#/Ab", "A", "A#/Bb", "B",
];

function formatDuration(ms: number): string {
  const totalSeconds = Math.round(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

type LogEntry = { stream: "stdout" | "stderr"; line: string };

type StreamEvent =
  | { type: "log"; stream: "stdout" | "stderr"; line: string }
  | { type: "result"; data: DemoFeatures }
  | { type: "error"; message: string };

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8010";

type Status = "idle" | "loading" | "error" | "done";

export default function TestFeatureExtractionPage() {
  const [file, setFile] = useState<File | null>(null);
  const [lyrics, setLyrics] = useState("");
  const [status, setStatus] = useState<Status>("idle");
  const [error, setError] = useState<string | null>(null);
  const [features, setFeatures] = useState<DemoFeatures | null>(null);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [elapsedMs, setElapsedMs] = useState(0);

  useEffect(() => {
    if (status !== "loading") return;
    const start = Date.now();
    const id = setInterval(() => setElapsedMs(Date.now() - start), 100);
    return () => clearInterval(id);
  }, [status]);

  async function handleSubmit(e: FormEvent) {
    e.preventDefault();
    if (!file) return;

    setStatus("loading");
    setError(null);
    setFeatures(null);
    setLogs([]);
    setElapsedMs(0);

    const formData = new FormData();
    formData.append("audio", file);
    if (lyrics.trim()) {
      formData.append("lyrics", lyrics);
    }

    try {
      const res = await fetch(`${API_URL}/demo/extract-features/stream`, {
        method: "POST",
        body: formData,
      });

      if (!res.body) {
        throw new Error("This browser doesn't support streaming responses.");
      }
      if (!res.ok) {
        throw new Error(`Request failed (${res.status})`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          handleEvent(JSON.parse(line) as StreamEvent);
        }
      }
      if (buffer.trim()) {
        handleEvent(JSON.parse(buffer) as StreamEvent);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong");
      setStatus("error");
    }

    function handleEvent(event: StreamEvent) {
      if (event.type === "log") {
        setLogs((prev) => [...prev, { stream: event.stream, line: event.line }]);
      } else if (event.type === "result") {
        setFeatures(event.data);
        setStatus("done");
      } else if (event.type === "error") {
        setError(event.message);
        setStatus("error");
      }
    }
  }

  return (
    <div className="flex flex-1 justify-center bg-muted/30 px-6 py-16">
      <div className="flex w-full max-w-6xl flex-col gap-6">
        <div className="flex flex-col gap-1">
          <p className="text-sm font-medium tracking-wide text-muted-foreground uppercase">
            Internal test tool
          </p>
          <h1 className="text-2xl font-semibold tracking-tight">
            Pipeline 2 feature extraction
          </h1>
          <p className="text-sm text-muted-foreground">
            A new song has two inputs, same as a dataset track — its audio
            and its lyrics — except neither has ever been scored by Spotify.
            Upload a short audio clip (3s or longer) and optionally paste
            its lyrics to run both through the demo-app feature-extraction
            pipeline — Librosa, PANNs, Essentia, TF-Hub VGGish + Ridge
            regression, the speechiness regression, and VADER — and see what
            it produces.
          </p>
        </div>

        {/* Top-left: inputs. Bottom-left: logs. Right (full height): outputs. */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2 lg:grid-rows-[auto_1fr] lg:items-start">
          <Card className="lg:col-start-1 lg:row-start-1">
            <CardHeader>
              <CardTitle>Upload audio + lyrics</CardTitle>
              <CardDescription>
                Audio: any format Librosa can decode, a few seconds or
                longer works best — very short clips may not give the
                embedding-based features enough audio for a reliable result.
                Lyrics are optional — omit for instrumental tracks.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="flex flex-col gap-4">
                <div className="flex flex-col gap-2">
                  <Label htmlFor="audio-file">Audio file</Label>
                  <Input
                    id="audio-file"
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setFile(e.target.files?.[0] ?? null)}
                  />
                </div>

                <div className="flex flex-col gap-2">
                  <Label htmlFor="lyrics-input">Lyrics (optional)</Label>
                  <Textarea
                    id="lyrics-input"
                    placeholder="Paste the song's lyrics here to also get lyric sentiment + the alignment gap..."
                    rows={5}
                    value={lyrics}
                    onChange={(e) => setLyrics(e.target.value)}
                  />
                </div>

                {/* Plain native <button>, not the shadcn/Base UI Button — that
                    wrapper's disabled-state tracking was silently swallowing
                    clicks after the disabled->enabled transition on this
                    Next 16/React 19/Turbopack stack. Styled identically via
                    the same buttonVariants() the wrapper itself uses. */}
                <button
                  type="submit"
                  disabled={!file || status === "loading"}
                  className={buttonVariants({ variant: "default", size: "default", className: "w-fit" })}
                >
                  {status === "loading"
                    ? `Extracting… (${(elapsedMs / 1000).toFixed(1)}s)`
                    : "Extract features"}
                </button>
              </form>
            </CardContent>
          </Card>

          <div className="lg:col-start-1 lg:row-start-2">
            <LogPanel logs={logs} live={status === "loading"} />
          </div>

          <div className="lg:col-start-2 lg:row-start-1 lg:row-span-2">
            <OutputPanel status={status} error={error} features={features} />
          </div>
        </div>
      </div>
    </div>
  );
}

function LogPanel({ logs, live }: { logs: LogEntry[]; live: boolean }) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [logs]);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          Logs
          {live && (
            <Badge variant="secondary" className="animate-pulse">
              live
            </Badge>
          )}
        </CardTitle>
        <CardDescription>
          Raw stdout/stderr from the WSL2 subprocess — this shells out to WSL2
          and loads the TensorFlow/PyTorch models fresh each request, so
          silence for the first few seconds is expected.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="max-h-80 overflow-y-auto rounded-lg border bg-black/95 p-3 font-mono text-xs leading-relaxed">
          {logs.length === 0 ? (
            <span className="text-zinc-500">Waiting for output…</span>
          ) : (
            logs.map((entry, i) => (
              <div
                key={i}
                className={
                  entry.stream === "stderr" ? "text-amber-400/80" : "text-zinc-300"
                }
              >
                {entry.line}
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </CardContent>
    </Card>
  );
}

function OutputPanel({
  status,
  error,
  features,
}: {
  status: Status;
  error: string | null;
  features: DemoFeatures | null;
}) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>Extracted features</CardTitle>
        {!features && !error && (
          <CardDescription>
            {status === "loading"
              ? "Waiting for results…"
              : "Upload audio (and optionally lyrics) on the left, then click Extract features."}
          </CardDescription>
        )}
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        {error && (
          <Alert variant="destructive">
            <AlertTitle>Extraction failed</AlertTitle>
            <AlertDescription className="whitespace-pre-wrap">
              {error}
            </AlertDescription>
          </Alert>
        )}

        {features && <FeatureResults features={features} />}

        {!features && !error && status !== "loading" && (
          <p className="text-sm text-muted-foreground/60 italic">
            No results yet.
          </p>
        )}
      </CardContent>
    </Card>
  );
}

function FeatureResults({ features }: { features: DemoFeatures }) {
  return (
    <>
      <div className="flex flex-col gap-4">
        <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          Spotify-scale features [0, 1]
        </h2>
        <ScaleBar
          label="Speechiness"
          value={features.speechiness}
          source="Google TF-Hub VGGish embedding + Ridge regression"
          explanation="Estimates how much of the track is spoken word versus music (0 = pure music, 1 = pure speech). Same VGGish-embedding-plus-Ridge-regression approach as Valence/Acousticness/Instrumentalness/Danceability/Energy, fit on real Spotify speechiness values — replaced an earlier regression calibrated only on synthetic text-to-speech clips, which real-song validation showed was weak."
        />
        <ScaleBar
          label="Liveness"
          value={features.liveness}
          source="PANNs CNN14 (PyTorch, AudioSet-pretrained)"
          explanation="Estimates the probability an audience was present when this was recorded. There's no dedicated liveness model, so this reuses a general-purpose sound classifier (PANNs, trained on 527 everyday sound categories) and averages its predicted probability across the applause/cheering/crowd-noise categories specifically."
        />
        <ScaleBar
          label="Valence"
          value={features.valence}
          source="Google TF-Hub VGGish embedding + Ridge regression"
          explanation="A measure of musical positivity — high values sound cheerful/euphoric, low values sound sad/angry. Google's official VGGish model turns the audio into a 128-dim embedding, then a Ridge regression (fit on ~490k real Spotify tracks) maps that embedding onto a valence score. Real-song validation: correlation 0.71 against actual Spotify values."
        />
        <ScaleBar
          label="Acousticness"
          value={features.acousticness}
          source="Google TF-Hub VGGish embedding + Ridge regression"
          explanation="Confidence that the track is acoustic (not electronic/synthesized). Same VGGish embedding as Valence, a separate Ridge regression fit on real Spotify acousticness values. Real-song validation: correlation 0.86 — the strongest-tracking feature in the pipeline besides energy."
        />
        <ScaleBar
          label="Instrumentalness"
          value={features.instrumentalness}
          source="Google TF-Hub VGGish embedding + Ridge regression"
          explanation="Confidence that the track has no vocals. Same VGGish embedding as Valence and Acousticness, a separate Ridge regression fit on real Spotify instrumentalness values. Real-song validation: correlation 0.65."
        />
      </div>

      <Separator />

      <div className="flex flex-col gap-3">
        <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
          Raw descriptors
        </h2>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
          <RawStat
            label="Energy"
            value={features.energy.toFixed(4)}
            source="Google TF-Hub VGGish embedding + Ridge regression"
            explanation="A measure of intensity and activity — Spotify describes energetic tracks as loud, fast, and noisy. Same VGGish-embedding-plus-Ridge-regression approach as Valence/Acousticness/Instrumentalness, fit on real Spotify energy values. Real-song validation: correlation 0.92 — currently the single strongest-tracking feature in the pipeline."
          />
          <RawStat
            label="Loudness"
            value={`${features.loudness.toFixed(2)} dB`}
            source="Essentia ReplayGain algorithm"
            explanation="Integrated loudness in decibels, via the ReplayGain 1.0 specification (equal-loudness-filtered signal energy) — a real signal-processing algorithm, not an ML model, and the same units Spotify's loudness column uses. Typical range is roughly -60 to 0 dB, quieter tracks closer to -60."
          />
          <RawStat
            label="Tempo"
            value={`${features.tempo.toFixed(1)} BPM`}
            source="Librosa (onset-strength autocorrelation)"
            explanation="The track's estimated speed in beats per minute, found by detecting the periodicity in how the audio's onset strength (note-attack energy) rises and falls over time."
          />
          <RawStat
            label="Spectral centroid"
            value={`${features.spectral_centroid.toFixed(0)} Hz`}
            source="Librosa"
            explanation="The frequency spectrum's 'center of mass' in Hz — a rough proxy for brightness. Higher values sound sharper/brighter, lower values sound darker/bassier. Not a Spotify feature — included since it falls out of the same Librosa analysis as tempo."
          />
          <RawStat
            label="Danceability"
            value={features.danceability.toFixed(2)}
            source="Google TF-Hub VGGish embedding + Ridge regression"
            explanation="Estimates how suitable a track is for dancing. Same VGGish-embedding-plus-Ridge-regression approach as Valence/Acousticness/Instrumentalness/Energy, fit on real Spotify danceability values — replaced an earlier Essentia-algorithm approach that real-song validation showed was weak. Real-song validation: correlation 0.73."
          />
          <RawStat
            label="Duration"
            value={formatDuration(features.duration_ms)}
            source="Librosa — clip length"
            explanation="The audio clip's length. Not estimated from anything — it's just the duration of the file you uploaded, in the same units (ms internally) as the dataset's duration_ms column."
          />
          <RawStat
            label="Key"
            value={PITCH_CLASS_NAMES[features.key]}
            source="Librosa chroma + Krumhansl-Schmuckler"
            explanation="The track's musical key (which of the 12 pitch classes it's centered on), detected by averaging the track's pitch-class energy (chroma) over time and finding which of 24 major/minor key profiles it correlates with best — the standard Krumhansl-Schmuckler technique."
          />
          <RawStat
            label="Mode"
            value={features.mode === 1 ? "Major" : "Minor"}
            source="Same key-detection step as Key"
            explanation="Whether the detected key is major or minor, from the same chroma-profile correlation used for Key — major and minor each have a different expected pitch-class weighting, and whichever fits best wins."
          />
        </div>
      </div>

      {/* Loose != null: the streaming endpoint omits these keys entirely
          (not an explicit null) when no lyrics were submitted, so
          `undefined` needs to be treated the same as `null` here. */}
      {features.lyric_sentiment != null && (
        <>
          <Separator />
          <SentimentSection
            lyricSentiment={features.lyric_sentiment}
            alignmentGap={features.alignment_gap}
            valence={features.valence}
          />
        </>
      )}
    </>
  );
}

function SentimentSection({
  lyricSentiment,
  alignmentGap,
  valence,
}: {
  lyricSentiment: number;
  alignmentGap: number | null;
  valence: number;
}) {
  const valenceNormalized = 2 * valence - 1;
  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
        Lyric sentiment &amp; alignment [-1, 1]
      </h2>

      <BipolarBar
        label="Lyric sentiment"
        value={lyricSentiment}
        source="VADER (vaderSentiment) — compound score"
        explanation="How positive or negative the lyrics read, from -1 (very negative) to +1 (very positive). Computed with VADER, a rule-based sentiment analyzer built for informal text — it understands emphasis from ALL CAPS and punctuation like &quot;!&quot;, which is why the lyrics aren't lowercased before scoring."
      />

      {alignmentGap != null && (
        <>
          <BipolarBar
            label="Alignment gap"
            value={alignmentGap}
            source="lyric_sentiment - valence_normalized"
            badge={
              alignmentGap > 0.05
                ? "lyrics more positive"
                : alignmentGap < -0.05
                  ? "music more positive"
                  : "aligned"
            }
            explanation="The difference between how the lyrics read and how the music sounds. Positive means the lyrics are more upbeat than the instrumental suggests; negative means the instrumental sounds happier than the words. Computed as lyric sentiment minus the song's valence, rescaled onto the same -1..1 scale."
          />
          <p className="text-xs text-muted-foreground">
            Formula: <code className="rounded bg-muted px-1 py-0.5">alignment_gap = lyric_sentiment - valence_normalized</code>,
            where <code className="rounded bg-muted px-1 py-0.5">valence_normalized = (2 * valence) - 1</code> rescales
            valence ({valence.toFixed(4)}) from [0, 1] onto the same [-1, 1] scale as VADER —
            here that&apos;s {valenceNormalized.toFixed(4)}.
          </p>
        </>
      )}
    </div>
  );
}

function FeatureExplanation({ children }: { children: string }) {
  return (
    <details className="text-xs text-muted-foreground/70">
      <summary className="cursor-pointer font-medium select-none hover:text-foreground">
        What is this?
      </summary>
      <p className="mt-1.5 leading-relaxed text-muted-foreground">{children}</p>
    </details>
  );
}

function BipolarBar({
  label,
  value,
  source,
  badge,
  explanation,
}: {
  label: string;
  value: number;
  source: string;
  badge?: string;
  explanation: string;
}) {
  const clamped = Math.max(-1, Math.min(1, value));
  const fillPct = Math.abs(clamped) * 50; // half the bar, from center outward
  const fromLeft = clamped >= 0;

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="flex items-center gap-2 font-medium">
          {label}
          {badge && (
            <Badge variant="secondary" className="font-normal">
              {badge}
            </Badge>
          )}
        </span>
        <span className="text-muted-foreground tabular-nums">
          {value.toFixed(4)}
        </span>
      </div>
      <div className="relative h-2 w-full overflow-hidden rounded-full bg-muted">
        <div className="absolute top-0 bottom-0 left-1/2 w-px bg-border" />
        <div
          className="absolute top-0 bottom-0 rounded-full bg-primary"
          style={
            fromLeft
              ? { left: "50%", width: `${fillPct}%` }
              : { right: "50%", width: `${fillPct}%` }
          }
        />
      </div>
      <span className="text-xs text-muted-foreground/70">{source}</span>
      <FeatureExplanation>{explanation}</FeatureExplanation>
    </div>
  );
}

function ScaleBar({
  label,
  value,
  source,
  explanation,
}: {
  label: string;
  value: number;
  source: string;
  explanation: string;
}) {
  const pct = Math.max(0, Math.min(1, value)) * 100;
  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex items-center justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span className="text-muted-foreground tabular-nums">
          {value.toFixed(4)}
        </span>
      </div>
      <Progress value={pct} />
      <span className="text-xs text-muted-foreground/70">{source}</span>
      <FeatureExplanation>{explanation}</FeatureExplanation>
    </div>
  );
}

function RawStat({
  label,
  value,
  source,
  note,
  explanation,
}: {
  label: string;
  value: string;
  source: string;
  note?: string;
  explanation: string;
}) {
  return (
    <div className="flex flex-col gap-1 rounded-lg border bg-card px-3 py-2.5">
      <span className="text-xs font-medium text-muted-foreground">
        {label}
      </span>
      <span className="text-lg font-semibold tabular-nums">{value}</span>
      <span className="text-xs text-muted-foreground/70">{source}</span>
      {note && (
        <span className="text-xs text-amber-700/80 dark:text-amber-400/80">
          {note}
        </span>
      )}
      <FeatureExplanation>{explanation}</FeatureExplanation>
    </div>
  );
}
