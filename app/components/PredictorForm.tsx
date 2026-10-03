"use client";

import { useRef, useState } from "react";
import { Upload, Sparkles, ArrowRight, Loader2, FileAudio } from "lucide-react";

interface PredictorFormProps {
  onSubmit: (audioFile: File, lyrics: string) => void;
  isLoading: boolean;
}

export default function PredictorForm({
  onSubmit,
  isLoading,
}: PredictorFormProps) {
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [lyrics, setLyrics] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const wordCount = lyrics.trim() === "" ? 0 : lyrics.trim().split(/\s+/).length;
  const canSubmit = !!audioFile && lyrics.trim().length > 0 && !isLoading;

  function handleFiles(files: FileList | null) {
    if (!files || files.length === 0) return;
    const file = files[0];
    if (!/\.(mp3|wav)$/i.test(file.name)) return;
    setAudioFile(file);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragging(false);
    handleFiles(e.dataTransfer.files);
  }

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!canSubmit || !audioFile) return;
    onSubmit(audioFile, lyrics);
  }

  return (
    <div className="w-full max-w-xl mx-auto">
      <div className="flex justify-center mb-4">
        <span className="inline-flex items-center gap-1.5 rounded-full bg-accent-light text-accent text-xs font-medium px-3 py-1">
          <Sparkles size={12} />
          ML-Powered Analytics
        </span>
      </div>

      <h1 className="text-3xl font-bold text-center text-primary mb-2">
        Predict Song Popularity
      </h1>
      <p className="text-center text-muted-foreground text-sm mb-8">
        Upload an audio file and paste the song lyrics to estimate its
        Spotify popularity using AI.
      </p>

      <form onSubmit={handleSubmit} className="space-y-5">
        <div className="card p-5">
          <label className="text-sm font-medium text-primary mb-3 block">
            Audio File
          </label>

          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`rounded-xl border-2 border-dashed flex flex-col items-center justify-center text-center py-10 px-4 cursor-pointer transition-colors ${
              isDragging
                ? "border-accent bg-accent-light"
                : "border-border bg-muted/40 hover:bg-muted"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp3,.wav,audio/mpeg,audio/wav"
              className="hidden"
              onChange={(e) => handleFiles(e.target.files)}
            />

            {audioFile ? (
              <>
                <FileAudio className="text-accent mb-2" size={28} />
                <p className="text-sm font-medium text-primary">
                  {audioFile.name}
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  {(audioFile.size / (1024 * 1024)).toFixed(1)} MB - click to
                  replace
                </p>
              </>
            ) : (
              <>
                <span className="flex h-10 w-10 items-center justify-center rounded-full bg-muted text-muted-foreground mb-3">
                  <Upload size={18} />
                </span>
                <p className="text-sm font-medium text-primary">
                  Upload Audio File
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  Drag & drop your .mp3 or .wav file here
                </p>
              </>
            )}
          </div>
        </div>

        <div className="card p-5">
          <label className="text-sm font-medium text-primary mb-3 block">
            Paste Song Lyrics
          </label>
          <textarea
            value={lyrics}
            onChange={(e) => setLyrics(e.target.value)}
            placeholder={
              "Paste the complete lyrics here...\n\nI drove to the store\nNot knowing what I was looking for\nA stranger's eyes met mine\nIn the rain of the neon signs..."
            }
            rows={7}
            className="w-full resize-none rounded-lg bg-muted/40 border border-border p-3 text-sm text-primary placeholder:text-muted-foreground focus:outline-none focus:ring-1 focus:ring-accent"
          />
          <div className="flex justify-between mt-2 text-xs text-muted-foreground">
            <span>Tip: Include all verses, chorus, and bridge for best results</span>
            <span>{wordCount} words</span>
          </div>
        </div>

        <div className="flex flex-col items-center gap-2 pt-2">
          <button
            type="submit"
            disabled={!canSubmit}
            className="inline-flex items-center gap-2 rounded-full bg-accent hover:bg-accent-hover text-accent-foreground font-medium text-sm px-6 py-2.5 transition-colors enabled:cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <Loader2 size={16} className="animate-spin" />
                Predicting...
              </>
            ) : (
              <>
                Predict Popularity
                <ArrowRight size={16} />
              </>
            )}
          </button>
          {!canSubmit && !isLoading && (
            <span className="text-xs text-muted-foreground">
              Upload an audio file and paste lyrics to continue
            </span>
          )}
        </div>
      </form>
    </div>
  );
}
