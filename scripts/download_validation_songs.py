"""Downloads real audio for a random sample of dataset tracks via yt-dlp, so
Pipeline 2's output can be checked against many real Spotify values instead
of the single manually-sourced Glue Song. Confidence-scored matching, same
idea as essteec/music-prediction's audio-acquisition scripts (title
similarity + duration + artist presence) but scaled down for a small
validation batch, not 491k rows — so it's fine to just skip a low-confidence
match rather than accept it.

Writes one mp3 per matched song to --output-dir, named by track id, plus a
manifest CSV (id, name, artists, and the real Spotify columns) that
scripts/validate_pipeline_accuracy.py reads to compare Pipeline 2's output
against.

Run from Windows (yt-dlp/ffmpeg need no Essentia/WSL2):

    uv run python scripts/download_validation_songs.py "C:\\Users\\User\\Downloads\\songs(1).csv"
"""

from __future__ import annotations

import argparse
import ast
import difflib
import re
from pathlib import Path

import pandas as pd
import yt_dlp

_TARGET_COLUMNS = [
    "danceability", "energy", "key", "loudness", "mode", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo",
    "duration_ms",
]

_MIN_CONFIDENCE = 60


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _confidence_score(track_name: str, artists: list[str], duration_ms: float, candidate: dict) -> float:
    title = _normalize(candidate.get("title", ""))
    if not title:
        return 0.0

    title_score = difflib.SequenceMatcher(None, _normalize(track_name), title).ratio() * 40

    candidate_duration = candidate.get("duration")
    if candidate_duration:
        diff = abs(candidate_duration * 1000 - duration_ms)
        if diff > max(60_000, duration_ms * 0.3):
            return 0.0  # hard reject: almost certainly the wrong recording
        duration_score = 30 if diff <= 5_000 else 20 if diff <= 15_000 else 10 if diff <= 30_000 else 0
    else:
        duration_score = 0

    artist_score = 30 if any(_normalize(a) in title for a in artists) else 0

    return title_score + duration_score + artist_score


def _find_best_match(ydl: yt_dlp.YoutubeDL, track_name: str, artists: list[str], duration_ms: float) -> dict | None:
    query = f"ytsearch5:{track_name} {' '.join(artists[:3])}"
    try:
        results = ydl.extract_info(query, download=False)
    except yt_dlp.utils.DownloadError:
        return None

    candidates = results.get("entries") or []
    scored = [(c, _confidence_score(track_name, artists, duration_ms, c)) for c in candidates]
    scored = [(c, s) for c, s in scored if s >= _MIN_CONFIDENCE]
    if not scored:
        return None
    return max(scored, key=lambda pair: pair[1])[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("songs_csv", help="Path to the raw dataset CSV (songs(1).csv)")
    parser.add_argument("--n", type=int, default=20, help="Number of songs to sample")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="data/validation_songs")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"

    existing_manifest = pd.read_csv(manifest_path) if manifest_path.exists() else None
    already_have = set(existing_manifest["id"]) if existing_manifest is not None else set()

    print("loading dataset...")
    columns = ["id", "name", "artists", *_TARGET_COLUMNS]
    songs = pd.read_csv(args.songs_csv, usecols=columns).dropna(subset=_TARGET_COLUMNS)
    songs = songs[~songs["id"].isin(already_have)]
    sample = songs.sample(n=args.n, random_state=args.seed)

    manifest_rows = []
    ydl_opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist"}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for _, row in sample.iterrows():
            artists = ast.literal_eval(row["artists"]) if isinstance(row["artists"], str) else []
            print(f"searching: {row['name']} - {artists}")
            match = _find_best_match(ydl, row["name"], artists, row["duration_ms"])
            if match is None:
                print("  skipped (no confident match)")
                continue

            video_url = match["url"] if str(match.get("url", "")).startswith("http") else f"https://www.youtube.com/watch?v={match['id']}"
            audio_path = output_dir / f"{row['id']}.mp3"
            download_opts = {
                "quiet": True,
                "no_warnings": True,
                "format": "bestaudio",
                "outtmpl": str(audio_path.with_suffix("")),
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "mp3"}],
            }
            try:
                with yt_dlp.YoutubeDL(download_opts) as download_ydl:
                    download_ydl.download([video_url])
            except yt_dlp.utils.DownloadError as exc:
                print(f"  download failed: {exc}")
                continue

            print(f"  matched: {match.get('title')}")
            manifest_rows.append({"id": row["id"], "name": row["name"], "artists": row["artists"], **{c: row[c] for c in _TARGET_COLUMNS}})

    new_manifest = pd.DataFrame(manifest_rows)
    combined = pd.concat([existing_manifest, new_manifest], ignore_index=True) if existing_manifest is not None else new_manifest
    combined.to_csv(manifest_path, index=False)
    print(f"\ndownloaded {len(manifest_rows)}/{len(sample)} new songs, {len(combined)} total in manifest at {manifest_path}")


if __name__ == "__main__":
    main()
