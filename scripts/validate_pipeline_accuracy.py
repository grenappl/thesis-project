"""Runs Pipeline 2 against every song in data/validation_songs/manifest.csv
(produced by download_validation_songs.py) and compares its output to the
real Spotify values, printing per-feature MAE and correlation across the
batch — a less noisy read than the single manually-sourced Glue Song test.

Must run under WSL2 (Pipeline 2 needs Essentia):

    uv run python scripts/validate_pipeline_accuracy.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from api.pipelines.demo.pipeline import extract_demo_features

_COMPARABLE_FEATURES = [
    "danceability", "loudness", "valence", "acousticness", "instrumentalness",
    "energy", "speechiness", "liveness", "tempo", "duration_ms",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-dir", default="data/validation_songs")
    parser.add_argument("--panns-checkpoint", default="models/panns/Cnn14_mAP=0.431.pth")
    parser.add_argument("--vggish-ridge-models-dir", default="models/vggish_ridge")
    parser.add_argument("--per-song-csv", default=None, help="Optional path to dump per-song real vs predicted values")
    args = parser.parse_args()

    validation_dir = Path(args.validation_dir)
    manifest = pd.read_csv(validation_dir / "manifest.csv")

    predicted_rows = []
    for _, row in manifest.iterrows():
        audio_path = validation_dir / f"{row['id']}.mp3"
        if not audio_path.exists():
            print(f"skipping {row['id']} — audio file missing")
            continue

        print(f"extracting: {row['name']}")
        try:
            features = extract_demo_features(
                audio_path,
                panns_checkpoint=args.panns_checkpoint,
                vggish_ridge_models_dir=args.vggish_ridge_models_dir,
            )
        except Exception as exc:  # keep going — one bad file shouldn't kill the whole batch
            print(f"  failed: {exc}")
            continue
        predicted_rows.append({"id": row["id"], **features})

    predicted = pd.DataFrame(predicted_rows)
    merged = manifest.merge(predicted, on="id", suffixes=("_real", "_predicted"))
    print(f"\ncompared {len(merged)}/{len(manifest)} songs\n")

    for feature in _COMPARABLE_FEATURES:
        real = merged[f"{feature}_real"].to_numpy(dtype=float)
        predicted_col = merged[f"{feature}_predicted"].to_numpy(dtype=float)
        mae = np.mean(np.abs(real - predicted_col))
        corr = np.corrcoef(real, predicted_col)[0, 1] if len(real) > 1 else float("nan")
        print(f"{feature}: MAE={mae:.4f} corr={corr:.3f}")

    if args.per_song_csv:
        columns = ["id", "name"] + [f"{f}_real" for f in _COMPARABLE_FEATURES] + [f"{f}_predicted" for f in _COMPARABLE_FEATURES]
        merged[columns].to_csv(args.per_song_csv, index=False)
        print(f"\nper-song values written to {args.per_song_csv}")


if __name__ == "__main__":
    main()
