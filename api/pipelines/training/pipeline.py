"""Pipeline 1 — training-side feature extraction.

Reads directly from the Spotify dataset (`songs(1).csv` or equivalent).
No audio processing happens here: the twelve audio descriptors are dataset
columns already computed by Spotify, and the only derived features are
lyric sentiment and the alignment gap. Runs natively on Windows.

Do not import anything from `api.pipelines.demo` here — the two pipelines
must stay independent (see feature_extraction.md).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from api.pipelines.training.alignment import compute_alignment_gap, normalize_valence
from api.pipelines.training.sentiment import compute_lyric_sentiment

# Read directly from dataset columns — no computation needed.
AUDIO_DESCRIPTOR_COLUMNS = [
    "tempo",
    "loudness",
    "key",
    "mode",
    "energy",
    "danceability",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "duration_ms",
    "valence",
]

REQUIRED_COLUMNS = [*AUDIO_DESCRIPTOR_COLUMNS, "lyrics"]


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the dataset CSV and check it has the columns this pipeline needs."""
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset at '{path}' is missing required columns: {missing}")
    return df


def extract_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lyric_sentiment, valence_normalized, and alignment_gap columns.

    The twelve audio descriptor columns are left untouched (already present
    in the dataset). Returns a new DataFrame; does not mutate the input.
    """
    result = df.copy()
    result["lyric_sentiment"] = result["lyrics"].apply(compute_lyric_sentiment)
    result["valence_normalized"] = normalize_valence(result["valence"])
    result["alignment_gap"] = compute_alignment_gap(
        result["lyric_sentiment"], result["valence"]
    )
    return result


def run(input_path: str | Path, output_path: str | Path) -> pd.DataFrame:
    """Load the dataset, extract features, and write the result to CSV."""
    df = load_dataset(input_path)
    result = extract_training_features(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Pipeline 1: extract training-side features from the Spotify dataset."
    )
    parser.add_argument("input_csv", help="Path to the raw dataset CSV (e.g. songs(1).csv)")
    parser.add_argument("output_csv", help="Path to write the feature-augmented CSV to")
    args = parser.parse_args()

    extracted = run(args.input_csv, args.output_csv)
    print(f"Wrote {len(extracted)} rows with extracted features to {args.output_csv}")
