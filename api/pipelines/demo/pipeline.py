"""Pipeline 2 — demo-app-side feature extraction.

Extracts the same descriptor set as Pipeline 1, but from a raw audio file
for a new, user-submitted song that isn't in the training dataset. Two
inputs: the audio file (required) and the song's lyrics (optional — omit
for instrumental tracks).

- Librosa: tempo, spectral centroid, duration, key, mode (plain DSP).
- VGGish (TF-Hub, isolated subprocess) + PANNs CNN14 embeddings, fed to one
  gradient-boosted regression head per target trained on ~491k real
  Spotify tracks: valence, acousticness, danceability, energy, speechiness,
  instrumentalness, loudness, liveness. See regression_heads.py.
- VADER: lyric sentiment + the alignment gap against the estimated valence.

Essentia is no longer called at all (essentia_features.py is kept but
unused). This module still runs under WSL2 — see feature_extraction.md.
Run it with:

    uv run python -m api.pipelines.demo.pipeline <audio_path> [--lyrics-file <path>]

Do not import anything from `api.pipelines.training` here — the two
pipelines must stay independent (see feature_extraction.md).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from api.pipelines.demo.alignment import compute_alignment_gap
from api.pipelines.demo.audio_io import load_audio
from api.pipelines.demo.librosa_features import extract_librosa_features
from api.pipelines.demo.lyric_sentiment import compute_lyric_sentiment
from api.pipelines.demo.panns_embedding import extract_panns_embedding
from api.pipelines.demo.regression_heads import DEFAULT_HEADS_DIR, predict_features


def _log(message: str) -> None:
    """Stage marker for the live log stream. Printed to stderr, not stdout —
    stdout is reserved for the final JSON result (see
    DemoFeatureExtractionService._extract_json, which finds the JSON block
    by scanning stdout specifically)."""
    print(f"[pipeline] {message}", file=sys.stderr, flush=True)


def _run_vggish_subprocess(audio_path: str | Path) -> np.ndarray:
    """VGGish embedding from a fresh process — see
    api.pipelines.demo.vggish_tfhub's docstring for why this can't be an
    in-process import (Essentia's bundled TensorFlow crashes if a
    standalone tensorflow/tensorflow_hub is loaded in the same process).
    """
    cmd = [sys.executable, "-m", "api.pipelines.demo.vggish_tfhub", str(audio_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"vggish_tfhub subprocess failed:\n{result.stderr.strip()}")
    return np.array(json.loads(result.stdout.strip().splitlines()[-1]), dtype=np.float32)


def extract_demo_features(
    audio_path: str | Path,
    lyrics: str | None = None,
    panns_checkpoint: str | None = None,
    regression_heads_dir: str | Path = DEFAULT_HEADS_DIR,
) -> dict[str, float]:
    """Run the full demo-side extraction for one uploaded audio file, plus
    optional lyrics for the sentiment + alignment gap features.

    `lyrics` is optional (instrumental tracks have none) — when omitted,
    the result has every audio-derived feature but no `lyric_sentiment`/
    `alignment_gap`.
    """
    _log("loading audio")
    y, sr = load_audio(audio_path)

    _log("extracting Librosa features (tempo, spectral centroid, duration, key, mode)")
    features = extract_librosa_features(y, sr)

    _log("computing PANNs CNN14 embedding")
    panns_embedding = extract_panns_embedding(y, sr, checkpoint_path=panns_checkpoint)

    _log("computing TF-Hub VGGish embedding (isolated subprocess)")
    vggish_embedding = _run_vggish_subprocess(audio_path)

    _log("predicting valence/acousticness/danceability/energy/speechiness/instrumentalness/loudness/liveness")
    features.update(predict_features(vggish_embedding, panns_embedding, heads_dir=regression_heads_dir))

    if lyrics is not None:
        _log("computing lyric sentiment (VADER compound score)")
        lyric_sentiment = compute_lyric_sentiment(lyrics)
        features["lyric_sentiment"] = lyric_sentiment

        _log("computing alignment gap (lyric_sentiment - valence_normalized)")
        features["alignment_gap"] = compute_alignment_gap(lyric_sentiment, features["valence"])
    else:
        _log("no lyrics provided — skipping sentiment + alignment gap")

    return features


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline 2: extract demo-app features from a raw audio file.")
    parser.add_argument("audio_path", help="Path to the uploaded audio file")
    parser.add_argument("--lyrics-file", default=None, help="Path to a UTF-8 text file with the song's lyrics (optional)")
    parser.add_argument("--panns-checkpoint", default=None, help="Path to Cnn14_mAP=0.431.pth")
    parser.add_argument(
        "--regression-heads-dir",
        default=str(DEFAULT_HEADS_DIR),
        help="Directory with the fitted .joblib regression heads (default: models/regression_heads)",
    )
    args = parser.parse_args()

    lyrics_text = Path(args.lyrics_file).read_text(encoding="utf-8") if args.lyrics_file else None
    result = extract_demo_features(
        args.audio_path,
        lyrics=lyrics_text,
        panns_checkpoint=args.panns_checkpoint,
        regression_heads_dir=args.regression_heads_dir,
    )
    print(json.dumps(result, indent=2))
