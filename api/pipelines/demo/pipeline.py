"""Pipeline 2 — demo-app-side feature extraction.

Extracts the same descriptor set as Pipeline 1, but from a raw audio file
for a new, user-submitted song that isn't in the training dataset. Two
inputs: the audio file (required) and the song's lyrics (optional — omit
for instrumental tracks). Combines Librosa (tempo, spectral centroid,
duration, key, mode), PANNs CNN14 (liveness), TF-Hub VGGish + a regression
head per target (valence, acousticness, instrumentalness, danceability,
energy, speechiness, loudness — run as an isolated subprocess, see
vggish_tfhub.py — all twelve Spotify audio descriptors are now covered),
and VADER (lyric sentiment + the alignment gap against the VGGish valence
estimate). Essentia is no longer called at all — loudness (its last job)
moved to the VGGish path after real-song validation showed a large accuracy
win (corr 0.605 -> 0.796+); see feature_extraction.md's "Real-song
validation" section. `essentia_features.py` still exists but nothing in
this pipeline calls it anymore.

This whole module runs under WSL2 — see feature_extraction.md for why the
pipeline isn't split across Windows and WSL2 per-feature. Run it with:

    uv run python -m api.pipelines.demo.pipeline <audio_path> [--lyrics-file <path>]

Do not import anything from `api.pipelines.training` here — the two
pipelines must stay independent (see feature_extraction.md).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from api.pipelines.demo.alignment import compute_alignment_gap
from api.pipelines.demo.audio_io import load_audio
from api.pipelines.demo.liveness_panns import extract_liveness
from api.pipelines.demo.librosa_features import extract_librosa_features
from api.pipelines.demo.lyric_sentiment import compute_lyric_sentiment


def _log(message: str) -> None:
    """Stage marker for the live log stream. Printed to stderr, not stdout —
    stdout is reserved for the final JSON result (see
    DemoFeatureExtractionService._extract_json, which finds the JSON block
    by scanning stdout specifically)."""
    print(f"[pipeline] {message}", file=sys.stderr, flush=True)


def _run_vggish_subprocess(audio_path: str | Path, models_dir: str | None) -> dict[str, float]:
    """Runs valence/acousticness/instrumentalness/danceability/energy/
    speechiness/loudness in a fresh process — see
    api.pipelines.demo.vggish_tfhub's docstring for why this can't be an
    in-process import (Essentia's bundled TensorFlow crashes if a
    standalone tensorflow/tensorflow_hub is loaded in the same process).
    """
    cmd = [sys.executable, "-m", "api.pipelines.demo.vggish_tfhub", str(audio_path)]
    if models_dir is not None:
        cmd += ["--models-dir", models_dir]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"vggish_tfhub subprocess failed:\n{result.stderr.strip()}")
    return json.loads(result.stdout)


def extract_demo_features(
    audio_path: str | Path,
    lyrics: str | None = None,
    panns_checkpoint: str | None = None,
    vggish_ridge_models_dir: str | None = None,
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

    _log("extracting liveness (PANNs CNN14)")
    features["liveness"] = extract_liveness(y, sr, checkpoint_path=panns_checkpoint)

    _log(
        "extracting valence/acousticness/instrumentalness/danceability/energy/speechiness/loudness "
        "(TF-Hub VGGish + a regression head per target, isolated subprocess)"
    )
    features.update(_run_vggish_subprocess(audio_path, vggish_ridge_models_dir))

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
    import json

    parser = argparse.ArgumentParser(
        description="Pipeline 2: extract demo-app features from a raw audio file."
    )
    parser.add_argument("audio_path", help="Path to the uploaded audio file")
    parser.add_argument(
        "--lyrics-file", default=None, help="Path to a UTF-8 text file with the song's lyrics (optional)"
    )
    parser.add_argument("--panns-checkpoint", default=None, help="Path to Cnn14_mAP=0.431.pth")
    parser.add_argument(
        "--vggish-ridge-models-dir",
        default=None,
        help="Directory with the fitted .joblib regression models (default: models/vggish_ridge)",
    )
    args = parser.parse_args()

    lyrics_text = None
    if args.lyrics_file:
        lyrics_text = Path(args.lyrics_file).read_text(encoding="utf-8")

    result = extract_demo_features(
        args.audio_path,
        lyrics=lyrics_text,
        panns_checkpoint=args.panns_checkpoint,
        vggish_ridge_models_dir=args.vggish_ridge_models_dir,
    )
    print(json.dumps(result, indent=2))
