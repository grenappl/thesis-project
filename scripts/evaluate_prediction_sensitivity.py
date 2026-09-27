"""Is Pipeline 2 accurate enough for the popularity model?

For every validation song, scores the deployed popularity model twice: once
on Spotify's real audio features (manifest.csv) and once on Pipeline 2's
estimates (the per-song CSV from validate_pipeline_accuracy.py). Lyric
sentiment is the same real VADER score in both runs, so any difference comes
from audio-feature estimation alone. Then swaps in one estimated feature at a
time to show which feature's error moves the prediction most.

Runs on Windows (no WSL2 needed — only the model, pandas and VADER):

    uv run python scripts/evaluate_prediction_sensitivity.py "C:/Users/User/Downloads/songs(1).csv"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root, for `api`

from api.core.config import get_settings
from api.pipelines.demo.librosa_features import _MAJOR_KEY_PROFILE, _MINOR_KEY_PROFILE
from api.pipelines.demo.lyric_sentiment import compute_lyric_sentiment
from api.schemas.prediction import PredictionRequest
from api.services.popularity_prediction_service import PopularityPredictionService

AUDIO = ["tempo", "loudness", "key", "mode", "energy", "danceability", "speechiness",
         "acousticness", "instrumentalness", "liveness", "duration_ms", "valence"]


def key_mode_from_chroma(chroma_mean: np.ndarray) -> tuple[int, int]:
    """Same Temperley profile match as librosa_features.estimate_key."""
    best = max(
        (np.corrcoef(chroma_mean, np.roll(profile, tonic))[0, 1], tonic, mode)
        for mode, profile in ((1, _MAJOR_KEY_PROFILE), (0, _MINOR_KEY_PROFILE))
        for tonic in range(12)
    )
    return best[1], best[2]


def load_estimates(validation_dir: Path) -> pd.DataFrame:
    per_song = pd.read_csv(validation_dir / "per_song_results.csv").set_index("id")
    est = per_song[[c for c in per_song.columns if c.endswith("_predicted")]]
    est.columns = [c.removesuffix("_predicted") for c in est.columns]
    if not {"key", "mode"} <= set(est.columns):
        # older per-song CSVs have no key/mode: recompute from the cached chroma
        cache = np.load(validation_dir / "bass_chroma_cache.npz", allow_pickle=True)
        km = {sid: key_mode_from_chroma(c) for sid, c in zip(cache["id"], cache["chroma"])}
        est = est.loc[est.index.intersection(list(km))].copy()
        est["key"] = [km[i][0] for i in est.index]
        est["mode"] = [km[i][1] for i in est.index]
    return est


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("songs_csv", help="Raw Spotify dataset CSV (songs(1).csv), for lyrics and real popularity")
    parser.add_argument("--validation-dir", default="data/validation_songs")
    parser.add_argument("--out-csv", default="data/validation_songs/prediction_sensitivity.csv")
    args = parser.parse_args()
    validation_dir = Path(args.validation_dir)

    real = pd.read_csv(validation_dir / "manifest.csv").set_index("id")
    est = load_estimates(validation_dir)
    ids = real.index.intersection(est.index)

    wanted = set(ids)
    chunks = pd.read_csv(args.songs_csv, usecols=["id", "lyrics", "popularity"], chunksize=50_000)
    extra = pd.concat(c[c["id"].isin(wanted)] for c in chunks).drop_duplicates("id").set_index("id")
    ids = ids.intersection(extra.index)
    sentiment = {i: compute_lyric_sentiment(extra.at[i, "lyrics"]) for i in ids}

    service = PopularityPredictionService(get_settings())

    def predict(row: pd.Series, sid: str) -> float:
        fields = {f: row[f] for f in AUDIO}
        fields["key"], fields["mode"] = int(fields["key"]), int(fields["mode"])
        return service.predict(PredictionRequest(**fields, lyric_sentiment=sentiment[sid])).predicted_popularity

    rows = []
    for sid in ids:
        r, e = real.loc[sid], est.loc[sid]
        out = {"id": sid, "name": real.at[sid, "name"], "popularity": extra.at[sid, "popularity"],
               "pred_real": predict(r, sid), "pred_estimated": predict(e, sid)}
        for f in AUDIO:  # real features, with just this one swapped for its estimate
            out[f"swap_{f}"] = predict(r.copy().where(r.index != f, e[f]), sid)
        rows.append(out)
    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    shift = df["pred_estimated"] - df["pred_real"]
    rmse = lambda a, b: float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"songs: {len(df)}")
    print(f"prediction shift from using estimated features: mean |shift| {shift.abs().mean():.2f} pts, "
          f"median {shift.abs().median():.2f}, max {shift.abs().max():.2f}, mean signed {shift.mean():+.2f}")
    print(f"corr(pred on real features, pred on estimates): {np.corrcoef(df['pred_real'], df['pred_estimated'])[0, 1]:.3f}")
    print(f"RMSE vs actual popularity: real features {rmse(df['pred_real'], df['popularity']):.2f}, "
          f"estimated features {rmse(df['pred_estimated'], df['popularity']):.2f}")
    print("\nper-feature: mean |shift| when only that feature is estimated")
    for f in sorted(AUDIO, key=lambda f: -(df[f"swap_{f}"] - df["pred_real"]).abs().mean()):
        print(f"  {f:17}{(df[f'swap_{f}'] - df['pred_real']).abs().mean():6.2f}")
    print(f"\nper-song results written to {args.out_csv}")


if __name__ == "__main__":
    main()
