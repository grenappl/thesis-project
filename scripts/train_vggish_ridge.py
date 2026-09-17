"""Fits the regression heads used by api.pipelines.demo.vggish_tfhub.

Trains regressors on the Kaggle precomputed-embeddings dataset (Google's
official TF-Hub VGGish embeddings, same source vggish_tfhub.py uses at
inference time — see feature_extraction.md for why that match matters) for
every target in _TARGET_MODELS. Run once, from WSL2 (or Windows — this
script itself needs no Essentia):

    uv run python scripts/train_vggish_ridge.py \\
        data/kaggle_embeddings/vggish_embeddings.npz \\
        "/mnt/c/Users/User/Downloads/songs(1).csv"

Writes models/vggish_ridge/{target}.joblib for each target in
_TARGET_MODELS — the filename predates the GBM addition below (started as
Ridge-only for three targets); not renamed since it's referenced throughout
feature_extraction.md/CLAUDE.md/PIPELINE_SETUP.md and a rename isn't worth
that churn.

Model choice per target: all seven now use HistGradientBoostingRegressor,
not Ridge — GBM was tested head-to-head against Ridge for every target
tried in this project (not assumed to win, checked every time) and won on
every metric, every time: valence/acousticness/danceability/loudness first
(triggered by a Ridge-shrinkage problem seen in a real spot-check — L2
shrinkage pulls predictions toward the mean, so songs with an extreme true
value get systematically underestimated; acousticness's extreme-value-tail
MAE dropped from 0.108 to 0.073, the clearest case), then
instrumentalness/energy/speechiness in a follow-up round (corr 0.628→0.702,
0.891→0.897, 0.725→0.762). `loudness` itself moved onto VGGish here for the
first time in the first round — it had never been tried before, on the
assumption Essentia's `ReplayGain` (a real DSP measurement) didn't need it;
even plain Ridge (0.796 held-out corr) blew past Essentia's fitted-constant
approach (0.605 real-song corr). `_ridge` is kept as an available factory,
not deleted, in case a future target doesn't follow this pattern — nothing
here should be read as "GBM always wins," just "it has so far, on every
target actually tested."

`liveness` was tried against VGGish too (Ridge held-out corr=0.419, GBM
corr=0.472) and is the one target where this *didn't* help — both are well
below the current PANNs crowd-noise-heuristic's real-song correlation
(0.694). Not adopted; `liveness` stays a PANNs proxy, not a VGGish
regression — see `api.pipelines.demo.liveness_panns`.

`tempo` was tried too (held-out R^2=0.099, corr=0.315 — weak, as expected
since VGGish embeddings aren't built to capture rhythmic periodicity) and
deliberately left out of both _TARGET_MODELS and vggish_tfhub.py's
_TARGETS: end-to-end it scored a better raw correlation (0.416) than the
current Librosa estimate (0.182) only by compressing nearly every
prediction into a narrow BPM band, which is a worse fit for this pipeline's
actual use than a noisier, wider-ranging estimate. See
feature_extraction.md's "Tempo" section for the full investigation
(including why tempo's real problem is octave confusion, not just weak
periodicity detection).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_ALPHAS = np.logspace(-3, 3, 13)


def _ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=_ALPHAS))


def _gbm():
    return HistGradientBoostingRegressor(random_state=42)


_TARGET_MODELS = {
    "valence": _gbm,
    "acousticness": _gbm,
    "danceability": _gbm,
    "loudness": _gbm,
    "instrumentalness": _gbm,
    "energy": _gbm,
    "speechiness": _gbm,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_npz", help="Path to vggish_embeddings.npz (id, features)")
    parser.add_argument("songs_csv", help="Path to the raw Spotify dataset CSV (songs(1).csv)")
    parser.add_argument("--output-dir", default="models/vggish_ridge")
    args = parser.parse_args()

    targets = list(_TARGET_MODELS)

    print("loading embeddings...")
    embeddings = np.load(args.embeddings_npz, allow_pickle=True)
    embeddings_df = pd.DataFrame({"id": embeddings["id"]})
    features = embeddings["features"]

    print("loading targets...")
    songs = pd.read_csv(args.songs_csv, usecols=["id", *targets])

    merged = embeddings_df.reset_index().merge(songs, on="id", how="inner").dropna(subset=targets)
    print(f"matched {len(merged)}/{len(embeddings_df)} embeddings to rows with targets")

    X = features[merged["index"].to_numpy()]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for target, model_factory in _TARGET_MODELS.items():
        y = merged[target].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = model_factory()
        model.fit(X_train, y_train)

        r2 = model.score(X_test, y_test)
        corr = np.corrcoef(model.predict(X_test), y_test)[0, 1]
        print(f"{target} ({model_factory.__name__.strip('_')}): held-out R^2={r2:.3f} corr={corr:.3f}")

        joblib.dump(model, output_dir / f"{target}.joblib")

    print(f"saved models to {output_dir}/")


if __name__ == "__main__":
    main()
