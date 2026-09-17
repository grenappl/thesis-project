"""Shared definitions for the modeling notebooks (4-9).

This is deliberately NOT a package -- it is one flat module holding only the
definitions that must be identical across every notebook.

Section 4.9.1 (feature-configuration isolation, configuration parity) is only
meaningful if every notebook builds its feature matrices from a single source
of truth. Copy-pasting the ladder definition into six notebooks is exactly how
that check silently stops being true. Everything else -- the actual analysis,
the plots, the narrative -- stays in the notebooks where it can be read and
edited interactively.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_GLOB = os.path.join(HERE, 'data_filtered', 'songs_*.csv')
ARTIFACTS = os.path.join(HERE, 'artifacts')

# Held fixed across every model in the ladder (Section 4.2.2).
RANDOM_SEED = 42

TARGET = 'popularity'

# Table 2 / Section 4.1.4 -- Spotify's twelve distributed audio descriptors.
AUDIO_12 = [
    'tempo', 'loudness', 'key', 'mode', 'energy', 'danceability',
    'speechiness', 'acousticness', 'instrumentalness', 'liveness',
    'duration_ms', 'valence',
]
KEY_COL = 'key'
LYRIC_COLS = ['lyric_sentiment']
ALIGNMENT_COLS = ['alignment_gap']

# Columns that must never reach a model.
#   year, genre                -- filtering and stratification only; NFR1
#                                 (Section 4.4.1) excludes recency from the
#                                 feature set
#   artist follower/popularity -- excluded for the leakage reason in 4.9.3(A)
#   valence_norm               -- the throwaway intermediate from Section
#                                 4.2.1. Handing it to Control would give that
#                                 model half of the alignment relationship the
#                                 ablation is supposed to withhold.
FORBIDDEN_COLS = [
    'year', 'genre', 'total_artist_followers', 'avg_artist_popularity',
    'artist_ids', 'artists', 'niche_genres', 'valence_norm', 'lyrics',
    'id', 'name', 'album_name', 'is_valid_lyrics', 'index',
]

# Table 5 -- the five-model comparison ladder.
LADDER = {
    'baseline': (),
    'audio_only': ('audio',),
    'lyric_only': ('lyric',),
    'control': ('audio', 'lyric'),
    'alignment_augmented': ('audio', 'lyric', 'alignment'),
}

# Trained under both algorithms; Baseline is algorithm-agnostic (Section 4.2.2).
ALGO_MODELS = ['audio_only', 'lyric_only', 'control', 'alignment_augmented']


def load_corpus(usecols=None):
    """Concatenate the 24 per-year filtered CSVs into one frame."""
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(f'no per-year CSVs matched {DATA_GLOB}')
    frames = [pd.read_csv(f, usecols=usecols, low_memory=False) for f in files]
    df = pd.concat(frames, ignore_index=True)
    # the per-year files carry leftover index columns from earlier notebooks
    df = df.loc[:, [c for c in df.columns if not str(c).startswith('Unnamed')]]
    return df


def one_hot_key(df):
    """Section 4.2.1: key is nominal over twelve pitch classes, so it is
    one-hot encoded rather than passed as an integer. Twelve dummies replace
    one integer column -- a net gain of eleven columns, as the section states.
    """
    cat = pd.Categorical(df[KEY_COL].astype(int), categories=list(range(12)))
    dummies = pd.get_dummies(cat, prefix='key').astype(np.int8)
    dummies.index = df.index
    return dummies


def build_feature_frame(df):
    """Return (X, blocks): every candidate input with key expanded, plus the
    block -> column-name map the ladder selects from."""
    audio_plain = [c for c in AUDIO_12 if c != KEY_COL]
    key_dummies = one_hot_key(df)
    X = pd.concat(
        [
            df[audio_plain].astype(float),
            key_dummies,
            df[LYRIC_COLS + ALIGNMENT_COLS].astype(float),
        ],
        axis=1,
    )
    blocks = {
        'audio': audio_plain + list(key_dummies.columns),
        'lyric': list(LYRIC_COLS),
        'alignment': list(ALIGNMENT_COLS),
    }
    return X, blocks


def columns_for(model_name, blocks):
    """The exact columns Table 5 specifies for one rung of the ladder."""
    cols = []
    for block in LADDER[model_name]:
        cols.extend(blocks[block])
    return cols


def assert_clean(X):
    """Guard against a forbidden column or a NaN reaching a model."""
    bad = [c for c in X.columns if c in FORBIDDEN_COLS]
    if bad:
        raise AssertionError(f'forbidden columns present in matrix: {bad}')
    if X.isna().to_numpy().any():
        raise AssertionError('NaNs present in feature matrix')


def regression_metrics(y_true, y_pred):
    """R2, RMSE, MAE (Section 3.5.1), computed directly so the numbers do not
    depend on which scikit-learn version is installed."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_true - y_pred
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        'r2': 1.0 - ss_res / ss_tot,
        'rmse': float(np.sqrt(np.mean(resid ** 2))),
        'mae': float(np.mean(np.abs(resid))),
    }


def artist_lists(series):
    """artist_ids is stored as a JSON array string; parse to a list."""
    return series.apply(json.loads)


def ensure_artifacts():
    os.makedirs(ARTIFACTS, exist_ok=True)
    return ARTIFACTS


def artifact(*parts):
    ensure_artifacts()
    return os.path.join(ARTIFACTS, *parts)


def save_json(obj, name):
    path = artifact(name)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(obj, fh, indent=2)
    return path


def load_json(name):
    with open(artifact(name), encoding='utf-8') as fh:
        return json.load(fh)
