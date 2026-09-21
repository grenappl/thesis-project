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


def artist_graph(artist_list_series):
    """Artist co-appearance graph: one node per artist, an edge between every
    pair of artists credited on the same track, weighted by the number of
    tracks they share."""
    import networkx as nx

    graph = nx.Graph()
    for lst in artist_list_series:
        graph.add_nodes_from(lst)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                a, b = lst[i], lst[j]
                if graph.has_edge(a, b):
                    graph[a][b]['weight'] += 1
                else:
                    graph.add_edge(a, b, weight=1)
    return graph


def community_partition(df, fractions, resolution=3.0, pop_weight=3.0,
                        seed=RANDOM_SEED, n_pop_bands=5):
    """Artist-disjoint partition of `df` that satisfies Section 4.9.1 literally.

    1. Detect collaboration communities in the artist graph (Louvain; Blondel
       et al., 2008), so artists who collaborate tend to share a community.
    2. Assign whole communities to partitions greedily, largest first, choosing
       for each the partition whose targets it overshoots least -- balancing
       total size, genre mix, and popularity-quintile mix at once.
    3. Any track whose credited artists landed in different partitions is
       dropped. What remains has pairwise-disjoint artist sets under every
       credit, not just the primary one.

    `df` needs `artist_list` (parsed), `genre` and `popularity` columns.
    Returns (labels, community, info): `labels` holds a partition name or
    'dropped_cross_partition_collaboration' per row; `community` is the
    primary artist's community id.
    """
    import networkx as nx

    graph = artist_graph(df['artist_list'])
    comms = nx.community.louvain_communities(
        graph, weight='weight', resolution=resolution, seed=seed)
    comm_of = {a: ci for ci, members in enumerate(comms) for a in members}

    track_comms = df['artist_list'].apply(lambda lst: [comm_of[a] for a in lst])
    primary_comm = track_comms.str[0].to_numpy()
    n_comms = len(comms)

    genres = sorted(df['genre'].unique())
    genre_idx = df['genre'].map({g: i for i, g in enumerate(genres)}).to_numpy()
    bands = pd.qcut(df['popularity'], n_pop_bands, labels=False, duplicates='drop').to_numpy()
    n_bands = int(bands.max()) + 1

    size = np.bincount(primary_comm, minlength=n_comms).astype(float)
    genre_vec = np.zeros((n_comms, len(genres)))
    np.add.at(genre_vec, (primary_comm, genre_idx), 1)
    band_vec = np.zeros((n_comms, n_bands))
    np.add.at(band_vec, (primary_comm, bands), 1)

    n = len(df)
    genre_mix = genre_vec.sum(axis=0) / n
    band_mix = band_vec.sum(axis=0) / n
    parts = list(fractions)
    target = {p: fractions[p] * n for p in parts}
    target_g = {p: fractions[p] * n * genre_mix for p in parts}
    target_b = {p: fractions[p] * n * band_mix for p in parts}
    cur = {p: 0.0 for p in parts}
    cur_g = {p: np.zeros(len(genres)) for p in parts}
    cur_b = {p: np.zeros(n_bands) for p in parts}

    assign = {}
    for c in np.argsort(-size, kind='stable'):
        if size[c] == 0:
            continue  # community holds only guest artists; placed via its tracks' primaries
        best = None
        for p in parts:
            over_total = max(0.0, cur[p] + size[c] - target[p]) / target[p]
            over_g = np.maximum(0, cur_g[p] + genre_vec[c] - target_g[p]) / np.maximum(target_g[p], 1)
            over_b = np.maximum(0, cur_b[p] + band_vec[c] - target_b[p]) / np.maximum(target_b[p], 1)
            room = -(target[p] - cur[p]) / n  # tie-break toward the emptiest partition
            cost = over_total ** 2 + (over_g ** 2).sum() + pop_weight * (over_b ** 2).sum() + 1e-3 * room
            if best is None or cost < best[0]:
                best = (cost, p)
        p = best[1]
        assign[c] = p
        cur[p] += size[c]
        cur_g[p] += genre_vec[c]
        cur_b[p] += band_vec[c]

    # A community with no primary-credited tracks has no size and was skipped
    # above; its artists only ever appear as guests. Place it with the
    # partition of the first track that credits it, so it never forces a drop
    # on its own.
    for lst in track_comms:
        home = next((assign[c] for c in lst if c in assign), None)
        for c in lst:
            if c not in assign and home is not None:
                assign[c] = home

    track_parts = track_comms.apply(lambda lst: {assign[c] for c in lst})
    single = track_parts.str.len() == 1
    labels = pd.Series(
        np.where(single, track_parts.apply(lambda s: next(iter(s))),
                 'dropped_cross_partition_collaboration'),
        index=df.index)

    info = {
        'method': 'Louvain community detection + greedy balanced community assignment',
        'resolution': resolution,
        'pop_weight': pop_weight,
        'n_pop_bands': n_bands,
        'seed': seed,
        'fractions': fractions,
        'artists': graph.number_of_nodes(),
        'collaboration_edges': graph.number_of_edges(),
        'communities': n_comms,
        'largest_community_pct': float(100 * size.max() / n),
        'dropped_tracks': int((~single).sum()),
        'dropped_pct': float(100 * (~single).mean()),
    }
    return labels, pd.Series(primary_comm, index=df.index), info


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
