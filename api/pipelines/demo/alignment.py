"""Lyric/audio emotional alignment gap for a new song.

A deliberate near-duplicate of api/pipelines/training/alignment.py — the two
pipelines must never import from each other (see feature_extraction.md).
The formula is identical; only the source of `valence` differs: Pipeline 1
reads it straight from the Spotify dataset, Pipeline 2 uses Essentia's
demo-estimated valence (already rescaled to [0, 1] to match Spotify's
scale — see essentia_features.py) since there's no real Spotify value for
a song that was never scored by Spotify.
"""


def normalize_valence(valence: float) -> float:
    """Rescale a [0, 1] valence to [-1, 1] so it's on the same scale as the
    VADER compound sentiment score.
    """
    return (2 * valence) - 1


def compute_alignment_gap(lyric_sentiment: float, valence: float) -> float:
    """alignment_gap = lyric_sentiment - valence_normalized.

    Positive gap: the lyrics read more positive than the music sounds.
    Negative gap: the music sounds more positive than the lyrics read.
    (NOT the reverse — valence is subtracted from lyric sentiment, not the
    other way around.)
    """
    return lyric_sentiment - normalize_valence(valence)
