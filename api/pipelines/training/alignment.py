"""Lyric/audio emotional alignment gap.

This is the central formula of the thesis. The order of operations and the
sign of the subtraction are load-bearing: getting either wrong silently
flips the meaning of every downstream "alignment" feature.
"""


def normalize_valence(valence: float) -> float:
    """Rescale Spotify's [0, 1] valence to [-1, 1] so it's on the same scale
    as the VADER compound sentiment score.
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
