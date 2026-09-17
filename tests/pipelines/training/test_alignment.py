import pytest

from api.pipelines.training.alignment import compute_alignment_gap, normalize_valence


@pytest.mark.parametrize(
    "valence, expected",
    [
        (0.0, -1.0),
        (0.5, 0.0),
        (1.0, 1.0),
        (0.582, 0.164),
    ],
)
def test_normalize_valence_rescales_to_minus_one_one(valence, expected):
    assert normalize_valence(valence) == pytest.approx(expected)


def test_alignment_gap_matches_known_reference_value():
    # Reference case from notebooks/data_filtered/songs_2023.csv (Glue Song):
    # valence=0.582 -> valence_normalized=0.164, lyric_sentiment=-0.1372
    # -> alignment_gap=-0.3012
    gap = compute_alignment_gap(lyric_sentiment=-0.1372, valence=0.582)
    assert gap == pytest.approx(-0.3012, abs=1e-4)


def test_positive_gap_means_lyrics_more_positive_than_music():
    # Sad-sounding music (low valence) with happy lyrics -> positive gap.
    gap = compute_alignment_gap(lyric_sentiment=0.8, valence=0.1)
    assert gap > 0


def test_negative_gap_means_music_more_positive_than_lyrics():
    # Happy-sounding music (high valence) with sad lyrics -> negative gap.
    gap = compute_alignment_gap(lyric_sentiment=-0.8, valence=0.9)
    assert gap < 0


def test_gap_is_lyric_sentiment_minus_valence_not_the_reverse():
    lyric_sentiment, valence = 0.3, 0.2
    gap = compute_alignment_gap(lyric_sentiment, valence)
    reversed_gap = normalize_valence(valence) - lyric_sentiment
    assert gap == pytest.approx(lyric_sentiment - normalize_valence(valence))
    assert gap == pytest.approx(-reversed_gap)
