import pytest

from api.pipelines.demo.alignment import compute_alignment_gap, normalize_valence


@pytest.mark.parametrize(
    "valence, expected",
    [(0.0, -1.0), (0.5, 0.0), (1.0, 1.0), (0.582, 0.164)],
)
def test_normalize_valence_rescales_to_minus_one_one(valence, expected):
    assert normalize_valence(valence) == pytest.approx(expected)


def test_alignment_gap_matches_known_reference_value():
    # Same formula/reference as Pipeline 1 — see
    # tests/pipelines/training/test_alignment.py.
    gap = compute_alignment_gap(lyric_sentiment=-0.1372, valence=0.582)
    assert gap == pytest.approx(-0.3012, abs=1e-4)


def test_positive_gap_means_lyrics_more_positive_than_music():
    assert compute_alignment_gap(lyric_sentiment=0.8, valence=0.1) > 0


def test_negative_gap_means_music_more_positive_than_lyrics():
    assert compute_alignment_gap(lyric_sentiment=-0.8, valence=0.9) < 0
