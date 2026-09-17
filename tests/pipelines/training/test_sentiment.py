import math

import pandas as pd
import pytest

from api.pipelines.training.sentiment import clean_lyrics_for_sentiment, compute_lyric_sentiment


def test_positive_lyrics_score_positive():
    score = compute_lyric_sentiment("I am so happy, everything is wonderful and great")
    assert score > 0


def test_negative_lyrics_score_negative():
    score = compute_lyric_sentiment("I hate this, everything is terrible and sad and awful")
    assert score < 0


def test_score_is_bounded():
    score = compute_lyric_sentiment("happy " * 200)
    assert -1.0 <= score <= 1.0


@pytest.mark.parametrize(
    "missing_value",
    [
        "",
        "   ",
        None,
        float("nan"),
        pd.NA,
        123,
    ],
)
def test_missing_or_empty_lyrics_are_neutral(missing_value):
    assert compute_lyric_sentiment(missing_value) == 0.0


def test_nan_from_pandas_series_is_handled():
    series = pd.Series(["good song", None, "", "bad awful terrible"])
    scores = series.apply(compute_lyric_sentiment)
    assert scores.iloc[1] == 0.0
    assert scores.iloc[2] == 0.0
    assert not any(math.isnan(s) for s in scores)


def test_clean_strips_bracketed_section_tags():
    cleaned = clean_lyrics_for_sentiment("[Chorus]\nThis is the actual lyric line.")
    assert "[Chorus]" not in cleaned
    assert "This is the actual lyric line." in cleaned


def test_clean_strips_lrc_timestamps():
    cleaned = clean_lyrics_for_sentiment("00:12.34 This is the actual lyric line.")
    assert "00:12.34" not in cleaned
    assert "This is the actual lyric line." in cleaned


def test_clean_normalizes_literal_escape_sequences():
    # Raw dataset rows contain a literal two-character \n, not a real
    # newline — verified directly against songs(1).csv.
    raw = "First line." + chr(92) + "nSecond line."
    cleaned = clean_lyrics_for_sentiment(raw)
    assert chr(92) + "n" not in cleaned
    assert "First line. Second line." == cleaned


def test_clean_preserves_capitalization_and_punctuation():
    # VADER uses ALL-CAPS and "!" as real intensity signals — cleaning must
    # not strip either.
    cleaned = clean_lyrics_for_sentiment("I am SO happy!!!")
    assert cleaned == "I am SO happy!!!"


def test_clean_only_metadata_yields_empty_string():
    assert clean_lyrics_for_sentiment("[Instrumental]") == ""


def test_lyrics_that_are_only_metadata_score_neutral():
    assert compute_lyric_sentiment("[Instrumental]") == 0.0


def test_bracket_noise_does_not_change_score_from_cleaned_equivalent():
    noisy = "[Chorus]\n00:01.00\nI am so happy, everything is wonderful and great"
    clean = "I am so happy, everything is wonderful and great"
    assert compute_lyric_sentiment(noisy) == compute_lyric_sentiment(clean)


def test_matches_notebook_reference_value_on_real_dataset_lyrics():
    # "Glue Song" from songs(1).csv: notebooks/3_sentiment_analysis.ipynb
    # computed lyric_sentiment=-0.1372 for this track's lyrics.
    lyrics = (
        "I've never known someone like you, ooh\nTangled in love, stuck by you\n"
        "From the glue\nDon't forget to kiss me\nOr else you'll have to miss me\n"
        "I guess I'm stuck forever by the glue\nOh, and you\n\n"
        "Finding the right words to use for this song\n"
        "I have you in mind, so it won't take so long\nNever thought I'd find you\n"
        "But you're here, and so I love you\n\nI'm not lying\nWhen I say I've been stuck\n"
        "By the glue onto you\nI've been stuck by glue\nRight onto you\n"
        "I've been stuck by glue\n\nI've never known\n\nI've never known someone like you\n"
        "I've never known\n\nI've never known someone like you, ooh"
    )
    assert compute_lyric_sentiment(lyrics) == pytest.approx(-0.1372, abs=1e-4)
