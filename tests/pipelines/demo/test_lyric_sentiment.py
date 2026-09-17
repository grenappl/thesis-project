import pytest

from api.pipelines.demo.lyric_sentiment import clean_lyrics_for_sentiment, compute_lyric_sentiment


def test_positive_lyrics_score_positive():
    assert compute_lyric_sentiment("I am so happy, everything is wonderful and great") > 0


def test_negative_lyrics_score_negative():
    assert compute_lyric_sentiment("I hate this, everything is terrible and sad and awful") < 0


def test_score_is_bounded():
    assert -1.0 <= compute_lyric_sentiment("happy " * 200) <= 1.0


@pytest.mark.parametrize("missing_value", ["", "   ", None, 123])
def test_missing_or_empty_lyrics_are_neutral(missing_value):
    assert compute_lyric_sentiment(missing_value) == 0.0


def test_clean_strips_bracketed_section_tags():
    cleaned = clean_lyrics_for_sentiment("[Chorus]\nThis is the actual lyric line.")
    assert "[Chorus]" not in cleaned
    assert "This is the actual lyric line." in cleaned


def test_clean_strips_lrc_timestamps():
    cleaned = clean_lyrics_for_sentiment("00:12.34 This is the actual lyric line.")
    assert "00:12.34" not in cleaned


def test_clean_preserves_capitalization_and_punctuation():
    assert clean_lyrics_for_sentiment("I am SO happy!!!") == "I am SO happy!!!"


def test_lyrics_that_are_only_metadata_score_neutral():
    assert compute_lyric_sentiment("[Instrumental]") == 0.0
