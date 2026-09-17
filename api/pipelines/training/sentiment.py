"""Lyric sentiment scoring via VADER (rule-based, no training data needed)."""

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_BRACKETED_SECTION_RE = re.compile(r"\[.*?\]")
_LRC_TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}(\.\d+)?")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_lyrics_for_sentiment(lyrics: str) -> str:
    """Strip structural noise from raw lyrics before VADER scoring.

    The raw dataset has literal escape-sequence artifacts in `lyrics`
    (e.g. a literal two-character `\\n` instead of a real newline — verified
    directly against songs(1).csv, not assumed) and bracketed section tags
    like `[Chorus]`/`[Verse 1]`. Left in, these inject tokens VADER has no
    real sentiment to assign, diluting the compound score with noise.

    Deliberately does NOT lowercase or touch punctuation — VADER uses
    capitalization (ALL CAPS = emphasis) and punctuation (e.g. "!") as real
    sentiment-intensity signals, so "cleaning" those would throw away
    information VADER is designed to use. This only removes text that was
    never part of the actual lyrics: escape-sequence artifacts, bracketed
    metadata, and LRC-style timestamps — mirroring the structural-noise
    steps in notebooks/2_lyrics_filtering.ipynb, minus the lowercasing that
    notebook does for language detection (a different task with different
    needs) and minus the validity-filtering that pipeline handles separately.
    """
    text = _UNICODE_ESCAPE_RE.sub(lambda m: chr(int(m.group(1), 16)), lyrics)
    text = text.replace("\\n", " ").replace("\\r", " ").replace("\\t", " ")
    text = text.replace('\\"', '"').replace("\\'", "'")
    text = text.replace("\\\\", "")
    text = _BRACKETED_SECTION_RE.sub(" ", text)
    text = _LRC_TIMESTAMP_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def compute_lyric_sentiment(lyrics: object) -> float:
    """VADER compound sentiment score, bounded [-1, 1].

    Missing lyrics (NaN, None, non-string) or lyrics that are blank/whitespace
    — including lyrics that are blank only *after* cleaning, e.g. a track
    whose whole "lyrics" field was just "[Instrumental]" — are treated as
    neutral (0.0) rather than raising, since the dataset has rows with no
    usable lyrics and the pipeline must not crash on them.
    """
    if not isinstance(lyrics, str) or not lyrics.strip():
        return 0.0
    cleaned = clean_lyrics_for_sentiment(lyrics)
    if not cleaned:
        return 0.0
    return _analyzer.polarity_scores(cleaned)["compound"]
