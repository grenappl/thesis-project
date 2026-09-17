"""Lyric sentiment scoring via VADER, for a new song's user-submitted lyrics.

A deliberate near-duplicate of api/pipelines/training/sentiment.py — the two
pipelines must never import from each other (see feature_extraction.md), so
this logic is copied rather than shared. Runs natively (no Essentia/WSL2
dependency), same as the rest of the Librosa/scipy-based demo extraction.
"""

import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

_UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9a-fA-F]{4})")
_BRACKETED_SECTION_RE = re.compile(r"\[.*?\]")
_LRC_TIMESTAMP_RE = re.compile(r"\d{1,2}:\d{2}(\.\d+)?")
_WHITESPACE_RE = re.compile(r"\s+")


def clean_lyrics_for_sentiment(lyrics: str) -> str:
    """Strip structural noise from lyrics before VADER scoring.

    Removes escape-sequence artifacts, bracketed section tags (e.g.
    `[Chorus]`), and LRC-style timestamps a user might paste in from a
    lyrics site — but deliberately does NOT lowercase or touch punctuation,
    since VADER uses capitalization (ALL CAPS = emphasis) and punctuation
    (e.g. "!") as real sentiment-intensity signals.
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

    Missing/blank lyrics (including lyrics that are blank only *after*
    cleaning, e.g. just "[Instrumental]") score neutral (0.0) rather than
    raising.
    """
    if not isinstance(lyrics, str) or not lyrics.strip():
        return 0.0
    cleaned = clean_lyrics_for_sentiment(lyrics)
    if not cleaned:
        return 0.0
    return _analyzer.polarity_scores(cleaned)["compound"]
