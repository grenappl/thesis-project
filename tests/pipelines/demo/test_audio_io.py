import pytest

from api.pipelines.demo.audio_io import (
    AudioTooShortError,
    UnsupportedAudioError,
    load_audio,
)


def test_load_audio_decodes_valid_wav(tone_wav):
    y, sr = load_audio(tone_wav)
    assert y.size > 0
    assert sr > 0


def test_silent_audio_loads_without_crashing(silent_wav):
    y, sr = load_audio(silent_wav)
    assert y.size > 0
    assert (y == 0).all()


def test_short_clip_raises_too_short_error(short_wav):
    with pytest.raises(AudioTooShortError):
        load_audio(short_wav)


def test_empty_clip_raises_too_short_error(empty_wav):
    with pytest.raises(AudioTooShortError):
        load_audio(empty_wav)


def test_unsupported_format_raises_unsupported_error_not_crash(unsupported_file):
    with pytest.raises(UnsupportedAudioError):
        load_audio(unsupported_file)


def test_missing_file_raises_unsupported_error_not_crash(missing_file):
    with pytest.raises(UnsupportedAudioError):
        load_audio(missing_file)
