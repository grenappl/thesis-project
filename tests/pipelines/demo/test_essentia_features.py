"""Essentia only installs on Linux (see pyproject.toml's sys_platform marker),
so on Windows these functions can never succeed — what's tested here is that
they fail with a clear, actionable RuntimeError instead of an unhandled
ImportError/stack trace. The real extraction logic is exercised under WSL2
(see feature_extraction.md); this file just guards that the fallback holds
on whichever platform tests happen to run on.
"""

import importlib.util

import pytest

from api.pipelines.demo.essentia_features import extract_loudness

_ESSENTIA_INSTALLED = importlib.util.find_spec("essentia") is not None


@pytest.fixture
def fake_audio(tmp_path):
    path = tmp_path / "fake.wav"
    path.write_bytes(b"irrelevant")
    return path


@pytest.mark.skipif(_ESSENTIA_INSTALLED, reason="essentia is installed; guard path isn't exercised")
def test_extract_loudness_without_essentia_raises_clear_error(fake_audio):
    with pytest.raises(RuntimeError, match="WSL2"):
        extract_loudness(fake_audio)
