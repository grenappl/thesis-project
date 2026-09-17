import numpy as np
import pytest
import soundfile as sf

SR = 22050


def _write_wav(path, y, sr=SR):
    sf.write(str(path), y, sr)
    return path


@pytest.fixture
def tone_wav(tmp_path):
    """A clean, well-behaved 2-second 440Hz tone with a steady 4Hz tremolo."""
    duration = 2.0
    t = np.linspace(0, duration, int(SR * duration), endpoint=False)
    modulation = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)
    y = 0.5 * np.sin(2 * np.pi * 440.0 * t) * modulation
    return _write_wav(tmp_path / "tone.wav", y.astype(np.float32))


@pytest.fixture
def silent_wav(tmp_path):
    """Two seconds of true digital silence."""
    y = np.zeros(int(SR * 2.0), dtype=np.float32)
    return _write_wav(tmp_path / "silence.wav", y)


@pytest.fixture
def short_wav(tmp_path):
    """A 0.1s clip — below the pipeline's minimum-duration threshold."""
    y = np.zeros(int(SR * 0.1), dtype=np.float32)
    return _write_wav(tmp_path / "short.wav", y)


@pytest.fixture
def empty_wav(tmp_path):
    """A zero-sample clip."""
    y = np.zeros(0, dtype=np.float32)
    return _write_wav(tmp_path / "empty.wav", y)


@pytest.fixture
def unsupported_file(tmp_path):
    """A file with an audio-like extension that is not actually audio."""
    path = tmp_path / "not_audio.mp3"
    path.write_bytes(b"this is definitely not an mp3 file, just plain text bytes")
    return path


@pytest.fixture
def missing_file(tmp_path):
    return tmp_path / "does_not_exist.wav"
