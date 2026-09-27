"""Cache full + bass chroma (per-song mean, 12-d) for the validation songs,
for the relative-key tiebreak experiment. Same load path as the pipeline
(librosa, 22050 Hz mono).

    uv run python scripts/calibration/bass_chroma_cache.py
"""
from pathlib import Path

import librosa
import numpy as np
import pandas as pd

d = Path("data/validation_songs")
man = pd.read_csv(d / "manifest.csv")
ids, full, bass, bass_frames_weighted = [], [], [], []
for i, sid in enumerate(man["id"]):
    p = d / f"{sid}.mp3"
    if not p.exists():
        continue
    y, sr = librosa.load(p, sr=22050, mono=True)
    full.append(librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1))
    # C1 (32.7 Hz) up 3 octaves to ~C4 (262 Hz): the bass register
    b = librosa.feature.chroma_cqt(y=y, sr=sr, fmin=librosa.note_to_hz("C1"), n_octaves=3)
    bass.append(b.mean(axis=1))
    ids.append(sid)
    print(i, sid, flush=True)
np.savez(d / "bass_chroma_cache.npz", id=np.array(ids), chroma=np.array(full), bass=np.array(bass))
print("done", len(ids))
