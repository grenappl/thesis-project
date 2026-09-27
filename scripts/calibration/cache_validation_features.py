"""One-time cache of per-song intermediate representations for the 66
validation songs, so calibration experiments don't each need a ~70-minute
full pipeline run. Runs under WSL2 (PANNs needs wget; VGGish needs TF).

Saves data/validation_songs/feature_cache.npz with, per song (aligned to
manifest order):
  vggish      (128,)  mean-pooled TF-Hub VGGish embedding — same as pipeline
  panns_emb   (2048,) PANNs CNN14 clip embedding
  panns_prob  (527,)  PANNs CNN14 clipwise AudioSet probabilities
  chroma_cqt / chroma_cqt_harm / chroma_cens  (12,) time-averaged chroma
  librosa_tempo       current pipeline tempo (std_bpm=1.5)
"""

import sys
import time


import librosa
import numpy as np
import pandas as pd
from librosa.feature.rhythm import tempo as librosa_tempo

from api.pipelines.demo.audio_io import load_audio
from api.pipelines.demo.vggish_tfhub import compute_vggish_embedding
from panns_inference import AudioTagging

import os

OUT = "data/validation_songs/feature_cache.npz"
KEYS = ["id", "vggish", "panns_emb", "panns_prob", "chroma_cqt",
        "chroma_cqt_harm", "chroma_cens", "librosa_tempo"]

manifest = pd.read_csv("data/validation_songs/manifest.csv")
panns = AudioTagging(checkpoint_path="models/panns/Cnn14_mAP=0.431.pth")

# resumable: a killed session shouldn't throw away finished songs
if os.path.exists(OUT):
    prev = np.load(OUT, allow_pickle=True)
    out = {k: list(prev[k]) for k in KEYS}
else:
    out = {k: [] for k in KEYS}
done = set(out["id"])


def save():
    np.savez(OUT, **{k: np.array(v) for k, v in out.items()})


for i, row in manifest.iterrows():
    if row["id"] in done:
        continue
    t0 = time.time()
    path = f"data/validation_songs/{row['id']}.mp3"
    y, sr = load_audio(path)

    y32 = librosa.resample(y, orig_sr=sr, target_sr=32000)
    prob, emb = panns.inference(y32[None, :])

    y_harm = librosa.effects.harmonic(y)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)

    out["id"].append(row["id"])
    out["vggish"].append(compute_vggish_embedding(path))
    out["panns_emb"].append(emb[0])
    out["panns_prob"].append(prob[0])
    out["chroma_cqt"].append(librosa.feature.chroma_cqt(y=y, sr=sr).mean(axis=1))
    out["chroma_cqt_harm"].append(librosa.feature.chroma_cqt(y=y_harm, sr=sr).mean(axis=1))
    out["chroma_cens"].append(librosa.feature.chroma_cens(y=y_harm, sr=sr).mean(axis=1))
    out["librosa_tempo"].append(float(librosa_tempo(onset_envelope=onset_env, sr=sr, std_bpm=1.5)[0]))
    print(f"[{i + 1}/{len(manifest)}] {row['name'][:40]} ({time.time() - t0:.1f}s)", flush=True)
    if len(out["id"]) % 5 == 0:
        save()

save()
print(f"saved {OUT} ({len(out['id'])} songs)")
