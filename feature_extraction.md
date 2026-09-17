# Feature Extraction Pipeline

This document describes the feature set behind *Lyric and Audio Emotional
Alignment as a Predictor of Spotify Track Popularity* and the two pipelines
that produce it: `api/pipelines/training/` (Pipeline 1) and
`api/pipelines/demo/` (Pipeline 2). For environment setup and how to invoke
each pipeline, see [`PIPELINE_SETUP.md`](PIPELINE_SETUP.md).

## Why two pipelines

Model **training** works from `songs(1).csv` (491,632 rows), where every
audio descriptor Spotify ever computed is already a column — no audio
processing needed. The **demo app** lets a user submit a brand-new song that
was never scored by Spotify, so the same descriptors have to be estimated
from the raw audio file. A new song still has the same *two inputs* a
dataset track does — audio and lyrics — just neither is known to Spotify
ahead of time, so Pipeline 2 accepts lyrics text alongside the audio file,
same as Pipeline 1 reads a `lyrics` column. These are different problems
with different libraries, different accuracy profiles, and different
platform requirements (see below), so they are implemented as two
independent modules that never import from each other — including their
own separate (near-identical) copies of the VADER/alignment logic:

```
api/pipelines/
├── training/            # Pipeline 1 — reads the dataset, no audio processing
│   ├── sentiment.py       VADER lyric sentiment
│   ├── alignment.py       valence normalization + alignment gap formula
│   └── pipeline.py        orchestrator (load CSV → add columns → write CSV)
└── demo/                 # Pipeline 2 — extracts features from raw audio + lyrics
    ├── audio_io.py         shared decode + edge-case handling
    ├── librosa_features.py tempo, spectral centroid, key, mode, duration
    ├── liveness_panns.py   PANNs CNN14 crowd-noise proxy
    ├── essentia_features.py unused — kept but not called, see "Real-song validation" below
    ├── vggish_tfhub.py      valence/acousticness/instrumentalness/danceability/energy/speechiness/loudness (TF-Hub VGGish + a regression head per target, own subprocess)
    ├── lyric_sentiment.py  VADER lyric sentiment (own copy, not shared with Pipeline 1)
    ├── alignment.py        alignment gap against the VGGish valence estimate (own copy)
    └── pipeline.py          orchestrator (combines all of the above)
```

## Full feature list

| Feature | Pipeline 1 (training) source | Pipeline 2 (demo) source |
|---|---|---|
| `tempo` | dataset column | Librosa (onset-strength autocorrelation) |
| `loudness` | dataset column | TF-Hub VGGish embedding + gradient-boosted trees, trained on Kaggle precomputed embeddings |
| `key` | dataset column | Librosa chroma + Krumhansl-Schmuckler key-profile correlation |
| `mode` | dataset column | same key-detection step as `key` (major/minor) |
| `energy` | dataset column | TF-Hub VGGish embedding + Ridge regression, trained on Kaggle precomputed embeddings |
| `danceability` | dataset column | TF-Hub VGGish embedding + gradient-boosted trees, trained on Kaggle precomputed embeddings |
| `speechiness` | dataset column | TF-Hub VGGish embedding + Ridge regression, trained on Kaggle precomputed embeddings |
| `acousticness` | dataset column | TF-Hub VGGish embedding + gradient-boosted trees, trained on Kaggle precomputed embeddings |
| `instrumentalness` | dataset column | TF-Hub VGGish embedding + Ridge regression, trained on Kaggle precomputed embeddings |
| `liveness` | dataset column | PANNs CNN14 (mean prob. over crowd-noise classes) |
| `duration_ms` | dataset column | Librosa — `len(y) / sr * 1000`, no estimation needed |
| `valence` | dataset column | TF-Hub VGGish embedding + gradient-boosted trees, trained on Kaggle precomputed embeddings |
| `spectral_centroid` | *not used* | Librosa (extra descriptor, demo-only) |
| `lyric_sentiment` | VADER compound score on `lyrics` | VADER compound score on submitted lyrics (optional input) |
| `valence_normalized` | computed: `(2 * valence) - 1` | computed the same way, from the VGGish-Ridge `valence` estimate |
| `alignment_gap` | computed: `lyric_sentiment - valence_normalized` | same formula — only present if lyrics were submitted |

All twelve Spotify audio descriptors are now estimated by Pipeline 2, at
three different tiers of confidence: `tempo`/`key`/`mode`/`duration_ms`/
`loudness` are real, established DSP techniques (not proxies);
`valence`/`acousticness`/`instrumentalness`/`danceability`/`energy`/
`speechiness` are Ridge regressions fit directly against real Spotify
values (see "Valence, acousticness, instrumentalness" below); `liveness` is
the one remaining hand-built proxy (a general-purpose audio tagger
repurposed for this, not fit against real Spotify values). See
[Known limitations](#known-limitations--open-items) and "Real-song
validation" below for actual measured accuracy, not just which tier a
feature is in.

### Source legend

- **Dataset column** — read as-is, no computation. This is 10 of the 12
  audio descriptors for Pipeline 1 (all except the two derived ones below).
- **Computed** — deterministic formula over existing columns (`valence_normalized`,
  `alignment_gap`) or a rule-based NLP score with no training data
  (`lyric_sentiment`, via VADER).
- **Approximated** — Pipeline 2's estimates are proxies fit or trained against
  DSP heuristics or general-purpose pretrained models, not Spotify's actual
  (proprietary, undocumented) feature-extraction algorithms. Treat them as
  correlated stand-ins, not ground truth.

## Pipeline 1 — training-side (`api/pipelines/training/`)

Runs natively on Windows. No audio processing — purely reading dataset
columns and two lightweight NLP/arithmetic steps.

### Lyric sentiment (`sentiment.py`)

`compute_lyric_sentiment(lyrics)` runs VADER (`vaderSentiment` package) on
the `lyrics` column and returns the **compound** score, already bounded
`[-1, 1]`. Missing lyrics (`NaN`, `None`, non-string, or blank/whitespace
after stripping) return `0.0` (neutral) instead of raising — the dataset has
rows with no usable lyrics and the pipeline must not crash on them.

**Cleaning before scoring.** `clean_lyrics_for_sentiment()` strips
structural noise from raw lyrics before handing them to VADER — verified
directly against `songs(1).csv`, `lyrics` contains literal two-character
`\n` escape artifacts in 51 rows and bracketed section tags like
`[Chorus]`/`[Verse 1]` in 10,681 rows (2.2% of the 491,632 total). Left in,
these inject tokens VADER has no real sentiment to assign, diluting the
compound score. Removed: escape-sequence artifacts (`\n`/`\r`/`\t`,
`\uXXXX`, stray backslashes), bracketed metadata, LRC-style timestamps —
mirroring the structural-noise steps in
`notebooks/2_lyrics_filtering.ipynb`. **Deliberately not** lowercased or
stripped of punctuation, unlike that notebook's cleaning — VADER uses
capitalization (ALL CAPS = emphasis) and punctuation (e.g. `"!"`) as real
sentiment-intensity signals, so touching those would throw away information
VADER is designed to use; the notebook lowercases for a different task
(language detection) with different needs. A track whose lyrics are
*entirely* metadata (e.g. `"[Instrumental]"`) cleans down to an empty
string and scores neutral, same as a missing-lyrics row.

Verified against the real dataset, not just a synthetic example: cleaning
+ scoring the actual raw lyrics for "Glue Song" from `songs(1).csv`
reproduces the notebook's reference value exactly
(`lyric_sentiment = -0.1372`) — see
`tests/pipelines/training/test_sentiment.py::test_matches_notebook_reference_value_on_real_dataset_lyrics`.

### Alignment gap (`alignment.py`) — the central formula

This is the single most important formula in the thesis; the order and the
sign of the subtraction are load-bearing.

```
valence_normalized = (2 * valence) - 1        # rescales Spotify's [0,1] valence to [-1,1]
alignment_gap       = lyric_sentiment - valence_normalized
```

**Sign convention:**
- **Positive** `alignment_gap` → the lyrics read more positive than the music
  sounds (happy/sad lyrics over sad/happy-sounding music is the interesting
  case for a "mismatch" study).
- **Negative** `alignment_gap` → the music sounds more positive than the
  lyrics read.
- The subtraction is `lyric_sentiment - valence_normalized`, **not** the
  reverse — flipping it flips the sign of every alignment feature in the
  dataset with no error to catch it.

Verified against the existing hand-computed reference in
`notebooks/data_filtered/songs_2023.csv` ("Glue Song": `valence=0.582` →
`valence_normalized=0.164`, `lyric_sentiment=-0.1372` →
`alignment_gap=-0.3012`) — see `tests/pipelines/training/test_alignment.py`.

### Orchestrator (`pipeline.py`)

```python
from api.pipelines.training.pipeline import run

run("songs(1).csv", "songs_with_features.csv")
```

`load_dataset()` checks all 12 audio-descriptor columns plus `lyrics` are
present and raises a clear `ValueError` naming the missing columns rather
than failing deep inside pandas. `extract_training_features()` adds
`lyric_sentiment`, `valence_normalized`, and `alignment_gap` and leaves the
existing columns untouched.

## Pipeline 2 — demo-app-side (`api/pipelines/demo/`)

For a new song not in the training dataset. Runs as a single process inside
**WSL2** (see [`PIPELINE_SETUP.md`](PIPELINE_SETUP.md#why-pipeline-2-needs-wsl2)
for why the whole pipeline, not just the Essentia calls, lives there).

### Audio I/O (`audio_io.py`)

`load_audio(path)` is the single entry point every extractor uses. It
decodes with Librosa and raises one of two specific exceptions instead of
letting a raw decoder crash propagate:

- `UnsupportedAudioError` — file doesn't exist, or can't be decoded at all
  (corrupt file, format Librosa/`audioread`/`soundfile` don't support).
- `AudioTooShortError` — zero-sample audio, or a clip shorter than
  `MIN_DURATION_SEC` (0.5s) to extract meaningful spectral/rhythmic features
  from.

Silence is *not* an error — a valid, decodable silent clip loads fine and
produces zero-valued (not `NaN`, not crashed) downstream features.

### Librosa features (`librosa_features.py`)

| Feature | Method |
|---|---|
| `energy` | Hand-weighted composite, **not** raw RMS — see below |
| `tempo` | onset-strength autocorrelation (`librosa.onset.onset_strength` → `librosa.feature.rhythm.tempo`) |
| `spectral_centroid` | `librosa.feature.spectral_centroid`, averaged across frames |
| `duration_ms` | `len(y) / sr * 1000` — arithmetic, not extraction |
| `key`, `mode` | chroma-profile correlation, see below |

On silent audio, Librosa itself already returns `0.0` for the first three
rather than `NaN` or raising — verified in
`tests/pipelines/demo/test_librosa_features.py` rather than assumed.
`extract_key_and_mode` has its own silence guard (below).

**Tempo (`extract_tempo`) — the one feature real-song validation couldn't
turn into a clean win, but did explain.** Against 18 real songs
(`scripts/validate_pipeline_accuracy.py`), the raw onset-strength-
autocorrelation tempo estimate correlated at essentially zero (**0.025**)
against real Spotify tempo — several unrelated songs produced near-
identical BPM outputs (e.g. four different tracks all landing at
129.199219), the signature of librosa's tempo estimator collapsing onto its
internal prior (centered at 120 BPM by default, `std_bpm=1.0`) rather than
tracking the audio's actual periodicity. Loosening the prior via
`std_bpm=1.5` (vs. the default `1.0`) helped — 0.025 → 0.390 at n=18 — but
re-swept at n=66 the improvement was much smaller and the specific best
`std_bpm` value turned out not to be well-determined (see
`_TEMPO_STD_BPM`'s comment in `librosa_features.py`); n=66 correlation
settled at **0.182**.

**Second validation pass: is the remaining error just octave confusion?**
Half/double-tempo errors (locking onto twice or half the true BPM — beats
vs. half-notes) are a well-known, general MIR failure mode. An **octave-
tolerant** re-scoring — crediting a prediction if it's correct after ×1,
×2, or ×0.5 — jumped correlation from **0.182 to 0.939** and cut MAE from
20.8 to 6.2 BPM. 17 of 66 songs (26%) were landing at exactly half or
double the real tempo. This means the underlying periodicity detection is
genuinely strong; the actual failure is almost entirely *which* octave, not
*how fast*. (An earlier attempt at this session concluded octave-folding
would be a no-op, based on inspecting post-fold values from an older
implementation that no longer reflects the current `std_bpm=1.5` code —
that conclusion doesn't hold up against the current version, which does
produce genuine outside-plausible-range octave errors.)

This *doesn't* translate into an easy production fix, though: 0.939 is an
**oracle ceiling** that requires knowing the real tempo to pick the right
octave. A ground-truth-free fix (folding blindly into a fixed preferred
band like [80,160]) was tested and made things *worse*, not better
(corr **-0.076** at that band) — it can't distinguish a genuine octave
error from a correctly-estimated fast or slow tempo, and ends up breaking
correct predictions to "fix" ones that weren't wrong. Real octave
disambiguation without ground truth would need a genuine secondary signal
(e.g. cross-checking against a second, independent tempo estimator and
using agreement/disagreement as a confidence signal) — a real, open
problem, not implemented here. **Kept as the current Librosa/`std_bpm=1.5`
implementation, honestly flagged as weak** (corr 0.182) rather than swapped
for a VGGish-Ridge regression that scored a better raw correlation (0.416,
tested end-to-end on the same 66 songs) by compressing nearly every
prediction into a narrow ~105-140 BPM band regardless of the real tempo —
a worse fit for this pipeline's actual use (a single plausible-looking BPM
reading per uploaded song), even though it's a better aggregate statistic.

**`key`/`mode` via Krumhansl-Schmuckler** — unlike danceability/speechiness/
energy (all regression-based estimates, not real DSP), key detection has an
established, well-understood technique: average the track's chroma (12-bin pitch-class
energy, `librosa.feature.chroma_cqt`) over time, then correlate that against
24 candidate profiles (the empirically-measured Krumhansl-Kessler major/minor
key profiles, each of the 12 possible tonics) and take the best match. Both
`key` (0=C, 1=C#/Db, ..., 11=B) and `mode` (1=major, 0=minor) come from
whichever of the 24 wins — `librosa`'s chroma index convention already
matches Spotify's key encoding, so no reindexing is needed. Verified against
two clean test chords, not just assumed correct: a C-E-G triad detects
`key=0, mode=1` exactly, and an A-C-E triad detects `key=9, mode=0` exactly
(`tests/pipelines/demo/test_librosa_features.py`). Real songs are messier
than a clean triad, so treat this as reliable-ish rather than exact — it's
the standard technique for this exact task, not a regression estimate like
speechiness/danceability/valence.

**`energy`** used to be a hand-weighted Librosa composite here (50% RMS
loudness, 30% spectral brightness, 20% onset density, weights never fit
against real data) — real-song validation showed a VGGish-embedding Ridge
regression against real Spotify energy clearly beats it (corr 0.570 →
0.916). It's computed by `api.pipelines.demo.vggish_tfhub` now; see the
"Real-song validation" section above for the numbers and
`feature_extraction.md`'s Valence/acousticness/instrumentalness section for
how the VGGish-Ridge approach works.

### Speechiness (moved to `vggish_tfhub.py`)

Speechiness used to be a from-scratch proxy here: four hand-picked
low-level features (zero-crossing-rate variance, spectral-flatness
variance, low-energy frame ratio, ~4Hz modulation energy) fed through a
`scikit-learn` regression (`SpeechinessRegressor`), calibrated on a
**synthetic** proxy set — 16 Windows-SAPI text-to-speech clips (target
`0.93`) and 14 synthetic/real non-speech clips (target `0.04`), assembled
and fit by `scripts/generate_calibration_speech.ps1` /
`generate_calibration_synthetic.py` / `build_speechiness_manifest.py` /
`calibrate_speechiness.py`. Held-out R²=0.82 on that proxy set — but real-
song validation (`scripts/validate_pipeline_accuracy.py`, n=66) showed it
correlated at only **0.206** against real Spotify speechiness. Targets
assigned by construction ("this is TTS, therefore speechy") never actually
matched how real, more ambiguous audio (rap, spoken-word-over-music) scores
on Spotify's own metric.

Replaced entirely with the same VGGish-embedding Ridge regression as
valence/acousticness/instrumentalness/danceability/energy — held-out
R²=0.526/corr=0.725 against real Spotify speechiness (full 491k-track
dataset), a large jump from the proxy-calibrated version. The whole
synthetic-calibration sub-system (the four scripts above, `speechiness.py`,
`calibrate_speechiness.py`, the calibration audio clips, and the frontend's
"uncalibrated" badge tied to a `speechiness_calibrated` API field) was
removed rather than kept alongside — it had no remaining purpose once real
Spotify data was available to fit against directly.

### Liveness proxy (`liveness_panns.py`)

Liveness estimates "was this recorded with an audience present," which has
no DSP shortcut. Approximated with **PANNs CNN14**
(`panns-inference` package, AudioSet-pretrained, checkpoint
`Cnn14_mAP=0.431.pth`, 327MB, MD5 `541141fa2ee191a88f24a3219fff024e`): the
model tags the clip against AudioSet's ~527 classes, and `liveness` is the
mean predicted probability across the classes whose labels contain
`applause`, `cheering`, or `crowd` (case-insensitive substring match
against PANNs' label list, so a labeling change upstream doesn't silently
start scoring zero classes — it raises instead). Audio is resampled to
32kHz (the rate CNN14 was trained on) if needed.

The checkpoint is not bundled — download it from the
[PANNs release on Zenodo](https://zenodo.org/records/3987831) (direct link:
`https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth?download=1`)
and pass its path via `extract_liveness(y, sr, checkpoint_path=...)`, or let
`panns_inference` auto-download it to `~/panns_data/` on first use.

**Windows caveat:** `panns_inference` itself (not the model) shells out to
`wget` on first import to fetch a small AudioSet label CSV. `wget` isn't a
standard Windows command, so a fresh install fails there with a raw
`FileNotFoundError` — `_load_model()` in `liveness_panns.py` catches that
broadly (not just `ImportError`) so it still fails with a clear message
rather than a confusing stack trace. Works without issue under WSL2, where
`wget` is present.

### Essentia features (`essentia_features.py`) — no longer used

Essentia doesn't do anything in the current pipeline. `danceability`
(Essentia's `Danceability` algorithm) and `loudness` (Essentia's
`ReplayGain` algorithm) both used to live here; both were replaced by
VGGish-embedding regressions after real-song validation showed the
replacement clearly won (`danceability` corr 0.353 → 0.726, `loudness`
corr 0.605 → 0.832 — see "Real-song validation" below). The module and its
test are kept, not deleted, but nothing in `pipeline.py` calls them
anymore — see that section's closing note for why this makes the whole
WSL2 requirement worth revisiting as a separate decision.

#### Loudness's history (`extract_loudness`, no longer called)

Kept for the record: `ReplayGain` (ReplayGain 1.0 spec: equal-loudness-
filtered signal energy) is a real, standard loudness-measurement
technique, not a hand-built proxy — the same general family streaming
platforms use for loudness normalization. It still had a real, confidently-
wrong bug, caught by real-song validation
(`scripts/validate_pipeline_accuracy.py`, 18 real songs): the raw output of
Essentia's `ReplayGain` isn't loudness — per Essentia's own docs it's "the
distance to the suitable average replay level (~-31dB)," i.e. a *gain
adjustment*, inversely related to actual loudness. Using it directly as
"loudness" produced a **-0.52 correlation** against real Spotify loudness.
Negating it and re-basing around an empirically-fit reference constant
(`-19.686`) flipped that to **+0.52** (later **+0.605** at n=66) — a real,
working fix at the time. It was superseded entirely once loudness was
tried against VGGish embeddings for the first time (never attempted
before, since `ReplayGain` being a real DSP measurement made it seem like
it didn't need a regression) and turned out to have much more headroom
than assumed: VGGish + gradient-boosted trees reached **0.832** end-to-end,
a bigger jump than the sign-bug fix itself.

### Valence, acousticness, instrumentalness (`vggish_tfhub.py`)

These three used to be Essentia MusiCNN model heads, living in
`essentia_features.py` alongside danceability/loudness. They aren't
anymore — moved out to their own module, `vggish_tfhub.py`, which has to
run as an **isolated subprocess**, never imported into the same process as
`essentia_features.py`. Two things forced this:

1. **Essentia's own VGGish embedding doesn't match the training data.**
   Essentia ships `audioset-vggish-3.pb`, described as trained on a
   "preliminary subset of YouTube-8M" — an independently-trained model with
   the same architecture as Google's official VGGish release, but different
   weights. Validated against the Kaggle precomputed-embeddings dataset
   (which was built from Google's official `tfhub.dev/google/vggish/1`):
   correlation ~0.80, cosine similarity only ~0.56. Regression models
   trained on the Kaggle embeddings do not transfer to Essentia-computed
   embeddings at inference time — the two are related but meaningfully
   different feature spaces.
2. **Essentia's bundled TensorFlow and a standalone `tensorflow`/
   `tensorflow_hub` install cannot coexist in one process.** Importing
   `essentia` and then loading `tensorflow_hub`'s VGGish module in the same
   script crashes with `Check failed: ... ALREADY_EXISTS: Op with name
   Bitcast` — a C++ TensorFlow op-registry collision, not something
   catchable from Python. Confirmed by isolating the two: removing the
   `essentia` import (using `librosa.load(..., sr=16000, mono=True)`
   instead of `es.MonoLoader`) let Google's official TF-Hub VGGish load
   cleanly, producing embeddings with **0.97 correlation / 0.97 cosine
   similarity** against the Kaggle dataset for a real test song — the
   architecture diagnosis was correct, and switching embedding sources
   fixes it.

So `vggish_tfhub.py` loads `tfhub.dev/google/vggish/1`, computes a
mean-pooled 128-dim embedding via `librosa`-loaded audio (16kHz mono, no
Essentia in this process at all), and feeds it to seven regression models
— `models/vggish_ridge/{valence,acousticness,instrumentalness,danceability,
energy,speechiness,loudness}.joblib` — trained on the Kaggle dataset's
precomputed VGGish embeddings against the real Spotify target columns (see
`scripts/train_vggish_ridge.py`). Not all seven use the same model class —
`load_ridge_models`/`predict_vggish_features` don't care, they just call
`.predict()` on whatever `joblib.load` returns. It's a plain script,
importable and runnable standalone:

```
uv run python -m api.pipelines.demo.vggish_tfhub <audio_path> [--models-dir <dir>]
```

`pipeline.py`'s orchestrator calls it with `subprocess.run([sys.executable,
"-m", "api.pipelines.demo.vggish_tfhub", ...])` rather than importing it
directly — the same isolation the crash above requires, just automated
instead of manual. This means every demo request now spawns two Python
processes inside the outer WSL2 subprocess (the main pipeline process, plus
this one) — a second TensorFlow runtime load per request, and the first
request after a cold start also pays for a one-time ~280MB VGGish module
download (cached afterward under `models/tfhub_cache/`, set via
`TFHUB_CACHE_DIR`). `demo_pipeline_timeout_seconds` was bumped from 120s to
180s to give this room.

**Training the Ridge models** (one-time, from WSL2):

```
uv run python scripts/train_vggish_ridge.py \
    data/kaggle_embeddings/vggish_embeddings.npz \
    "/mnt/c/Users/User/Downloads/songs(1).csv"
```

Fits each target's model (per `_TARGET_MODELS` — `StandardScaler ->
RidgeCV` for instrumentalness/energy/speechiness,
`HistGradientBoostingRegressor` for valence/acousticness/danceability/
loudness) on an 80/20 split, prints held-out R²/correlation, and writes the
seven `.joblib` files. Uses the *same* embedding source
(`vggish_embeddings.npz`) `vggish_tfhub.py` matches at inference time —
that match is the entire point; retraining against a different embedding
source (e.g. Essentia's own VGGish) would reintroduce the exact mismatch
described above.

### Lyric sentiment + alignment gap (`lyric_sentiment.py`, `alignment.py`)

The second input: a new song's lyrics, submitted alongside its audio.
`lyric_sentiment.py` is VADER — same cleaning + scoring as Pipeline 1's
`sentiment.py` (see its section above for what's stripped and why
capitalization/punctuation are deliberately left alone), copied rather
than imported per the never-mix rule. `alignment.py` is the identical
`alignment_gap = lyric_sentiment - valence_normalized` formula — the only
real difference from Pipeline 1 is *where `valence` comes from*: Pipeline 1
reads it from the Spotify dataset, Pipeline 2 uses the VGGish-Ridge
demo-estimated `valence` (already in `[0, 1]`, see above), since there's no
real Spotify value for a song Spotify has never scored.

Both are **optional** — pass `lyrics=None` (or omit `--lyrics-file` on the
CLI) for an instrumental track, and the result simply has no
`lyric_sentiment`/`alignment_gap` keys rather than raising.

### Orchestrator (`pipeline.py`)

```python
from api.pipelines.demo.pipeline import extract_demo_features

features = extract_demo_features(
    "uploaded_song.wav",
    lyrics="paste the song's lyrics here (optional)",
    panns_checkpoint="models/panns/Cnn14_mAP=0.431.pth",
    vggish_ridge_models_dir="models/vggish_ridge",
)
```

Every stage prints a `[pipeline] ...` marker to stderr as it runs (loading
audio, each extractor, sentiment, alignment) — that's what shows up live in
the demo visualizer's log panel, not just the raw TensorFlow/Essentia
noise. `_extract_json` in the backend service specifically scans stdout for
the JSON block, so these stay on stderr deliberately.

Also runnable as a script — see [`PIPELINE_SETUP.md`](PIPELINE_SETUP.md#running-the-pipelines).

## Windows / WSL2 platform split — and why

| Component | Platform | Why |
|---|---|---|
| Pipeline 1 (training) | Windows (native) | Pure pandas + VADER, no audio libraries at all. |
| Librosa | Capable of running natively on Windows | No platform-specific dependency. |
| Essentia | **Linux only, and currently unused** | No Windows wheel exists for `essentia-tensorflow` — its C++ core has never shipped one, and there's no sdist to build from source. As of the loudness move to VGGish (see "Real-song validation" below), **nothing in `pipeline.py` calls Essentia anymore** — this row is now historical/dormant, not an active reason to require WSL2. |
| PANNs/PyTorch (liveness) | Capable of running natively on Windows, with one caveat | `panns_inference` shells out to `wget` on first import to fetch a small AudioSet label CSV — `wget` isn't a standard Windows command, so a fresh install fails there with `FileNotFoundError` (see the Liveness section above). Works under WSL2, where `wget` exists. This is now the **only** thing tying Pipeline 2 to WSL2 — a narrow, workaroundable quirk, not a fundamental C++ Windows-wheel gap like Essentia's. |
| TF-Hub VGGish + a regression head per target (valence, acousticness, instrumentalness, danceability, energy, speechiness, loudness) | Capable of running natively on Windows | No platform-specific dependency, but runs as its own subprocess regardless of platform — it can't share a process with Essentia, and while Essentia is currently unused, the isolation is left in place rather than assumed safe to remove (see `vggish_tfhub.py`'s section above). |
| **Pipeline 2 as a whole** | **WSL2, though this may no longer be strictly necessary** | Historically required because every demo request needed Essentia's Linux-only output. That's no longer true — Essentia has zero callers now. The one remaining Windows blocker is PANNs' `wget` dependency above, which is narrower and likely fixable (e.g. pre-downloading the label file, or patching `panns_inference`'s fetch call) rather than a hard platform wall. Whether to actually pursue removing the WSL2 requirement is a deliberate, separate decision — not made here, and not a side effect of the loudness/GBM change that surfaced it. |

Both pipelines share a single `pyproject.toml`/`uv.lock`; the Linux-only
dependency is gated with a PEP 508 marker (`sys_platform == 'linux'`), so
`uv sync` installs the right set on each side automatically. Full setup
steps are in [`PIPELINE_SETUP.md`](PIPELINE_SETUP.md).

## Testing

`tests/pipelines/training/` and `tests/pipelines/demo/` mirror the pipeline
structure. Edge cases covered per the thesis's requirements:

- **Empty/missing lyrics** (Pipeline 1) — `None`, `NaN`, empty string,
  whitespace-only string all score neutral (`0.0`) rather than raising;
  covered for both direct calls and a `pandas.Series.apply` pass.
- **Silent audio** (Pipeline 2) — a true-zero waveform loads successfully
  (silence is valid audio, not corrupt) and every extractor returns `0.0`
  instead of `NaN` or crashing.
- **Very short clips** (Pipeline 2) — clips under 0.5s (including
  zero-sample files) raise `AudioTooShortError` before reaching any
  extractor.
- **Unsupported audio formats** (Pipeline 2) — a non-audio file with an
  audio-like extension, and a missing file path, both raise
  `UnsupportedAudioError` rather than an unhandled decoder exception.
- **Essentia unavailable** (Pipeline 2) — `essentia_features.py` is no
  longer called by `pipeline.py` (loudness moved to VGGish), but the module
  and its guard test are kept: on a machine without Essentia installed
  (e.g. Windows), calling `extract_loudness` directly still raises a
  `RuntimeError` naming WSL2 as the fix, instead of an `ImportError` stack
  trace; skipped automatically on a machine where Essentia *is* installed.
- **`vggish_tfhub.py`'s regression-model glue** — `predict_vggish_features`/
  `load_ridge_models` are tested with a mocked embedding and tiny
  constant-output Ridge models (`tests/pipelines/demo/test_vggish_tfhub.py`),
  not a real TF-Hub download — that part was validated manually instead
  (see the correlation numbers in the "Valence, acousticness,
  instrumentalness" section above), same reasoning as the PANNs/Essentia
  checkpoints not being downloaded in CI.

Run everything with:

```powershell
uv run pytest tests
```

Essentia's guard-path test exercises the Windows fallback there and
self-skips under WSL2 (where Essentia *is* installed), same as always,
even though `essentia_features.py` isn't in the production call path
anymore. Verified live under WSL2 too, not just by the test suite:
`api.pipelines.demo.vggish_tfhub` (all seven targets, including loudness
now) ran successfully against real audio files (66 of them, per the
"Real-song validation" section above), and the Windows-side Librosa
extractors ran against a real audio file as well.

## Running it here, now

1. **One-time setup** (Windows):
   ```powershell
   uv sync
   ```
   This installs everything Pipeline 1 needs, plus everything Pipeline 2
   needs except Essentia (skipped via the platform marker).

2. **Pipeline 1** — extract training features from the dataset:
   ```powershell
   uv run python -m api.pipelines.training.pipeline "C:\Users\User\Downloads\songs(1).csv" notebooks\data_filtered\songs_features.csv
   ```
   For the full 491,632-row dataset this takes a few minutes (VADER scores
   each row's lyrics individually); a few thousand rows run in seconds.

3. **Pipeline 2** — one-time WSL2 setup (see
   [`PIPELINE_SETUP.md`](PIPELINE_SETUP.md#one-time-wsl-setup) for the full
   steps), then download the model checkpoints into `models/` (gitignored)
   and run from WSL2:
   ```bash
   # PANNs checkpoint is 327MB — download manually, or omit --panns-checkpoint
   # and let panns_inference auto-download it to ~/panns_data/ on first use.
   mkdir -p models/panns
   curl -sSL -o "models/panns/Cnn14_mAP=0.431.pth" "https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth?download=1"

   # Ridge models (valence/acousticness/instrumentalness/danceability/
   # energy/speechiness) — one-time training, see the "Valence,
   # acousticness, instrumentalness" section above.
   uv run python scripts/train_vggish_ridge.py \
     data/kaggle_embeddings/vggish_embeddings.npz "/mnt/c/Users/User/Downloads/songs(1).csv"

   uv run python -m api.pipelines.demo.pipeline path/to/song.wav \
     --panns-checkpoint "models/panns/Cnn14_mAP=0.431.pth" \
     --vggish-ridge-models-dir models/vggish_ridge
   ```

4. **Tests** (Windows or WSL2 — both work; Essentia's tests self-skip on
   whichever side doesn't have it installed):
   ```powershell
   uv run pytest tests
   ```

## Real-song validation (`download_validation_songs.py`, `validate_pipeline_accuracy.py`)

The Kaggle held-out numbers above validate the *training* side (embeddings
vs. real Spotify columns) but say nothing about Pipeline 2's actual
end-to-end output on real audio. `scripts/download_validation_songs.py`
downloads real audio for a random sample of dataset tracks via `yt-dlp`
(confidence-scored title/duration/artist matching, same idea as
essteec/music-prediction's acquisition scripts but scaled down and willing
to just skip a low-confidence match rather than accept it), writes a
manifest of their real Spotify values. `scripts/validate_pipeline_accuracy.py`
then runs the *actual* demo pipeline against each file and reports
per-feature MAE/correlation against those real values — the same thing the
single manually-sourced Glue Song test was doing, just at n=18 instead of
n=1, and reproducible.

Current numbers (n=66, random sample, `random_state=42`, grown from an
initial n=18 batch — see below for why the growth itself matters):

| Feature | MAE | corr | Read |
|---|---|---|---|
| energy | 0.075 | 0.916 | strong — VGGish-GBM, replaced the old hand-weighted composite (was 0.570) |
| duration_ms | 2974 ms | 0.994 | arithmetic, expected near-perfect |
| acousticness | 0.074 | 0.875 | strong — VGGish-GBM, replaced VGGish-Ridge (was 0.858) |
| danceability | 0.091 | 0.726 | strong — VGGish-GBM, replaced Essentia's DFA algorithm (was 0.353) |
| valence | 0.121 | 0.747 | strong — VGGish-GBM, replaced VGGish-Ridge (was 0.712) |
| speechiness | 0.034 | 0.709 | strong — VGGish-Ridge, replaced a synthetic-TTS-calibrated regression (was 0.206) |
| **loudness** | **1.65 dB** | **0.832** | strong — VGGish-GBM, replaced Essentia `ReplayGain` (was 0.605) — the single biggest win of this round |
| liveness | 0.224 | 0.694 | moderate — see caveat below |
| instrumentalness | 0.132 | 0.645 | moderate |
| tempo | 20.8 BPM | 0.182 | weak, but see below — mostly octave confusion (half/double tempo), not bad periodicity detection |

**Danceability, energy, and speechiness were all re-fit the same way
valence/acousticness/instrumentalness were** — a VGGish-embedding Ridge
regression against real Spotify values, replacing Essentia's `Danceability`
algorithm, the old hand-weighted RMS/brightness/onset-density composite,
and a regression calibrated only on synthetic text-to-speech clips,
respectively. All three were clean, decisive wins, confirmed end-to-end:
danceability corr 0.353 → 0.726, energy corr 0.570 → 0.916, speechiness
corr 0.206 → 0.709. `essentia_features.py`, `librosa_features.py`'s old
`extract_energy`, and the standalone `speechiness.py`/
`calibrate_speechiness.py` module (plus its whole synthetic-calibration
script chain and the frontend's "uncalibrated" badge) no longer exist.

**Second round: gradient-boosted trees (`HistGradientBoostingRegressor`)
beat Ridge for valence/acousticness/danceability/loudness, and loudness
moved from Essentia DSP to VGGish entirely.** Triggered by a single-song
spot-check (the Glue Song — see below) where valence, acousticness,
danceability, and loudness all missed by more than their typical error,
and in a specific pattern: the real value was extreme (near the top or
bottom of the feature's range) and the prediction was pulled toward the
middle — the signature of Ridge's L2 shrinkage. Tested head-to-head against
GBM on the same embeddings: GBM won on every metric for all four,
including an extreme-value-tail MAE specifically constructed to test this
hypothesis (top/bottom 10% of real values only) — acousticness's tail MAE
dropped from 0.108 to 0.073, the largest gap of the four. Loudness with
VGGish (Ridge *or* GBM) was tried for the first time in this round — it had
never been attempted before, on the assumption Essentia's `ReplayGain` (a
real DSP measurement) didn't need it — and even plain Ridge (0.796 held-out
corr) blew past Essentia's fitted-constant approach (0.605 real-song corr).
`scripts/train_vggish_ridge.py`'s `_TARGET_MODELS` now maps each target to
its own model class explicitly (GBM for valence/acousticness/danceability/
loudness, Ridge for instrumentalness/energy/speechiness — untested with GBM,
not assumed to benefit). Confirmed end-to-end, not just held-out: loudness
corr 0.605 → 0.832 (MAE 2.20 → 1.65 dB), acousticness 0.858 → 0.875 (MAE
0.101 → 0.074), valence 0.712 → 0.747 (MAE 0.129 → 0.121), danceability MAE
0.095 → 0.091 (correlation unchanged at 0.726 — an n=66 rounding artifact,
not a null result).

**One honest caveat from re-running the Glue Song spot-check after this
change**: loudness and danceability improved for that specific song too
(loudness error 4.0dB → 1.8dB), but valence and acousticness *did not* —
valence's error grew slightly (0.300 → 0.314) and acousticness's grew more
(0.339 → 0.393), even though both improved in aggregate across all 66
songs. This isn't a contradiction: GBM reduces the *average* extreme-value
error, not every individual extreme case, and Glue Song's real acousticness
(0.770) is itself an extreme, atypical value. Aggregate improvement is real
and was the basis for adopting this change, but it doesn't mean every song
gets better — worth remembering before reading too much into any single
spot-check, including this one.

`tempo` was tried too (VGGish-Ridge held-out R²=0.099/corr=0.315, end-to-end
corr=0.416) but deliberately *not* adopted — it wins on aggregate
correlation only by compressing nearly every prediction into a narrow
~105-140 BPM band, which is a worse fit for this pipeline's actual job (one
plausible BPM reading per uploaded song) than the current, noisier-but-
wider-ranging Librosa estimate. See the "Tempo" section above for the
octave-confusion finding that actually explains most of tempo's remaining
error, and why blind octave-correction doesn't fix it in production either.

**`essentia_features.py` now has zero callers.** Loudness was its last job;
moving loudness to VGGish means nothing in `pipeline.py` imports or calls
Essentia at all anymore. The module and its test file are still present
(not deleted — this wasn't the point of this round of changes, and removing
it opens a bigger question: Essentia's Windows-wheel gap was the *entire
reason* Pipeline 2 requires WSL2 in the first place, so if Essentia is
truly unused, the whole WSL2 requirement may no longer be load-bearing —
worth a deliberate, separate decision, not a side effect of a feature
swap).

**Why the n=18 → n=66 jump matters more than either number alone:** three
features' correlations moved by more than 0.15 between the two runs —
`speechiness` (0.576 → 0.206), `liveness` (0.900 → 0.694, confirming the
suspicion that one obvious live track was inflating the small sample), and
`energy` (0.319 → 0.570, the other direction — it was never as weak as the
small sample suggested). Take any single-sample-size number here with that
in mind, this one included; rerun `download_validation_songs.py --n <more>`
(it appends to the existing manifest, skipping already-downloaded tracks)
for a tighter read before trusting a specific value.

Two real bugs this surfaced, one fix that held up and one that only
partially did (details in each feature's section above): `loudness` was
reporting Essentia's `ReplayGain` *gain* value directly, which is inversely
related to actual loudness (corr was **-0.52** before the fix) — the fix
(negate + re-base around a fit reference constant) held up well, corr
actually *improved* at n=66 (0.521 → 0.605) and the reference constant
barely moved on refit (-19.686 → -19.541). (This whole Essentia-based
approach was later replaced entirely by VGGish-GBM, corr 0.605 → 0.832 —
see "Real-song validation" below; kept here as the historical record of why
the sign bug happened and how it was diagnosed.) `tempo`'s estimator was
collapsing onto its internal ~120 BPM prior (corr **0.025** with the
default) — loosening it is a real, replicated improvement over the default
at both sample sizes, but the *specific* best value isn't well-determined:
a clean sweep at n=18 picked a clear peak, but re-swept at n=66 the same
range was noisy with no stable peak. This is now flagged in the code itself
(`_TEMPO_STD_BPM`'s comment) rather than presented as more settled than it
is — narrowing a failure mode isn't the same as fixing it.

## Known limitations / open items

- **Validation status by feature**, current as of the n=66 real-song run
  above: `loudness`/`energy`/`acousticness`/`danceability`/`valence`/
  `speechiness` are strong; `liveness`/`instrumentalness` are moderate;
  `tempo` is weak on a raw correlation basis (0.182), but a follow-up octave-tolerant
  analysis showed most of that error is half/double-tempo confusion, not
  bad periodicity detection (0.939 correlation once octave errors are
  credited) — a real, general MIR problem with no ground-truth-free fix
  found yet, not a simple bug (see the "Tempo" section above); `duration_ms`
  is arithmetic and reliably correct; `key`/`mode` have no real-song
  accuracy check yet (categorical, doesn't fit the MAE/correlation approach
  above — would need per-song correctness scoring instead). Held-out
  R²/correlation for the seven VGGish-embedding targets specifically
  (embedding vs. real Spotify columns, not end-to-end pipeline output, and
  not all the same model class — see `_TARGET_MODELS`) is printed by
  `scripts/train_vggish_ridge.py` — check it directly rather than trusting a
  number from an older conversation, since it drifts as the dataset or
  split changes.
- **No automated tests** for the FastAPI layer (`api/routers/demo_features.py`,
  `api/services/demo_feature_extraction_service.py`) — matches the rest of
  `api/` having no test suite yet, but worth adding given how much subprocess
  plumbing lives there.
