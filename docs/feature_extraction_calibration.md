# Recalibrating Audio Feature Extraction Against Spotify

This document explains how the demo application estimates Spotify's audio
features for a song Spotify has never analysed, and how those estimates were
calibrated step by step until they tracked Spotify's own values as closely
as the available data allows. It is written to support the methodology and
results chapters of the thesis. Implementation details live in
[feature_extraction.md](../feature_extraction.md).

## 1. Why calibration was needed

The thesis trains its popularity models on the Spotify dataset (491,632
tracks), where every audio descriptor (valence, energy, tempo, key, and so
on) is a value Spotify computed. The central feature, the alignment gap, is:

```
valence_normalized = (2 * valence) - 1
alignment_gap      = lyric_sentiment - valence_normalized
```

The demo application has to compute the same features for a **new** song:
an uploaded audio file and optional lyrics. Spotify's feature extractor is
proprietary and undocumented, and the Spotify Web API endpoint that exposed
these values is no longer available for new applications. The demo therefore
has to *estimate* each value from the raw audio.

If those estimates live on a different scale from Spotify's, the popularity
model receives inputs unlike anything it was trained on, and the alignment
gap for a new song means something different from the alignment gap in the
dataset. Calibration is the process of making the estimates line up with
Spotify's values, measured against real Spotify ground truth.

## 2. Data used

| Data | Size | Role |
|---|---|---|
| Spotify dataset (`songs(1).csv`) | 491,632 tracks | Ground-truth Spotify values for every feature |
| Precomputed audio embeddings (Kaggle derivative of the same dataset) | 491,632 tracks: VGGish (128-d), PANNs CNN14 (2048-d), MERT (768-d), mel statistics (512-d) | Training inputs, so models can be fit at full dataset scale without downloading 491k songs |
| Validation songs (`data/validation_songs/`) | 66 tracks | Real audio for dataset tracks, downloaded with `yt-dlp` using confidence-scored title/artist/duration matching; used for **end-to-end** testing |

The 66 validation songs are the key to the whole process. A model can score
well on held-out Kaggle rows and still fail in practice, because the demo
computes embeddings itself from a different copy of the audio. Running the
real pipeline on the validation songs and comparing its output with the
dataset's Spotify values tests the whole chain: audio decoding, embedding
extraction and the regression model.

## 3. Evaluation protocol

Every candidate change was scored at two levels:

1. **Held-out (training side).** An 80/20 split of the 491k Kaggle rows.
   This shows whether an embedding carries information about a feature.
2. **End-to-end (real songs).** The actual pipeline is run on the 66
   validation songs. Continuous features are reported as mean absolute
   error (MAE) and Pearson correlation; key and mode as accuracy.

**A change was adopted only if it improved the end-to-end numbers**, not
just the held-out ones. Several changes that looked good held-out were
rejected for this reason (Section 6).

To make experiments cheap, the intermediate features for each validation
song (embeddings, chroma, the raw tempo estimate) are cached in
`data/validation_songs/feature_cache.npz`. An experiment then takes minutes
instead of a ~70-minute full pipeline run. The cache was checked by
reproducing the production pipeline's numbers exactly before any experiment
relied on it.

A check that the local embeddings match the training embeddings: for the
validation songs, locally computed PANNs embeddings have a mean cosine
similarity of 0.961 with the Kaggle copies, and VGGish about 0.97. That
agreement is what lets models trained on the Kaggle embeddings transfer to
audio the demo processes itself.

## 4. The final method

The twelve Spotify descriptors fall into two groups.

**Signal-processing features** (computed directly, no training):

| Feature | Method |
|---|---|
| `duration_ms` | Length of the audio |
| `tempo` | Librosa onset-strength autocorrelation, with a loosened tempo prior (`std_bpm=1.5`) |
| `key`, `mode` | Librosa chroma averaged over time, correlated against the 24 major/minor key profiles of Temperley (1999) (the Krumhansl-Schmuckler method) |

**Learned features** (`valence`, `energy`, `danceability`, `acousticness`,
`instrumentalness`, `speechiness`, `loudness`, `liveness`):

1. Compute two pretrained audio embeddings of the uploaded song:
   - Google's VGGish (TF-Hub), mean-pooled to 128 dimensions.
   - PANNs CNN14, 2048 dimensions, reduced to 256 by an Incremental PCA
     fitted on the dataset (99.0% of variance kept).
2. Concatenate them into one 384-dimensional vector.
3. Feed it to one gradient-boosted regression model per feature
   (`HistGradientBoostingRegressor`), trained on all 491k dataset tracks
   against the real Spotify values. Features bounded in [0, 1] are clipped
   to that range.

In short, instead of trying to reverse-engineer Spotify's algorithm, the
pipeline **learns the mapping from general-purpose audio representations to
Spotify's own numbers**, using Spotify's values as the training target.

## 5. Calibration history

The pipeline reached this design through a sequence of rounds. Each row in
the tables below is an end-to-end result on real songs.

### Round 1: hand-built estimates replaced by learned regressions

The first version used hand-built estimates: a weighted mix of loudness,
brightness and onset density for energy, Essentia's `Danceability`
algorithm, and a speechiness model calibrated on synthetic text-to-speech
clips. Real-song validation showed these were weak. Each was replaced with a
regression on VGGish embeddings, trained on real Spotify values:

| Feature | Before (corr) | After (corr) |
|---|---|---|
| danceability | 0.353 | 0.726 |
| energy | 0.570 | 0.916 |
| speechiness | 0.206 | 0.709 |

Validation also caught two outright bugs. Loudness came from Essentia's
`ReplayGain`, which returns a *gain adjustment*, not loudness, so it was
inversely related to the truth (corr −0.52 before the fix). Tempo was
collapsing onto its built-in 120 BPM prior (corr 0.025 at the default
setting).

### Round 2: Ridge replaced by gradient boosting; loudness moved to embeddings

A single-song spot check ("Glue Song") showed a pattern: when Spotify's
value was extreme, the prediction was pulled toward the middle, the
signature of Ridge regression's shrinkage. Gradient-boosted trees beat
Ridge on every metric, including an error measure restricted to the most
extreme 10% of values. Loudness was also tried on embeddings for the first
time and clearly beat the signal-processing measurement (corr 0.605 →
0.832).

### Round 3: a second embedding, more capacity, and liveness

This round tested what would move the remaining features closer to
Spotify's values:

- **More model capacity** (up to 5000 boosting iterations, 63 leaves per
  tree, early stopping). All eight models converged before the cap.
- **A second embedding.** Adding PANNs CNN14 alongside VGGish improved
  every target (held-out valence corr 0.817 → 0.830).
- **Liveness.** Until now it was the mean probability of the "applause",
  "cheering" and "crowd" sound classes. That value was on the wrong scale
  entirely: its standard deviation across songs was 0.008, against 0.200
  for Spotify's liveness. It was replaced by a regression head like the
  other features.
- **Key profiles.** Temperley's key profiles replaced Krumhansl-Kessler's
  after both were scored on the 66 songs with three chroma variants.
  Temperley won on all three.

Final end-to-end results (66 real songs). "Before" is the Round 2 state:

| Feature | MAE before | MAE now | Corr before | Corr now |
|---|---|---|---|---|
| energy | 0.067 | **0.063** | 0.918 | **0.925** |
| acousticness | 0.074 | **0.064** | 0.875 | **0.923** |
| loudness | 1.65 dB | 1.65 dB | 0.832 | **0.845** |
| instrumentalness | 0.099 | **0.085** | 0.714 | **0.813** |
| danceability | 0.091 | **0.082** | 0.726 | **0.781** |
| valence | 0.121 | **0.110** | 0.747 | **0.779** |
| speechiness | 0.029 | 0.029 | 0.757 | **0.774** |
| liveness | 0.224 | **0.088** | 0.694 | **0.772** |
| duration_ms | 2974 ms | 2974 ms | 0.994 | 0.994 |
| tempo | 20.8 BPM | 20.8 BPM | 0.182 | 0.182 |

| Categorical | Before | Now |
|---|---|---|
| key (tonic correct) | 0.38 | **0.485** |
| key + mode exact | 0.36 | **0.470** |
| mode | 0.71 | 0.667 |

Valence, the feature the alignment gap depends on, now correlates at 0.779
with Spotify's value, with a mean error of 0.11 on a 0–1 scale.

## 6. Approaches tested and rejected

Rejected results matter for the thesis too, because they show the adopted
design was chosen on evidence rather than by default.

| Idea | Result | Decision |
|---|---|---|
| Tempo predicted by a VGGish regression | Corr 0.416, but only by squeezing almost every song into 105–140 BPM | Rejected: unusable as a per-song reading |
| Fold tempo into a fixed "plausible" band (e.g. 80–160 BPM) | Corr fell to −0.076 | Rejected: breaks correct fast/slow estimates |
| Use a VGGish tempo estimate to pick between half, normal and double tempo | Corr 0.182 → 0.142 | Rejected |
| Mode classifier on VGGish embeddings, trained on 491k labels | 0.667 on real songs | Rejected: no better than chroma, below always-major |
| Liveness regression on VGGish alone | Weaker than VGGish + PANNs | Superseded |
| Essentia's own VGGish model | Cosine similarity only ~0.56 with the embeddings the models were trained on | Rejected in favour of Google's official TF-Hub VGGish (~0.97) |

## 7. Mode and tempo

These are the two features calibration could not meaningfully improve.
Both were tested further, and both appear to be at the limit of what the
available data supports.

**Mode (major/minor).** On the validation songs, 74.2% are in a major key,
so always answering "major" scores 0.742. The chroma method scores 0.667.
Four further attempts:

| Attempt | Result |
|---|---|
| Mode classifier on VGGish embeddings (491k labels) | 0.667 on real songs |
| Mode classifier on MERT embeddings, a music-specific model (held-out, 200k rows) | 0.681, against a 0.674 always-major baseline on the same rows |
| 24-class key+mode classifier on MERT (held-out) | mode 0.715, key+mode exact 0.450, tonic 0.480: no better than chroma + Temperley on real songs (0.470 / 0.485) |
| Biasing the chroma decision toward major, threshold tuned by leave-one-out on the 66 songs | 0.697–0.742 depending on chroma variant; the tuned threshold drifts toward "always major" |
| Resolving relative-key swaps from the bass register (C1–C4 chroma), several variants, tuned settings scored leave-one-out | Mode 0.682 at best (one song better); key tonic up to 0.545, but that's 4 songs of 66, within noise |

Even a music-specific embedding trained on 200k labelled songs barely beats
the major-key base rate. That suggests Spotify's mode labels themselves are
only weakly recoverable from audio, not that our method is missing
something. The chroma estimate is kept because it responds to the actual
song. A constant "major" answer would score higher but carry no
information about the song.

Looking at *how* mode fails explains why. Of the 22 mode errors, 10 are
**relative-key swaps**: C major and A minor, for example, use exactly the
same seven notes and differ only in which note feels like "home". The method
also under-predicts minor (17% predicted vs 26% actual), so biasing it
toward major cannot help. The bass register was the natural place to find
the home note, and it didn't resolve the swaps reliably either.

What the pipeline does instead is **report its own uncertainty**. It returns
the runner-up key and the correlation margin between the winner and the
runner-up (`key_confidence`). On the 66 songs, mode was right 73% of the
time when the margin was at or above the median (about 0.10) and 61% below
it. When the margin is below 0.10, the demo shows "Low confidence — could
also be …" next to the runner-up, which is very often the relative key.

**Tempo.** The raw correlation is weak (0.182), but the error is almost
entirely *octave confusion*: 26% of songs are detected at exactly half or
double Spotify's tempo. Scored octave-tolerantly, the correlation is 0.939.
The beat periodicity is found correctly; choosing *which* octave is the
unsolved part. Attempts to choose the octave automatically:

| Attempt | Result |
|---|---|
| Fold into a fixed band (80–160 BPM) | Corr −0.076 (worse) |
| Pick the octave closest to a VGGish tempo estimate | Corr 0.142 (worse) |
| Tempo regression on MERT embeddings (held-out) | Right octave only 84.5% of the time, corr 0.254: not reliable enough to override the detector |

Octave ambiguity is a known open problem in music information retrieval,
because a song's "felt" tempo is partly a labelling convention. The
detector is kept as is. For the demo, the practical option is presentation
rather than calibration: show the detected tempo alongside its half and
double values.

## 8. Does the estimation error matter for popularity prediction?

The demo exists to predict the popularity of a new song, so the real test
of the feature estimates is whether they change that prediction. For each of
the 66 validation songs, the deployed popularity model (XGBoost,
alignment-augmented) was run twice: once on Spotify's real audio features
and once on Pipeline 2's estimates. Lyric sentiment was the same real VADER
score both times, so any difference comes from audio estimation alone
(`scripts/evaluate_prediction_sensitivity.py`).

| Measure | Result |
|---|---|
| Mean absolute shift in predicted popularity | **2.65 points** (median 1.99, max 8.54) |
| Mean signed shift | +0.09 points (no systematic bias) |
| Correlation between the two sets of predictions | 0.817 |
| Prediction error vs actual popularity, real features | RMSE 18.14 |
| Prediction error vs actual popularity, estimated features | RMSE 18.38 |

The popularity model's own error on its test set is about 15 points of
RMSE (18 on these 66 songs). Replacing Spotify's features with Pipeline 2's
estimates adds about 2.6 points of movement per song and barely changes
the error against actual popularity (18.14 → 18.38). **The feature
estimates are accurate enough that the popularity model's own uncertainty,
not feature estimation, dominates the demo's prediction error.**

Swapping in one estimated feature at a time shows which estimates matter
most:

| Feature estimated | Mean shift (points) |
|---|---|
| loudness | 1.57 |
| danceability | 1.16 |
| instrumentalness | 0.92 |
| energy | 0.65 |
| valence | 0.58 |
| acousticness | 0.57 |
| liveness | 0.53 |
| tempo | 0.39 |
| speechiness | 0.39 |
| key | 0.25 |
| mode | 0.12 |
| duration_ms | 0.11 |

Tempo and mode, the two features calibration could not improve (Section
7), have almost no effect on the prediction (0.39 and 0.12 points), so
their weakness does not limit the demo. Loudness contributes the most
shift, despite a 0.845 correlation, because the model is sensitive to it.
Any further calibration effort would be best spent there.

## 9. Limitations

- **Exact replication is not possible.** Spotify computed its values with a
  proprietary model on studio masters. The validation audio comes from
  YouTube, with different mastering, edits and encodings. Some residual
  error is therefore noise from the audio source, not model error.
- **The validation sample is small (66 songs).** Between the first 18 and
  the full 66 songs, three features' correlations moved by more than 0.15.
  Differences of about ±0.03 between two runs should not be over-read.
- **The estimates are correlated stand-ins, not ground truth.** For new
  songs in the demo, the alignment gap is computed from *estimated* valence
  (MAE 0.11). The popularity model itself is trained on real Spotify
  values.
- **Embedding match.** The learned features rely on the local VGGish and
  PANNs embeddings matching the precomputed training embeddings. They
  match closely on average (PANNs cosine similarity 0.961) but not for
  every song (minimum 0.676).

## 10. Reproducing the results

From WSL2 (Ubuntu-24.04), at the project root:

```bash
# 1. Compress the PANNs training embeddings (streams the 4 GB file)
uv run python scripts/build_panns_pca.py

# 2. Train the eight regression heads (also scores them on the validation songs)
uv run python scripts/train_vggish_ridge.py "/mnt/c/Users/User/Downloads/songs(1).csv"

# 3. End-to-end validation on the real songs (~70 min)
uv run python scripts/validate_pipeline_accuracy.py \
    --per-song-csv data/validation_songs/per_song_results.csv
```

The same steps run in a GPU-enabled Docker container on any Windows PC
with an NVIDIA card, without setting up WSL2 by hand. See
[gpu_calibration_setup.md](gpu_calibration_setup.md). The popularity
sensitivity check (Section 8) is:

```bash
uv run python scripts/evaluate_prediction_sensitivity.py "/path/to/songs(1).csv"
```

The validation songs themselves come from
`scripts/download_validation_songs.py`, which appends to the existing
manifest, so the sample can be enlarged for a tighter estimate.
