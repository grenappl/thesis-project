# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project purpose

Undergraduate CS thesis: *Lyric and Audio Emotional Alignment as a Predictor of Spotify Track Popularity*. The research question is whether the alignment (or mismatch) between a song's lyrical sentiment and its musical valence helps predict Spotify popularity. The central, load-bearing formula for the whole thesis:

```
valence_normalized = (2 * valence) - 1        # rescales Spotify's [0,1] valence to [-1,1]
alignment_gap       = lyric_sentiment - valence_normalized
```

Sign convention: **positive** `alignment_gap` means the lyrics read more positive than the music sounds; **negative** means the reverse. The subtraction order is `lyric_sentiment - valence_normalized`, not the reverse — flipping it silently inverts every alignment feature with nothing to catch it.

Full narrative and dataset-preparation details: [README.md](README.md), [docs/cleaning_process.md](docs/cleaning_process.md). Full feature-extraction methodology (what follows here is the condensed architectural version): [feature_extraction.md](feature_extraction.md).

## Commands

```bash
# Backend (from project root — api is a package, must run from root)
uv sync                                          # install deps (Windows: skips essentia-tensorflow)
uv run pytest tests                              # full test suite
uv run pytest tests/pipelines/demo/test_alignment.py::test_alignment_gap_matches_known_reference_value   # single test
uv run uvicorn api.main:app --reload --port 8010 # dev server (NOT `fastapi dev` — see gotcha below; port 8010 not the uvicorn default 8000, to avoid conflicts with other projects)
uv run alembic -c api/alembic.ini upgrade head   # run DB migrations

# Frontend (from app/)
npm run dev / build / start / lint

# One-shot launcher: preflight-checks WSL2 + model files, starts backend + frontend
# together in the current terminal (no popup windows)
.\run-demo.ps1
```

**Gotcha**: `uv run fastapi dev api/main.py` crashes on Windows terminals with a non-UTF8 codepage (rich_toolkit's startup banner has an emoji). Use `uv run uvicorn api.main:app --reload` instead — same dev server, no crash.

## Architecture

### Two independent feature-extraction pipelines that must never import from each other

```
api/pipelines/
├── training/   Pipeline 1 — reads audio descriptors + lyrics straight from the
│               Spotify dataset (songs(1).csv, 491,632 rows). No audio processing.
│               Windows-native. Currently unused/disabled by user request — logic
│               duplicated in notebooks/3_sentiment_analysis.ipynb instead — but
│               the module is kept as-is, don't remove without asking.
└── demo/       Pipeline 2 — extracts the same descriptor set from a raw audio
                file for a NEW song never scored by Spotify (the actual demo-app
                path). Two inputs: audio (required) + lyrics (optional).
```

Both pipelines independently implement lyric sentiment (VADER) and the alignment-gap formula — this duplication is deliberate, not an oversight. Pipeline 1 reads real Spotify `valence`; Pipeline 2 has no real Spotify value for a new song, so it uses its own estimated `valence` instead. Never share code between them.

### Pipeline 2 covers all 12 Spotify audio descriptors, at three confidence tiers

- **Real, established DSP techniques** (not proxies): `tempo`, `key`/`mode` (Librosa chroma + Krumhansl-Schmuckler key-profile correlation), `duration_ms`. `tempo`'s real-song correlation (0.182) is weak, but a follow-up octave-tolerant analysis found 26% of errors are exact half/double-tempo confusion — correcting for octave alone gets to 0.939 correlation, an oracle ceiling with no ground-truth-free fix found (blind range-folding made things worse, tested and rejected) — see feature_extraction.md's "Tempo" section. A VGGish alternative was tested (end-to-end corr 0.416) and deliberately not adopted — it wins only by compressing predictions into a narrow band, a worse fit for a single-song display feature than a noisier, wider-ranging estimate.
- **Fit against real Spotify ground truth**: `valence`/`acousticness`/`instrumentalness`/`danceability`/`energy`/`speechiness`/`loudness` — a regression head (Ridge for instrumentalness/energy/speechiness, `HistGradientBoostingRegressor` for valence/acousticness/danceability/loudness — see `_TARGET_MODELS` in `scripts/train_vggish_ridge.py`) on Google's official TF-Hub VGGish embedding (`vggish_tfhub.py`), trained on the Kaggle precomputed-embeddings dataset against the real Spotify columns. Runs as an **isolated subprocess** — see below for why. `danceability`/`energy`/`speechiness` moved here from the proxy tier after real-song validation showed this beats their old hand-built implementations (corr 0.353→0.726, 0.570→0.916, 0.206→0.709). `loudness` moved here from the DSP tier — Essentia's `ReplayGain` was replaced entirely after VGGish, tried for loudness for the first time, blew past it (corr 0.605→0.832). `valence`/`acousticness`/`danceability` were later upgraded from Ridge to GBM (corr 0.712→0.747, 0.858→0.875, unchanged at 0.726 but MAE improved) after a Ridge-shrinkage bias was diagnosed via a single-song spot-check and confirmed via an extreme-value-tail metric. Old implementations (`speechiness.py`, `calibrate_speechiness.py`, the calibration-audio scripts, the frontend "uncalibrated" badge) were removed entirely, not kept alongside.
- **Proxy**, unvalidated against real Spotify ground truth: `liveness` (PANNs CNN14 crowd-noise-class probability — a general AudioSet tagger repurposed for this, not a dedicated model). The one remaining feature in this tier.

`feature_extraction.md`'s "Known limitations" section is the authoritative, current list of what's calibrated vs. proxy vs. unvalidated — check it before assuming any Pipeline 2 number is accurate.

### The Windows/WSL2 split (this is the single most important operational fact about this repo — but Essentia's part of the reason is now dormant, see below)

Essentia (`essentia-tensorflow`) has no Windows wheel. **As of the loudness move above, nothing in `pipeline.py` calls Essentia at all anymore** — `essentia_features.py` and its test are kept but unused, not deleted. The one remaining thing tying Pipeline 2 to WSL2 is `panns_inference` (liveness) shelling out to `wget` on first import, which doesn't exist on Windows — a narrow, likely-fixable quirk, not a hard platform wall like Essentia's. Whether to actually pursue dropping the WSL2 requirement is a deliberate, separate decision, not made yet. Until/unless that happens: **the whole of Pipeline 2 still runs in WSL2** as one process — originally to avoid a subprocess round-trip per feature reaching Essentia; now more by inertia than necessity, but still how it runs. Pipeline 1 stays Windows-native regardless (no such constraint, and a WSL hop for 491k rows would be pure overhead).

One exception within that one-process rule: `vggish_tfhub.py` (valence/acousticness/instrumentalness/danceability/energy/speechiness/loudness) always runs as its *own* subprocess, spawned by `pipeline.py`, even though everything is already inside WSL2. Essentia bundles its own TensorFlow runtime; loading a standalone `tensorflow`/`tensorflow_hub` in the same process crashes (`Check failed: ... ALREADY_EXISTS: Op with name Bitcast`, a C++ op-registry collision). Never import `vggish_tfhub` from a module that also imports `essentia` — even though nothing currently does, this constraint stays real if `essentia_features.py` is ever called again.

The FastAPI backend (`api/services/demo_feature_extraction_service.py`) runs on Windows and shells out to `wsl.exe` per request — a background thread + `subprocess.Popen`, **not** `asyncio.create_subprocess_exec`. That's a fixed bug, not a style choice: under `uvicorn --reload` on Windows, the reload worker doesn't reliably get the ProactorEventLoop that Windows needs for asyncio subprocess pipes, and fails with a bare unhelpful `NotImplementedError`. Don't "simplify" this back to asyncio subprocess.

Full setup, one-time WSL2 steps, and why: [PIPELINE_SETUP.md](PIPELINE_SETUP.md).

### Demo-app request flow

`app/app/test-feature-extraction/page.tsx` (upload form + live log panel + results, in a fixed 3-region grid layout) → `POST /demo/extract-features/stream` (NDJSON stream of `{type: "log"|"result"|"error"}` events — headers are already sent once streaming starts, so every failure mode has to arrive as an `error` event, never a raised exception) → `DemoFeatureExtractionService` writes the upload + optional lyrics to temp files, shells into WSL2, parses `api.pipelines.demo.pipeline`'s stdout for a JSON block (interleaved with panns_inference's stray prints and TF/Essentia log noise — `_extract_json` specifically scans for the pretty-printed block, doesn't just `json.loads(stdout)`).

### Layered `api/` convention (router → service → repository → model)

Documented in [api/README.md](api/README.md); applies to the DB-backed `tracks` router/service/repository/model stack. `demo_features` (router + service) doesn't follow the repository/model layers — there's no DB involved, it's pure subprocess orchestration. Note the `Track` DB model's `audio_sentiment`/`emotional_alignment` field names predate and don't match Pipeline 2's `lyric_sentiment`/`alignment_gap` naming — they're not currently wired together.

### Models and downloaded data (all gitignored, fetched on demand)

- `models/panns/` — PANNs CNN14 checkpoint for liveness. Download commands in `feature_extraction.md`. `models/essentia/` needs nothing at all now — Essentia has zero callers in the pipeline (see "The Windows/WSL2 split" above). Any `.pb` files still there (from pre-VGGish implementations) are dead weight (~292MB+) — safe to delete, not referenced by any code path anymore.
- `models/vggish_ridge/` — the seven regression models (valence, acousticness, instrumentalness, danceability, energy, speechiness, loudness — mixed Ridge/GBM, see `_TARGET_MODELS`), trained by `scripts/train_vggish_ridge.py` against `data/kaggle_embeddings/`.
- `data/kaggle_embeddings/` — precomputed VGGish/PANNs/MERT/mel-spectrogram embeddings for the same 491,632 tracks as the training dataset (source: Kaggle `serkantysz/550k-spotify-songs-audio-lyrics-and-genres` derivative), being used to validate/recalibrate Pipeline 2's proxy features against real Spotify values at full dataset scale instead of one-off manual spot checks.
- `data/validation_songs/` — real audio for known dataset tracks, downloaded via `scripts/download_validation_songs.py` (yt-dlp, confidence-scored matching) so `scripts/validate_pipeline_accuracy.py` can compare Pipeline 2's *actual end-to-end output* against real Spotify values, not just held-out embedding regression numbers. This caught two real bugs (`loudness` had an inverted sign, `tempo`'s estimator was collapsing onto its default prior) — see feature_extraction.md's "Real-song validation" section for current per-feature accuracy and how to extend the sample.
