# Audio Feature Extraction Pipeline — Setup

`api/pipelines/audio_feature_extraction_pipeline.py` extracts audio-side features
(mood/valence-arousal via pretrained models, plus handcrafted acoustic descriptors)
to pair against the lyric-side sentiment (VADER) for the emotional-alignment
feature engineering described in the thesis.

It depends on two libraries:

| Library               | Purpose                                                                 | Platform                |
|------------------------|--------------------------------------------------------------------------|--------------------------|
| `librosa`               | General acoustic features — MFCCs, chroma, tempo, spectral descriptors  | Cross-platform (Windows, WSL, macOS) |
| `essentia-tensorflow`   | Pretrained mood/valence-arousal TensorFlow models                       | **Linux only** — no Windows wheel exists |

Because `essentia-tensorflow` has no Windows build (its C++ core has never
shipped a Windows wheel, and there's no sdist to compile from source), this
project runs the pipeline under **WSL** rather than natively on Windows.
Everything else — the FastAPI service, Postgres/Alembic, the frontend,
notebooks — is unaffected and keeps running on Windows as normal.

## Architecture

There is **one copy of the project**, not two. WSL mounts the Windows drive
directly, so `/mnt/d/Projects/thesis-project` inside WSL and
`D:\Projects\thesis-project` in Windows are the same files on the same disk —
edits in one are instantly visible in the other, no syncing required.

The only thing that has to be separate is the **virtual environment**: a
`.venv` is a folder of OS-specific compiled packages, and a Windows-built one
can't be reused on Linux (or vice versa). So there are two:

```
D:\Projects\thesis-project\              (one folder, one git repo)
│
├── pyproject.toml, uv.lock              ← shared — single lockfile, resolved
│                                           for both platforms at once
├── api/, app/, notebooks/...            ← shared source
│
├── .venv/                               ← Windows venv (FastAPI, notebooks, etc.)
└── (WSL venv lives outside this folder — see below)
```

`pyproject.toml` marks the Linux-only dependency with a PEP 508 environment
marker:

```
"essentia-tensorflow==2.1b6.dev1389 ; sys_platform == 'linux'"
```

`uv sync` on Windows reads this and skips it (no error). `uv sync` inside WSL
reads the same line and installs it. One `pyproject.toml`, one `uv.lock`, no
duplicate bookkeeping.

The WSL-side venv is **not** placed inside the shared `D:\...` folder — if it
were, it would collide with the Windows `.venv` at the same path. Instead it's
redirected to a Linux-native location via `UV_PROJECT_ENVIRONMENT`, so the two
never touch:

```
Windows .venv           → D:\Projects\thesis-project\.venv
WSL venv (via env var)  → ~/.venvs/thesis-project   (inside WSL's own filesystem)
```

## One-time WSL setup

Distro used: **Ubuntu-24.04** (already has `uv` and a managed Python 3.13.14
install — matches the project's `.python-version` pin, which is also a shared
file WSL reads automatically).

```bash
# 1. Open a WSL shell
wsl -d Ubuntu-24.04

# 2. Go to the project via the Windows-drive mount (same files as D:\...)
cd /mnt/d/Projects/thesis-project

# 3. Point uv's venv at a Linux-native path so it can't collide with the
#    Windows .venv. Persist it so every future shell picks it up.
echo 'export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/thesis-project' >> ~/.bashrc
source ~/.bashrc

# 4. Resolve + install (installs essentia-tensorflow here; skipped on Windows)
uv lock
uv sync

# 5. Verify
uv run python -c "import essentia.standard as es; print(es.MonoLoader)"
```

The `essentia-tensorflow` wheel is ~290MB and pulls in a bundled TensorFlow
runtime; you'll see benign `tensorflow ... Could not load dynamic library
libcudart.so` warnings on import if you don't have an NVIDIA GPU/CUDA set up —
TensorFlow just falls back to CPU. Nothing to fix there.

## Running the pipeline

Always run it from WSL, since that's the only environment where
`essentia-tensorflow` can load:

```bash
cd /mnt/d/Projects/thesis-project
uv run python api/pipelines/audio_feature_extraction_pipeline.py
```

Or, without opening a WSL shell yourself, from a Windows terminal:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "cd /mnt/d/Projects/thesis-project && uv run python api/pipelines/audio_feature_extraction_pipeline.py"
```

Output (extracted features) should land wherever the rest of the data
pipeline expects them (e.g. alongside `notebooks/dataset/`, or written to
Postgres via the API) so the notebooks and model training can pick them up —
those steps keep running on Windows as usual.

## Everyday workflow

- Edit code in Windows/VS Code as normal — nothing changes there.
- `uv sync`/`uv add` from Windows continues to manage the Windows `.venv` and
  silently ignores the Linux-only dependency.
- Only reach for WSL when you need to actually **run or test** the audio
  feature extraction pipeline (or anything else that imports `essentia`).
- If you add more dependencies to `pyproject.toml` from either side, run
  `uv lock` once (from either OS — it's the same file) and then `uv sync` on
  whichever side(s) need the updated packages.
