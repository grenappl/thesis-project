# Feature Extraction Pipelines — Setup

## Setting up on a new machine (fresh clone)

Everything below in one order, for when you're doing this on a different
computer than the one this was developed on. Each step links to the section
with the full explanation.

1. **Prerequisites** (once per machine):
   - [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed
     on Windows: `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - WSL2 enabled with an Ubuntu distro: `wsl --install -d Ubuntu-24.04`
     (Windows may ask for a reboot the first time WSL2 itself is enabled).
     Any reasonably current Ubuntu works; this project was set up against
     24.04.
   - `uv` installed *inside* WSL2 too — open `wsl -d Ubuntu-24.04` and run
     `curl -LsSf https://astral.sh/uv/install.sh | sh`.

2. **Clone the repo** — anywhere under the Windows filesystem (e.g.
   `D:\Projects\thesis-project`), since WSL2 will reach it through
   `/mnt/<drive>/...` rather than needing its own copy (see
   [Architecture](#architecture)).

3. **Windows-side install** (Pipeline 1 + everything in Pipeline 2 except
   Essentia), from the project root in PowerShell:
   ```powershell
   uv sync
   ```

4. **WSL2-side install** (adds Essentia) — see [One-time WSL
   setup](#one-time-wsl-setup) below for the full steps and *why* the
   `UV_PROJECT_ENVIRONMENT` env var matters. Short version, from inside
   `wsl -d Ubuntu-24.04`:
   ```bash
   cd /mnt/d/Projects/thesis-project   # adjust drive/path to match your clone
   echo 'export UV_PROJECT_ENVIRONMENT=$HOME/.venvs/thesis-project' >> ~/.bashrc
   source ~/.bashrc
   uv sync
   ```

5. **Download the model checkpoints** — not committed to git (`models/` is
   gitignored; a few MB + 327MB of binary weights don't belong in version
   control). From inside WSL2:
   ```bash
   cd /mnt/d/Projects/thesis-project   # adjust to match your clone
   mkdir -p models/panns
   curl -sSL -o "models/panns/Cnn14_mAP=0.431.pth" "https://zenodo.org/records/3987831/files/Cnn14_mAP=0.431.pth?download=1"
   ```
   (The PANNs file is 327MB — takes a few minutes depending on your
   connection. Verify it downloaded intact: `md5sum "models/panns/Cnn14_mAP=0.431.pth"`
   should read `541141fa2ee191a88f24a3219fff024e`.)

6. **Train the VGGish regression models** (valence/acousticness/
   instrumentalness/danceability/energy/speechiness/loudness) — not a
   download, a one-time training run against the Kaggle precomputed-
   embeddings dataset (see [feature_extraction.md](feature_extraction.md)
   for what this fixes and why). From WSL2:
   ```bash
   uv run python scripts/train_vggish_ridge.py \
     data/kaggle_embeddings/vggish_embeddings.npz "/mnt/c/Users/User/Downloads/songs(1).csv"
   ```

7. **Verify everything**:
   ```powershell
   uv run pytest tests            # Windows
   ```
   ```bash
   uv run pytest tests            # inside WSL2 (Essentia's guard test self-skips here)
   ```

That's the whole setup. Everything past this point explains the *why*
behind each of those steps.

---

There are two independent feature-extraction pipelines in `api/pipelines/`:

| Pipeline | Module | What it does | Platform |
|---|---|---|---|
| **1 — Training** | `api/pipelines/training/` | Reads audio descriptors + lyrics straight from the dataset, computes lyric sentiment (VADER) and the alignment gap | Windows (native) |
| **2 — Demo** | `api/pipelines/demo/` | Extracts the same descriptor set from a raw, user-submitted audio file (Librosa, PANNs, TF-Hub VGGish + a regression head per target) | **WSL2, though see below — this may no longer be strictly required** |

Full feature list, formulas, and library-to-descriptor mapping are documented in
[`feature_extraction.md`](feature_extraction.md). This file only covers *running*
the pipelines and the Windows/WSL2 split.

## Why Pipeline 2 runs under WSL2 (historically Essentia, now PANNs)

Essentia has no Windows wheel: its C++ core has never shipped a Windows build,
and there's no sdist to compile from source. So `essentia-tensorflow` is marked
Linux-only in `pyproject.toml`:

```
"essentia-tensorflow==2.1b6.dev1389 ; sys_platform == 'linux'"
```

`uv sync` on Windows skips it; `uv sync` inside WSL installs it. **But as of a
later change, `pipeline.py` no longer calls Essentia for anything** — loudness
(its last remaining job) moved to a VGGish-based regression after real-song
validation showed a large accuracy win (see feature_extraction.md's "Real-song
validation" section). This whole section is now about *why WSL2 was needed*,
not *why it's still needed*.

The one thing that still ties Pipeline 2 to WSL2: `panns_inference` (liveness)
shells out to `wget` on first import to fetch a small AudioSet label CSV, which
fails with `FileNotFoundError` on Windows (no `wget`). This is a narrow,
likely-fixable quirk (pre-download the file, or patch the fetch call), not a
hard C++ Windows-wheel gap like Essentia's — so dropping the WSL2 requirement
entirely may be realistic, but hasn't been pursued; this file still describes
the current, WSL2-based setup.

**The whole of Pipeline 2 runs in WSL2, not per-feature.** Librosa, PyTorch/
PANNs, and TF-Hub VGGish + regression are all individually capable of running
natively on Windows — but splitting the pipeline would still require a subprocess
round-trip per request to reach whichever piece needs WSL2 (historically Essentia,
now just PANNs' label download). Consolidating the whole extraction into a single
WSL2 call avoids that split: one process, one venv, one set of downloaded model
weights, one JSON result. Pipeline 1 (training) has no such constraint and stays
on native Windows, since a WSL2 hop for 491k dataset rows would only add overhead
for no benefit.

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
├── api/pipelines/training/              ← Pipeline 1 — Windows only
├── api/pipelines/demo/                  ← Pipeline 2 — WSL2 only
│
├── .venv/                               ← Windows venv (FastAPI, notebooks,
│                                           Pipeline 1, and Pipeline 2's code
│                                           import cleanly here too — only
│                                           calling the Essentia-backed
│                                           functions requires WSL2)
└── (WSL venv lives outside this folder — see below)
```

`uv sync` on Windows reads `pyproject.toml` and skips the Linux-only dependency
(no error). `uv sync` inside WSL reads the same line and installs it. One
`pyproject.toml`, one `uv.lock`, no duplicate bookkeeping.

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

Pipeline 2 also needs the PANNs `Cnn14_mAP=0.431.pth` checkpoint downloaded
separately (Essentia needs no checkpoint at all — it has zero callers in the
pipeline now) — see [`feature_extraction.md`](feature_extraction.md) for
links and expected paths.

## Running the pipelines

### Pipeline 1 (training) — Windows, from the project root

```powershell
uv run python -m api.pipelines.training.pipeline "C:\path\to\songs(1).csv" notebooks\data_filtered\songs_features.csv
```

### Pipeline 2 (demo) — always from WSL2

```bash
cd /mnt/d/Projects/thesis-project
uv run python -m api.pipelines.demo.pipeline path/to/uploaded_song.wav \
  --panns-checkpoint "models/panns/Cnn14_mAP=0.431.pth" \
  --vggish-ridge-models-dir models/vggish_ridge
```

Or, without opening a WSL shell yourself, from a Windows terminal:

```powershell
wsl -d Ubuntu-24.04 -- bash -lc "export UV_PROJECT_ENVIRONMENT=\$HOME/.venvs/thesis-project && cd /mnt/d/Projects/thesis-project && uv run python -m api.pipelines.demo.pipeline path/to/uploaded_song.wav"
```

**Gotcha:** `wsl -d ... -- bash -lc "..."` runs a *non-interactive* login
shell. Ubuntu's default `~/.bashrc` returns early for non-interactive
shells (before reaching the `UV_PROJECT_ENVIRONMENT` export added in step 4
above), so that line in `.bashrc` never actually takes effect for commands
invoked this way — `uv` would silently fall back to creating a *new* venv
at the project's own path, colliding with the Windows `.venv`. Always
`export UV_PROJECT_ENVIRONMENT=...` inline (as above) when driving WSL2
non-interactively from a script or from Windows; it's only safe to rely on
`.bashrc` inside an actual interactive `wsl -d Ubuntu-24.04` session.

## Running the tests

Pipeline 1's tests and most of Pipeline 2's tests (audio I/O, Librosa,
PANNs, the VGGish regression glue) run natively on Windows — only the
Essentia test is platform-gated (it asserts the graceful-failure message
on a machine without Essentia installed, and is skipped where Essentia
*is* installed). This tests `essentia_features.py` directly, not through
the pipeline — nothing in `pipeline.py` calls Essentia anymore (see
"Why Pipeline 2 runs under WSL2" above):

```powershell
uv run pytest tests
```

## Everyday workflow

- Edit code in Windows/VS Code as normal — nothing changes there. All of
  Pipeline 2's code imports cleanly on Windows too; only *calling* the
  Essentia-backed functions requires WSL2.
- `uv sync`/`uv add` from Windows continues to manage the Windows `.venv` and
  silently ignores the Linux-only dependency.
- Only reach for WSL2 when you need to actually **run** the demo pipeline
  end-to-end for a real uploaded song (or anything else that imports
  `essentia`).
- If you add more dependencies to `pyproject.toml` from either side, run
  `uv lock` once (from either OS — it's the same file) and then `uv sync` on
  whichever side(s) need the updated packages.
