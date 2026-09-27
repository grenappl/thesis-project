# Running the Audio Calibration on a GPU PC (Docker)

This guide recreates the Pipeline 2 calibration on a Windows PC with an
NVIDIA GPU. Everything runs inside one Docker container, so you don't need
to set up WSL2, Ubuntu or Python yourself. Docker Desktop handles that.

The container holds only the Python environment. Your project folder is
mounted into it, so every result (trained models, logs, CSVs) lands directly
in the project folder on the PC.

## What the GPU speeds up (and what it doesn't)

| Stage | Runs on | Time on the laptop (CPU) |
|---|---|---|
| `pca`: compress the PANNs training embeddings | CPU | ~8 min |
| `train`: fit the 8 regression heads on 491k songs | CPU, all cores | ~70 min (12 cores) |
| `cache`: per-song embeddings for experiments | **GPU** (PANNs) | ~70 min |
| `validate`: full pipeline on every validation song | **GPU** (PANNs) | ~70 min |
| `sensitivity`: popularity model, real vs estimated features | CPU | ~1 min |

The GPU accelerates PANNs, the heaviest per-song step. VGGish deliberately
stays on the CPU, because TensorFlow and PyTorch loading two different CUDA
library versions in one container is a common crash. Training uses
scikit-learn's gradient boosting, which is CPU-only, so it scales with the
number of CPU cores rather than the GPU.

## What you need on the GPU PC

- Windows 10 (22H2 or later) or Windows 11.
- An NVIDIA GPU with a recent driver (Game Ready or Studio driver from
  nvidia.com).
- 16 GB RAM or more. PCA and training hold about 6–8 GB in memory.
- About 20 GB of free disk space: ~10 GB for the Docker image, ~5 GB for
  the project and data.

## Step 1: Bundle everything on the laptop

Plug in a USB drive (about 6 GB free) and run this from the project folder
in PowerShell:

```powershell
.\scripts\calibration\bundle_calibration_data.ps1 -Destination E:\thesis-project
```

Replace `E:` with the drive letter. The script copies:

- the project code, including any uncommitted changes;
- the Kaggle embeddings (VGGish, PANNs, PANNs-PCA);
- the 66 validation songs with their manifest and caches;
- the PANNs checkpoint, regression heads and popularity model;
- the Spotify dataset CSV, saved as `data\songs.csv`.

It skips `.venv`, `node_modules`, the unused Essentia models, and the MERT
and mel embeddings. Add `-IncludeExtraEmbeddings` if you want those too.
Finally it writes `data\bundle_manifest.json`, a checksum for every data
file, so the GPU PC can confirm nothing was damaged in the copy.

To see what would be copied without copying anything, add `-DryRun`.

## Step 2: Copy to the GPU PC

Copy the `thesis-project` folder from the USB drive to a local disk on the
GPU PC, for example `C:\thesis-project`. Don't run it from the USB drive;
Docker reads files from it too slowly.

## Step 3: Install the NVIDIA driver and Docker Desktop

1. Install or update the NVIDIA driver from nvidia.com, then restart.
2. Install **Docker Desktop** from docker.com. Keep the default setting
   **"Use the WSL 2 based engine"**. Docker installs and manages WSL2
   itself, so you don't need to install Ubuntu.
3. Restart when asked, then start Docker Desktop and wait until it says
   "Engine running".
4. Check that Docker can see the GPU:

   ```powershell
   docker run --rm --gpus all ubuntu nvidia-smi
   ```

   You should see a table naming your GPU. If you get an error instead, see
   Troubleshooting below.

## Step 4: Build the calibration image (once)

In PowerShell, from the project folder:

```powershell
cd C:\thesis-project
docker compose -f docker-compose.calibration.yml build
```

The first build takes about 15–30 minutes, because it downloads PyTorch
and TensorFlow (several GB). Later builds reuse that and take seconds,
unless `pyproject.toml` or `uv.lock` change.

## Step 5: Check the environment

```powershell
docker compose -f docker-compose.calibration.yml run --rm calibrate check
```

This confirms that PyTorch sees the GPU, that every library imports, and
that every data and model file is present. It should end with **Ready.**
To also verify every file's checksum against the laptop's copy (slower):

```powershell
docker compose -f docker-compose.calibration.yml run --rm --entrypoint python calibrate scripts/calibration/check_environment.py --hash
```

## Step 6: Run the calibration

Every command below has the same form:

```powershell
docker compose -f docker-compose.calibration.yml run --rm calibrate <stages>
```

| To do this | `<stages>` |
|---|---|
| Recreate the whole calibration from scratch | `all` (check, pca, train, cache, validate, sensitivity) |
| Only re-score the current models on the real songs | `validate sensitivity` |
| Retrain the regression heads (e.g. after changing settings) | `train validate sensitivity` |
| Add 100 more validation songs, then re-score | `download 100 cache validate sensitivity` |
| Run the test suite | `test` |
| Run one experiment script from `scripts/calibration/` | `experiment relative_key_experiment.py` |

Several stages can be listed in one command; they run in order and stop at
the first failure. Each stage's output is saved to
`data/calibration_logs/<stage>.log` and also shown in the terminal.

**Before running `all` or `train`,** back up `models\regression_heads\`
and `models\panns\panns_pca256.joblib`. Those stages overwrite them.

**`cache` is resumable and skips songs it already has.** To recompute from
scratch, delete `data\validation_songs\feature_cache.npz` first.

## Step 7: Where the results are

| Result | Location |
|---|---|
| Per-stage logs, including the accuracy tables | `data/calibration_logs/*.log` |
| Trained regression heads | `models/regression_heads/*.joblib` |
| PANNs PCA | `models/panns/panns_pca256.joblib` |
| Per-song real vs estimated values | `data/validation_songs/per_song_results.csv` |
| Popularity sensitivity per song | `data/validation_songs/prediction_sensitivity.csv` |
| Experiment outputs | `data/calibration_outputs/` |

To use new models on the laptop, copy `models\regression_heads\` and
`models\panns\panns_pca256.joblib` back. The PCA and the heads must always
be copied together, because the heads are trained on that exact PCA.

## Troubleshooting

- **`docker run --gpus all ... nvidia-smi` fails, or `check` says "sees
  no GPU".** Update the NVIDIA driver and restart. In Docker Desktop, go to
  Settings → General and make sure the WSL 2 engine is on. Then restart
  Docker Desktop.
- **`$'\r': command not found`.** The shell scripts got Windows line
  endings, which happens if the files were edited or checked out on
  Windows. The repo's `.gitattributes` prevents this for git checkouts. To
  fix a copied folder, open `scripts\calibration\run_calibration.sh` in VS
  Code and switch "CRLF" to "LF" in the bottom-right corner.
- **A stage is killed partway, or Docker Desktop becomes unresponsive.**
  The Linux VM ran out of memory. Create `C:\Users\<you>\.wslconfig`
  containing the lines below, then run `wsl --shutdown` and restart Docker
  Desktop:

  ```
  [wsl2]
  memory=12GB
  ```

- **Very slow file access.** The project is probably on a USB or network
  drive. Move it to a local disk (Step 2).
