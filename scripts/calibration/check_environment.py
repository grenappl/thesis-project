"""Pre-flight check for calibration: is the GPU visible, do the libraries
import, and is every data/model file in place (and intact, if the bundle
manifest from bundle_calibration_data.ps1 is present)?

    python scripts/calibration/check_environment.py          # sizes only
    python scripts/calibration/check_environment.py --hash   # also SHA-256 (slower)

Exits non-zero if anything required is missing, so `run_calibration.sh all`
stops before spending an hour on a broken setup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

OK, WARN, FAIL = "  [ok]  ", "  [warn]", "  [FAIL]"
problems = 0


def report(status: str, message: str) -> None:
    global problems
    problems += status == FAIL
    print(f"{status} {message}")


def check_libraries() -> None:
    print("Libraries / GPU")
    try:
        import torch
        if torch.cuda.is_available():
            report(OK, f"PyTorch {torch.__version__}, CUDA GPU: {torch.cuda.get_device_name(0)}")
        else:
            report(WARN, f"PyTorch {torch.__version__} sees no GPU; PANNs will run on CPU (slower). "
                         "Check the NVIDIA driver and that Docker was started with GPU access.")
    except Exception as exc:
        report(FAIL, f"torch import failed: {exc}")
    for module in ("tensorflow", "tensorflow_hub", "librosa", "sklearn", "xgboost", "panns_inference"):
        try:
            __import__(module)
            report(OK, module)
        except Exception as exc:
            report(FAIL, f"{module} import failed: {exc}")


def check_files(do_hash: bool) -> None:
    print("\nData and models")
    songs_csv = Path(os.environ.get("SONGS_CSV", "data/songs.csv"))
    required = [
        songs_csv,
        Path("data/kaggle_embeddings/vggish_embeddings.npz"),
        Path("data/kaggle_embeddings/panns_embeddings.npz"),
        Path("models/panns/Cnn14_mAP=0.431.pth"),
        Path("data/validation_songs/manifest.csv"),
        Path("models/popularity/xgboost_alignment_augmented.json"),
        Path("models/popularity/model_metadata.json"),
    ]
    produced = [  # rebuilt by the pca/train/cache stages, so only a warning
        Path("data/kaggle_embeddings/panns_pca256.npz"),
        Path("models/panns/panns_pca256.joblib"),
        *(Path(f"models/regression_heads/{t}.joblib") for t in (
            "valence", "acousticness", "danceability", "energy",
            "speechiness", "instrumentalness", "loudness", "liveness")),
        Path("data/validation_songs/feature_cache.npz"),
    ]
    for p in required:
        report(OK if p.is_file() else FAIL, f"{p}" + ("" if p.is_file() else " (missing)"))
    for p in produced:
        report(OK if p.is_file() else WARN, f"{p}" + ("" if p.is_file() else " (missing; created by a later stage)"))

    manifest = Path("data/validation_songs/manifest.csv")
    if manifest.is_file():
        import pandas as pd
        ids = pd.read_csv(manifest)["id"]
        missing = [i for i in ids if not Path(f"data/validation_songs/{i}.mp3").is_file()]
        report(OK if not missing else WARN,
               f"validation songs: {len(ids) - len(missing)}/{len(ids)} audio files present")

    bundle = Path("data/bundle_manifest.json")
    if not bundle.is_file():
        report(WARN, "no data/bundle_manifest.json; skipping integrity check")
        return
    entries = json.loads(bundle.read_text(encoding="utf-8-sig"))
    bad = []
    for entry in entries:
        p = Path(entry["path"])
        if not p.is_file() or p.stat().st_size != entry["size"]:
            bad.append(entry["path"])
        elif do_hash:
            h = hashlib.sha256()
            with p.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            if h.hexdigest().upper() != entry["sha256"].upper():
                bad.append(entry["path"])
    what = "size + SHA-256" if do_hash else "size"
    if bad:
        report(FAIL, f"{len(bad)} bundled file(s) missing or damaged ({what}): {bad[:5]}")
    else:
        report(OK, f"all {len(entries)} bundled files intact ({what})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--hash", action="store_true", help="verify SHA-256 of bundled files, not just sizes")
    args = parser.parse_args()
    print(f"Python {sys.version.split()[0]}, working dir {Path.cwd()}\n")
    check_libraries()
    check_files(args.hash)
    print(f"\n{'Ready.' if problems == 0 else f'{problems} problem(s) found - fix these before calibrating.'}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
