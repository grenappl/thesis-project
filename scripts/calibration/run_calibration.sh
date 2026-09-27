#!/usr/bin/env bash
# Runs the Pipeline 2 calibration steps inside the GPU container (or any
# Linux/WSL2 shell with the project environment). Each stage logs to
# data/calibration_logs/<stage>.log as well as the terminal.
#
#   run_calibration.sh check                 environment + data check (default)
#   run_calibration.sh test                  pytest suite
#   run_calibration.sh download <N>          add N more validation songs (yt-dlp)
#   run_calibration.sh cache                 per-song embedding cache for experiments (GPU)
#   run_calibration.sh pca                   PANNs 2048->256 PCA from the Kaggle embeddings
#   run_calibration.sh train                 train the 8 regression heads (CPU, all cores)
#   run_calibration.sh validate              full pipeline on every validation song (GPU)
#   run_calibration.sh sensitivity           popularity prediction: real vs estimated features
#   run_calibration.sh all                   check pca train cache validate sensitivity
#   run_calibration.sh experiment <name.py>  any script in scripts/calibration/
#
# Several stages can be chained: run_calibration.sh pca train validate
set -euo pipefail
cd "$(dirname "$0")/../.."

SONGS_CSV="${SONGS_CSV:-data/songs.csv}"
LOGS=data/calibration_logs
mkdir -p "$LOGS" data/calibration_outputs

run() {  # run <stage> <command...>: tee output to the stage's log
    local stage=$1; shift
    echo "== $stage: $* ($(date +%H:%M:%S))" | tee "$LOGS/$stage.log"
    local start=$SECONDS
    "$@" 2>&1 | tee -a "$LOGS/$stage.log"
    echo "== $stage done in $(( (SECONDS - start) / 60 )) min" | tee -a "$LOGS/$stage.log"
}

[ $# -eq 0 ] && set -- check
[ "$1" = all ] && set -- check pca train cache validate sensitivity

while [ $# -gt 0 ]; do
    stage=$1; shift
    case "$stage" in
        check)       run check python scripts/calibration/check_environment.py ;;
        test)        run test python -m pytest tests -q ;;
        download)    n=${1:?download needs a song count, e.g. download 100}; shift
                     run download python scripts/download_validation_songs.py "$SONGS_CSV" --n "$n" ;;
        cache)       run cache python scripts/calibration/cache_validation_features.py ;;
        pca)         run pca python scripts/build_panns_pca.py ;;
        train)       run train python scripts/train_vggish_ridge.py "$SONGS_CSV" ;;
        validate)    run validate python scripts/validate_pipeline_accuracy.py \
                         --per-song-csv data/validation_songs/per_song_results.csv ;;
        sensitivity) run sensitivity python scripts/evaluate_prediction_sensitivity.py "$SONGS_CSV" ;;
        experiment)  script=${1:?experiment needs a script name}; shift
                     run "${script%.py}" python "scripts/calibration/$script" ;;
        *)           echo "unknown stage: $stage (see the top of $0)"; exit 2 ;;
    esac
done
