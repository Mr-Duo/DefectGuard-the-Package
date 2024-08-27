#!bin/bash

defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/feature/SETUP1-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/commit/SETUP1-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP1-deepjit-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder . \
    -repo_name FFmpeg \
    -repo_language C \