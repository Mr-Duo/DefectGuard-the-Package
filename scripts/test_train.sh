#!/bin/bash

defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/sampled/rus/feature/SETUP1-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/sampled/rus/commit/SETUP1-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP1-simcom-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP1/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C  \
    -epoch 30

defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/sampled/rus/feature/SETUP2-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/sampled/rus/commit/SETUP2-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP2-simcom-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP2/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30
    
defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/sampled/rus/feature/SETUP3-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/sampled/rus/commit/SETUP3-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP3-simcom-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP3/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30