#!/bin/bash

defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/sampled/ros/feature/SETUP1-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/sampled/ros/commit/SETUP1-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP1-deepjit-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP1 \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C  \
    -epoch 30

defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/sampled/ros/feature/SETUP2-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/sampled/ros/commit/SETUP1-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP2-deepjit-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP2 \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30
    
defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/sampled/ros/feature/SETUP3-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/sampled/ros/commit/SETUP1-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP3-deepjit-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP3 \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30