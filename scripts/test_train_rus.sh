#!/bin/bash

defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP1/rus/SETUP1-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP1/rus/SETUP1-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP1/SETUP1-simcom-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP1/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C  \
    -epoch 30

defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP2/rus/SETUP2-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP2/rus/SETUP2-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP2/SETUP2-simcom-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP2/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30
    
defectguard training \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP3/rus/SETUP3-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP3/rus/SETUP3-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP3/SETUP3-simcom-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP3/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30