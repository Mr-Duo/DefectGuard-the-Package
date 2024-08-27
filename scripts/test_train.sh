#!/bin/bash

python3 -m defectguard.cli training \
    -model lr \
    -feature_train_set "dataset/FFmpeg/feature/SETUP1-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/commit/SETUP1-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP1-deepjit-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP1 \
    -repo_name FFmpeg \
    -repo_language C

python3 -m defectguard.cli training \
    -model lr \
    -feature_train_set "dataset/FFmpeg/feature/SETUP2-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/commit/SETUP2-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP2-deepjit-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP2 \
    -repo_name FFmpeg \
    -repo_language C
    
python3 -m defectguard.cli training \
    -model lr \
    -feature_train_set "dataset/FFmpeg/feature/SETUP3-features-train-FFmpeg.jsonl" \
    -commit_train_set "dataset/FFmpeg/commit/SETUP3-simcom-train-FFmpeg.jsonl" \
    -commit_val_set "dataset/FFmpeg/commit/SETUP3-deepjit-val-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP3 \
    -repo_name FFmpeg \
    -repo_language C
    
python3 -m defectguard.cli evaluating \
    -model lr \
    -feature_test_set "dataset/FFmpeg/feature/SETUP1-features-test-FFmpeg.jsonl" \
    -commit_test_set "dataset/FFmpeg/commit/SETUP1-simcom-test-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP1 \
    -repo_name FFmpeg \
    -repo_language C

python3 -m defectguard.cli evaluating \
    -model lr \
    -feature_test_set "dataset/FFmpeg/feature/SETUP2-features-test-FFmpeg.jsonl" \
    -commit_test_set "dataset/FFmpeg/commit/SETUP2-simcom-test-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP2 \
    -repo_name FFmpeg \
    -repo_language C
    
python3 -m defectguard.cli evaluating \
    -model lr \
    -feature_test_set "dataset/FFmpeg/feature/SETUP2-features-test-FFmpeg.jsonl" \
    -commit_test_set "dataset/FFmpeg/commit/SETUP2-simcom-test-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP3 \
    -repo_name FFmpeg \
    -repo_language C