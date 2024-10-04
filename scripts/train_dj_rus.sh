#!/bin/bash

mkdir SETUP1 SETUP2 SETUP3
mkdir SETUP1/rus SETUP2/rus SETUP3/rus

python3 -m defectguard.cli training  \
    -model deepjit \
    -feature_train_set "dataset/FFmpeg/SETUP1/rus/SETUP1-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP1/rus/SETUP1-FFmpeg-deepjit-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP1/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C  \
    -epoch 30

python3 -m defectguard.cli training  \
    -model deepjit \
    -feature_train_set "dataset/FFmpeg/SETUP2/rus/SETUP2-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP2/rus/SETUP2-FFmpeg-deepjit-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP2/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30
    
python3 -m defectguard.cli training  \
    -model deepjit \
    -feature_train_set "dataset/FFmpeg/SETUP3/rus/SETUP3-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP3/rus/SETUP3-FFmpeg-deepjit-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP3/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model deepjit \
    -feature_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP1/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli evaluating \
    -model deepjit \
    -feature_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP2/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli evaluating \
    -model deepjit \
    -feature_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP3/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP3/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C