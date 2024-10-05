#!/bin/bash

mkdir SETUP1 SETUP2 SETUP3 SETUP4 SETUP5
mkdir SETUP1/unsampling SETUP2/unsampling SETUP3/unsampling SETUP4/unsampling SETUP5/unsampling

# python3 -m defectguard.cli training  \
#     -model deepjit \
#     -feature_train_set "dataset/FFmpeg/SETUP1/unsampling/SETUP1-FFmpeg-features-train.jsonl" \
#     -commit_train_set "dataset/FFmpeg/SETUP1/unsampling/SETUP1-FFmpeg-deepjit-train.jsonl" \
#     -commit_val_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP1/unsampling \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C  \
#     -epoch 30

# python3 -m defectguard.cli training  \
#     -model deepjit \
#     -feature_train_set "dataset/FFmpeg/SETUP2/unsampling/SETUP2-FFmpeg-features-train.jsonl" \
#     -commit_train_set "dataset/FFmpeg/SETUP2/unsampling/SETUP2-FFmpeg-deepjit-train.jsonl" \
#     -commit_val_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP2/unsampling \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C \
#     -epoch 30
    
# python3 -m defectguard.cli training  \
#     -model deepjit \
#     -feature_train_set "dataset/FFmpeg/SETUP3/unsampling/SETUP3-FFmpeg-features-train.jsonl" \
#     -commit_train_set "dataset/FFmpeg/SETUP3/unsampling/SETUP3-FFmpeg-deepjit-train.jsonl" \
#     -commit_val_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP3/unsampling \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C \
#     -epoch 30

# python3 -m defectguard.cli evaluating \
#     -model deepjit \
#     -feature_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-features-test.jsonl" \
#     -commit_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP1/unsampling \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C

# python3 -m defectguard.cli evaluating \
#     -model deepjit \
#     -feature_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-features-test.jsonl" \
#     -commit_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP2/unsampling \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C

# python3 -m defectguard.cli evaluating \
#     -model deepjit \
#     -feature_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-features-test.jsonl" \
#     -commit_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP3/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP3/unsampling \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C

python3 -m defectguard.cli training  \
    -model deepjit \
    -feature_train_set "dataset/FFmpeg/SETUP4/unsampling/SETUP4-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP4/unsampling/SETUP4-FFmpeg-deepjit-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP4/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model deepjit \
    -feature_test_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP4/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP4/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli training  \
    -model deepjit \
    -feature_train_set "dataset/FFmpeg/SETUP5/unsampling/SETUP5-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP5/unsampling/SETUP5-FFmpeg-deepjit-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP5/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model deepjit \
    -feature_test_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP5/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP5/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C