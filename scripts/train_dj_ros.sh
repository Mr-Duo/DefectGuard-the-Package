#!/bin/bash

mkdir SETUP1 SETUP2 SETUP3 SETUP4 SETUP5
mkdir SETUP1/ros SETUP2/ros SETUP3/ros SETUP4/ros SETUP5/ros

# python3 -m defectguard.cli training  \
#     -model deepjit \
#     -feature_train_set "dataset/FFmpeg/SETUP1/ros/SETUP1-FFmpeg-features-train.jsonl" \
#     -commit_train_set "dataset/FFmpeg/SETUP1/ros/SETUP1-FFmpeg-deepjit-train.jsonl" \
#     -commit_val_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP1/ros \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C  \
#     -epoch 30

# python3 -m defectguard.cli training  \
#     -model deepjit \
#     -feature_train_set "dataset/FFmpeg/SETUP2/ros/SETUP2-FFmpeg-features-train.jsonl" \
#     -commit_train_set "dataset/FFmpeg/SETUP2/ros/SETUP2-FFmpeg-deepjit-train.jsonl" \
#     -commit_val_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP2/ros \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C \
#     -epoch 30
    
# python3 -m defectguard.cli training  \
#     -model deepjit \
#     -feature_train_set "dataset/FFmpeg/SETUP3/ros/SETUP3-FFmpeg-features-train.jsonl" \
#     -commit_train_set "dataset/FFmpeg/SETUP3/ros/SETUP3-FFmpeg-deepjit-train.jsonl" \
#     -commit_val_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP3/ros \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C \
#     -epoch 30

# python3 -m defectguard.cli evaluating \
#     -model deepjit \
#     -feature_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-features-test.jsonl" \
#     -commit_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP1/ros \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C

# python3 -m defectguard.cli evaluating \
#     -model deepjit \
#     -feature_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-features-test.jsonl" \
#     -commit_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP2/ros \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C

# python3 -m defectguard.cli evaluating \
#     -model deepjit \
#     -feature_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-features-test.jsonl" \
#     -commit_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-deepjit-val.jsonl" \
#     -dictionary "dataset/FFmpeg/SETUP3/dict-FFmpeg.jsonl" \
#     -dg_save_folder SETUP3/ros \
#     -repo_name FFmpeg \
#     -device cuda \
#     -repo_language C

python3 -m defectguard.cli training  \
    -model deepjit \
    -feature_train_set "dataset/FFmpeg/SETUP4/ros/SETUP4-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP4/ros/SETUP4-FFmpeg-deepjit-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP4/ros \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model deepjit \
    -feature_test_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP4/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP4/ros \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli training  \
    -model deepjit \
    -feature_train_set "dataset/FFmpeg/SETUP5/ros/SETUP5-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP5/ros/SETUP5-FFmpeg-deepjit-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP5/ros \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model deepjit \
    -feature_test_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-deepjit-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP5/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP5/ros \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C