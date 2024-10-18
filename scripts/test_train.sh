#!/bin/bash

mkdir focal_loss
mkdir focal_loss/SETUP1 focal_loss/SETUP2 focal_loss/SETUP3 focal_loss/SETUP4 focal_loss/SETUP5
mkdir focal_loss/SETUP1/unsampling focal_loss/SETUP2/unsampling focal_loss/SETUP3/unsampling focal_loss/SETUP4/unsampling focal_loss/SETUP5/unsampling

python3 -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP1/unsampling/SETUP1-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP1/unsampling/SETUP1-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP1/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C  \
    -epoch 30

python3 -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP2/unsampling/SETUP2-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP2/unsampling/SETUP2-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP2/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30
    
python3 -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP3/unsampling/SETUP3-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP3/unsampling/SETUP3-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP3/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-simcom-test.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP1/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-simcom-test.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP2/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-simcom-test.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP3/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP3/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP4/unsampling/SETUP4-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP4/unsampling/SETUP4-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP4/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP4/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP4/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP5/unsampling/SETUP5-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP5/unsampling/SETUP5-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP5/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP5/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP5/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C