#!/bin/bash

mkdir focal_loss
mkdir focal_loss/SETUP1 focal_loss/SETUP2 focal_loss/SETUP3 focal_loss/SETUP4 focal_loss/SETUP5
mkdir focal_loss/SETUP1/rus focal_loss/SETUP2/rus focal_loss/SETUP3/rus focal_loss/SETUP4/rus focal_loss/SETUP5/rus

python3 -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP1/rus/SETUP1-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP1/rus/SETUP1-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP1/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C  \
    -epoch 30

python3 -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP2/rus/SETUP2-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP2/rus/SETUP2-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP2/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30
    
python3 -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP3/rus/SETUP3-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP3/rus/SETUP3-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP3/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-simcom-test.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP1/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-simcom-test.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP2/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-simcom-test.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP3/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP3/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP4/rus/SETUP4-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP4/rus/SETUP4-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP4/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP4/SETUP4-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP4/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP4/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/FFmpeg/SETUP5/rus/SETUP5-FFmpeg-features-train.jsonl" \
    -commit_train_set "dataset/FFmpeg/SETUP5/rus/SETUP5-FFmpeg-simcom-train.jsonl" \
    -commit_val_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP5/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP5/SETUP5-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP5/dict-FFmpeg.jsonl" \
    -dg_save_folder focal_loss/SETUP5/rus \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C