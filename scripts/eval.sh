#!/bin/bash

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP1/SETUP1-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP1/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP2/SETUP2-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP2/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-features-test.jsonl" \
    -commit_test_set "dataset/FFmpeg/SETUP3/SETUP3-FFmpeg-simcom-val.jsonl" \
    -dictionary "dataset/FFmpeg/SETUP3/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP3/unsampling \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C