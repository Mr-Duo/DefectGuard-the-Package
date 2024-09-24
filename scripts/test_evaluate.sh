#!/bin/bash

defectguard evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/feature/SETUP1-features-test-FFmpeg.jsonl" \
    -commit_test_set "dataset/FFmpeg/commit/SETUP1-simcom-test-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP1 \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C

defectguard evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/feature/SETUP2-features-test-FFmpeg.jsonl" \
    -commit_test_set "dataset/FFmpeg/commit/SETUP2-simcom-test-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP2 \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C
    
defectguard evaluating \
    -model simcom \
    -feature_test_set "dataset/FFmpeg/feature/SETUP2-features-test-FFmpeg.jsonl" \
    -commit_test_set "dataset/FFmpeg/commit/SETUP2-simcom-test-FFmpeg.jsonl" \
    -dictionary "dataset/FFmpeg/commit/dict-FFmpeg.jsonl" \
    -dg_save_folder SETUP3 \
    -repo_name FFmpeg \
    -device cuda \
    -repo_language C