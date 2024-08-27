#!bin/bash

defectguard evaluating \
    -model simcom \
    -dg_save_folder . \
    -repo_name FFmpeg \
    -repo_language C \
    -feature_test_set "dataset/FFmpeg/feature/SETUP1-features-test-FFmpeg.jsonl" \
    -commit_test_set "dataset/FFmpeg/commit/SETUP1-simcom-test-FFmpeg.jsonl"