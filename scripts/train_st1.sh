#!/bin/bash

python3 -m defectguard.cli training  \
    -model simcom \
    -feature_train_set "dataset/$repo/SETUP1/$sampling/SETUP1-$repo-features-train.jsonl" \
    -commit_train_set "dataset/$repo/SETUP1/$sampling/SETUP1-$repo-simcom-train.jsonl" \
    -commit_val_set "dataset/$repo/SETUP1/SETUP1-$repo-simcom-val.jsonl" \
    -dictionary "dataset/$repo/dict-$repo.jsonl" \
    -dg_save_folder SETUP1/$sampling \
    -repo_name $repo \
    -device cuda \
    -repo_language C  \
    -epoch 30

python3 -m defectguard.cli evaluating \
    -model simcom \
    -feature_test_set "dataset/$repo/SETUP3/SETUP3-$repo-features-test.jsonl" \
    -commit_test_set "dataset/$repo/SETUP3/SETUP3-$repo-simcom-test.jsonl" \
    -dictionary "dataset/$repo/dict-$repo.jsonl" \
    -dg_save_folder SETUP3/$sampling \
    -repo_name $repo \
    -device cuda \
    -repo_language C