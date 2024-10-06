#!/bin/bash

# Predefined lists
setup=("SETUP1" "SETUP2" "SETUP3" "SETUP4" "SETUP5")
sampling=("rus" "unsampling" "ros")

# Nested loop
echo "Nested loop processing:"
for i in "${sampling[@]}"; do
    for j in "${setup[@]}"; do
        echo "Eval simcom $j with $i"
        python3 -m defectguard.cli evaluating \
            -model simcom \
            -feature_test_set "dataset/FFmpeg/$j/$j-FFmpeg-features-test.jsonl" \
            -commit_test_set "dataset/FFmpeg/$j/$j-FFmpeg-simcom-test.jsonl" \
            -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
            -dg_save_folder $j/$i \
            -repo_name FFmpeg \
            -device cuda \
            -repo_language C

        echo "Eval deepjit $j with $i"
        python3 -m defectguard.cli evaluating \
            -model deepjit \
            -feature_test_set "dataset/FFmpeg/$j/$j-FFmpeg-features-test.jsonl" \
            -commit_test_set "dataset/FFmpeg/$j/$j-FFmpeg-deepjit-test.jsonl" \
            -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
            -dg_save_folder $j/$i \
            -repo_name FFmpeg \
            -device cuda \
            -repo_language C
    done
done