#!/bin/bash

# Predefined lists
setup=("SETUP1" "SETUP2" "SETUP3" "SETUP4" "SETUP5")
sampling=("rus" "unsampling" "ros")
model=("tlel", "lapredict", "lr")

# Nested loop
echo "Nested loop processing:"
for i in "${sampling[@]}"; do
    for j in "${setup[@]}"; do
        for k in "${model[@]}"; do
            echo "Train $k $j with $i"
            python3 -m defectguard.cli evaluating \
                -model $k \
                -feature_test_set "dataset/FFmpeg/$j/$j-FFmpeg-features-test.jsonl" \
                -commit_test_set "dataset/FFmpeg/$j/$j-FFmpeg-simcom-test.jsonl" \
                -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
                -dg_save_folder $j/$i \
                -repo_name FFmpeg \
                -device cuda \
                -repo_language C

            echo "Eval $k $j with $i"
            python3 -m defectguard.cli evaluating \
                -model $k \
                -feature_test_set "dataset/FFmpeg/$j/$j-FFmpeg-features-test.jsonl" \
                -commit_test_set "dataset/FFmpeg/$j/$j-FFmpeg-deepjit-test.jsonl" \
                -dictionary "dataset/FFmpeg/SETUP1/dict-FFmpeg.jsonl" \
                -dg_save_folder $j/$i \
                -repo_name FFmpeg \
                -device cuda \
                -repo_language C
            done
    done
done