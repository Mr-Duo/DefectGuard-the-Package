#!/bin/bash

# Predefined lists
setup=("SETUP1" "SETUP2" "SETUP3" "SETUP4" "SETUP5")
sampling=("rus" "unsampling" "ros")
model=("tlel" "lapredict" "lr")

# Nested loop
echo "Nested loop processing:"
for i in "${sampling[@]}"; do
    for j in "${setup[@]}"; do
        for k in "${model[@]}"; do
            echo "Train $k $j with $i"
            python3 -m defectguard.cli training  \
                -model $k \
                -feature_train_set "dataset/FFmpeg/$j/$i/$j-FFmpeg-features-train.jsonl" \
                -commit_train_set "dataset/FFmpeg/$j/$i/$j-FFmpeg-simcom-train.jsonl" \
                -commit_val_set "dataset/FFmpeg/$j/$j-FFmpeg-simcom-val.jsonl" \
                -dictionary "dataset/FFmpeg/SETUP2/dict-FFmpeg.jsonl" \
                -dg_save_folder $j/$i \
                -repo_name FFmpeg \
                -device cuda \
                -repo_language C \
                -epoch 30

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