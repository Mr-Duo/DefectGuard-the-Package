#!bin/bash

defectguard inferencing \
    -dg_save_folder . \
    -models deepjit \
    -dataset platform \
    -repo_name Tic-tac-toe-Game-using-Network-Socket-APIs \
    -repo_path /home/manh/Documents/DefectGuard \
    -uncommit \
    -top 10 \
    -sort \
    -repo_language C \
    -vsc
