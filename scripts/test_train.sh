#!/bin/bash
defectguard -debug training -model simcom -dg_save_folder . -repo_name javascript -repo_language JavaScript 
defectguard -debug evaluating -model simcom -dg_save_folder . -repo_name javascript -repo_language JavaScript 


