<<<<<<< HEAD
#!/bin/bash
defectguard -debug training -model simcom -dg_save_folder . -repo_name javascript -repo_language JavaScript 
defectguard -debug evaluating -model simcom -dg_save_folder . -repo_name javascript -repo_language JavaScript 


=======
#!bin/bash
defectguard -debug training -model deepjit -dg_save_folder . -repo_name javascript -repo_language JavaScript -device cuda -epochs 100;
defectguard -debug evaluating -model deepjit -dg_save_folder . -repo_name javascript -repo_language JavaScript -device cuda;
>>>>>>> 9730c4338fc3b995cc5a8371d9d2c19f45a00639
