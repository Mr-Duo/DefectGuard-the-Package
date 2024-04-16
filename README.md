# DefectGuard: An Integrated, Efficient and Effective Tool for JIT Defect Prediction

## About

* DefectGuard is a python package
* Basic functionalities:
    * Mining commits from Git repositories
    * Post-processing, training, inferencing JITDP model via CLI or import library
* DefectGuard had been integrated into VSC *(extension)*, Jenkins & GitHub Action *(via command)*

## Prerequisite

### PySZZ

This tool utilize the code of [PySZZ](https://github.com/grosa1/pyszz_v2/). Please install PySZZ and pass its path as a parameter [example](###mining-commits-from-git-repositories).

**Notice**: to run PySZZ:
- Python 3.9
- [srcML](https://www.srcml.org/) (i.e., the srcml command should be in the system path)

### Python
```
Python 3.11.4
```

### Libraries
Check out this [requirements.txt](https://github.com/manhtdd/DefectGuard-the-Package/blob/main/requirements.txt)
```
pip install -r requirements.txt
```

## Installation

### via Pip
```
pip install -i https://test.pypi.org/simple/ defectguard==0.1.32
```

### via Docker
Build and run the Image like [this](https://github.com/manhtdd/DefectGuard-the-Package/blob/main/test_suits/docker_build_run.sh)

## Basic usages

### Mining commits from Git repositories
```
Coming soon
```

### Post-processing data
```
Coming soon
```

### Training

We provide a way to call JITDP model to your pipeline, so you can create your own pipeline. Checkout [this file](https://github.com/manhtdd/DefectGuard-the-Package/blob/main/test_suits/train.py) for example


### Inference

Using library call, checkout [this example](https://github.com/manhtdd/DefectGuard-the-Package/blob/main/test_suits/train.py)

Using CLI, checkout [this example](https://github.com/manhtdd/DefectGuard-the-Package/blob/main/test_suits/test_top_flag_3.sh)
For other usages, either checkout other example in [this folder](https://github.com/manhtdd/DefectGuard-the-Package/tree/main/test_suits) or wait for our docs

### Integrate into CLI-like Continuous Integration
* For GitHub Action, checkout [this example](https://github.com/manhtdd/DefectGuard-the-Package/blob/main/.github/workflows/python-package.yml)
* For Jenkins, here an example of a Jenkinsfile:
```
pipeline {
  agent any
  stages {
    stage('Checkout code') {
      steps {
        git(url: 'https://github.com/manhlamabc123/Tic-tac-toe-Game-using-Network-Socket-APIs', branch: 'main')
      }
    }
    stage('Run DefectGuard') {
      steps {
        sh '. /.venv/bin/activate && git config user.name manhlamabc123 && defectguard -models deepjit -dataset platform -repo . -uncommit -top 9 -main_language C -sort'
      }
    }
  }
}
```
