# Diabetes Disease Prediction

Machine learning project for diabetes disease detection. This project uses Logistic Regression and Random Forest Classifier.

## Getting Started

### Prerequisties
* uv  - https://docs.astral.sh/uv/getting-started/installation/
* git - https://git-scm.com/downloads

### Supported OS
* Windows 11

### Installation
```sh
git clone https://github.com/huichuno/diabetes-predict-ml.git && cd diabetes-predict-ml

uv sync
```

### Usage

* Model training notebook: **diabetes.ipynb**

* Model inference notebook: **model_inference.ipynb**

## How to configure Visual Studio Code

#### Select python intepreter
* Press "ctrl + shift + p" > type "Python: Select Intepreter" > select "diabetes-predict-ml" venv

#### Select jupyter kernel
* Click "Select Kernel" > select "diabetes-predict-ml" kernel

## How to create project from scratch

### Install Python
```sh
uv python list
uv python install 3.13
```

### Setup venv
```sh
uv init --python 3.13
uv add pandas
uv add matplotlib
uv add seaborn
uv add scikit-learn
```

### Install jupyter kernel
```sh
uv add ipykernel
```
