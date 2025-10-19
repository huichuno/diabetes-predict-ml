# Diabetes Disease Prediction

Machine learning project for diabetes disease detection. This project uses Logistic Regression and Random Forest Classifier.

## Getting Started

### Prerequisites
* uv  - https://docs.astral.sh/uv/getting-started/installation/
* git - https://git-scm.com/downloads (Windows only)
* vsc - https://code.visualstudio.com/download

### Supported OS
* Windows 11
* Ubuntu 24.04 LTS

### Installation
```sh
git clone https://github.com/huichuno/diabetes-predict-ml.git && cd diabetes-predict-ml

uv sync
```

### Usage

* Model training notebook: **diabetes.ipynb**

* Model inference notebook: **model_inference.ipynb**

* Web app: **app.py**  (refer to 'How to launch *Diabetes Xpert Web App*' section below)

## How to configure Visual Studio Code

#### Select python intepreter
* Press "ctrl + shift + p" > type "Python: Select Intepreter" > select "diabetes-predict-ml" venv

#### Select jupyter kernel
* Click "Select Kernel" > select "Python Environment" > select "diabetes-predict-ml"

## How to launch *Diabetes Xpert Web App*

* Train and generate ML models using **diabetes.ipynb** notebook. Model files will be created in *bin* folder

* Launch web app
```sh
uv run streamlit run app.py

# Web app will open automatically in the browser.
# Otherwise, navigate to http://localhost:8501/ into your browswer.
```

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
