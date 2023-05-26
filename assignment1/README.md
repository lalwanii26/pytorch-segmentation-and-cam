# ECE 285 Assignment 1

## Overview

Assignment 1 includes 5 parts:

1. KNN
2. Linear Regression
3. Logistic Regression
4. Neural Network in NumPy
5. Classification using Neural Network

File structure:

```
assignment1
├── README.md  (this file)
├── ece285  (source code folder)
├── *.ipynb  (notebooks)
├── get_datasets.py  (download script)
└── datasets  (datasets folder)
```

## Prepare Datasets

Before you start, you need to run the following command (in terminal or in notebook beginning with `!` ) to download the datasets:

```sh
# This command will download required datasets and put it in "./datasets".
python get_datasets.py
```

## Implementation

You should run all code blocks in the following jupyter notebooks and write your answers to all inline questions included in the notebooks:

1. `knn.ipynb`
2. `linear_regression.ipynb`
3. `logistic_regression.ipynb`
4. `neural_network.ipynb`
5. `classification_nn.ipynb`

Go through the notebooks to understand the structure. These notebooks will require you to complete the following implementations:

1. `ece285/algorithms/knn.py`: Implement KNN algorithm
2. `ece285/algorithms/linear_regression.py`: Implement linear regression algorithm
3. `ece285/algorithms/logistic_regression.py`: Implement logistic regression algorithm
4. `ece285/layers/linear.py`: Implement linear layers with arbitrary input and output dimensions
5. `ece285/layers/relu.py`: Implement ReLU activation function, forward and backward pass
6. `ece285/layers/softmax.py`: Implement softmax function to calculate class probabilities
7. `ece285/layers/loss_func.py`: Implement CrossEntropy loss, forward and backward pass

You are required to go through all the modules (the ones that are already implemented for you as well) one by one to to understand the structure. This will help you when transitioning to deep learning libraries such as PyTorch. Here are some files that you should go through that are already implemented for you:

1. `ece285/layers/sequential.py`
2. `ece285/utils/trainer.py`
3. `ece285/utils/optimizer.py`

### How to Set Up Python Environment

We prepare a `requirements.txt` file for you to install python packages. You can install the packages by running the following command in terminal:

```sh
pip install -r requirements.txt
```

This should solve most package issues. But if you still have some problems, we recommend you to use conda environment. You can install anaconda or miniconda by following the instruction on [https://docs.anaconda.com/anaconda/install/index.html](https://docs.anaconda.com/anaconda/install/index.html). After you install it, you can run the following command to set up python environment:

```sh
conda create -n ece285 python=3.9.5  # same python version as the datahub
conda activate ece285
pip install -r requirements.txt
```
