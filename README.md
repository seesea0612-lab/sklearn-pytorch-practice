# Scikit-learn and PyTorch Practice

A basic machine learning practice project using scikit-learn and PyTorch.

## What This Project Does

This project contains two simple examples:

1. PyTorch Tensor operations
2. Linear Regression using the scikit-learn Diabetes dataset

## PyTorch Tensor Demo

The program creates two tensors and performs:

- Tensor addition
- Scalar multiplication
- Dot product

Example output:

```text
Tensor a: tensor([1., 2., 3.])
Tensor b: tensor([4., 5., 6.])
a + b = tensor([5., 7., 9.])
a * 2 = tensor([2., 4., 6.])
Dot product = tensor(32.)
```

## Linear Regression Demo

The built-in Diabetes dataset from scikit-learn is used.

The dataset contains:

- 442 samples
- 10 features

The data is divided into:

- 80% training data
- 20% test data

The model is trained using:

```python
LinearRegression()
```

## Evaluation Results

With `random_state=42`, the model produces:

```text
MAE: 42.79
MSE: 2900.19
```

MAE means Mean Absolute Error.

The model's predictions differ from the actual values by about 42.79 target-value units on average.

## Requirements

- Python 3
- scikit-learn
- PyTorch

## Installation

Install scikit-learn:

```bash
py -m pip install scikit-learn
```

Install the CPU version of PyTorch:

```bash
py -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Run

Run the program with:

```bash
py ml_basics_demo.py
```

## Main File

```text
ml_basics_demo.py
```