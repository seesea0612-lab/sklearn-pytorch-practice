# Scikit-learn and PyTorch Practice

A basic machine learning practice project using scikit-learn and PyTorch.

## What This Project Does

This project contains three simple examples:

1. PyTorch Tensor operations
2. Linear Regression using the scikit-learn Diabetes dataset
3. Wine dataset preparation and train/test splitting

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
- pandas
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
## Wine Dataset Preparation

This project also includes a small data preparation exercise using the Wine dataset provided by scikit-learn.

### Dataset

The dataset is loaded using:

```python
from sklearn.datasets import load_wine

wine = load_wine(as_frame=True)
```

The original dataset contains:

- 178 samples
- 13 input features
- 1 target column
- 14 columns in total

The raw dataset is saved to:

```text
data/raw/wine_raw.csv
```

### Data Quality Check

The dataset was checked for:

- Missing values
- Duplicate rows
- Data types

Results:

- Missing values: 0
- Duplicate rows: 0
- Feature data types: 13 `float64` columns
- Target data type: 1 `int64` column

No additional data cleaning was required.

The full data check report is available at:

```text
reports/wine_data_check.txt
```

### Train/Test Split

The dataset was split using scikit-learn's `train_test_split` with:

```python
test_size=0.20
random_state=42
stratify=y
```

Result:

- Training samples: 142
- Test samples: 36
- Training ratio: 79.78%
- Test ratio: 20.22%

`stratify=y` was used to preserve the class distribution between the training and test sets.

### Preprocessing

`StandardScaler` was used for feature standardization.

To avoid data leakage, the scaler was fitted only on the training set:

```python
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

The test set was therefore transformed using only the statistics learned from the training set.

### Generated Files

```text
data/
├── raw/
│   └── wine_raw.csv
└── processed/
    ├── wine_train.csv
    └── wine_test.csv

reports/
└── wine_data_check.txt
```

The complete preparation script is:

```text
wine_data_preparation.py
```

Run it with:

```bash
py wine_data_preparation.py
```
## Diabetes Linear Regression

This example uses the built-in Diabetes dataset from scikit-learn to train
and evaluate a Linear Regression model.

### Tasks

- Load the Diabetes dataset using `load_diabetes`
- Split the dataset into training and test sets
- Train a `LinearRegression` model
- Evaluate the model using MAE and MSE
- Compare predicted values with actual values

### Run

```powershell
py diabetes_regression.py
```

### Results

```text
Dataset size: 442 samples
Training samples: 353
Test samples: 89

MAE: 42.79
MSE: 2900.19
```

### Conclusion

The model achieved an MAE of 42.79. This means that the predicted
target values differ from the actual target values by about 43 units
on average. Linear Regression provides a useful baseline, but some
individual predictions still have relatively large errors.
## PyTorch Tensor, Gradient and Autograd Practice

File: `pytorch_autograd_basics.py`

This exercise practices the basic PyTorch workflow including tensors, gradients,
automatic differentiation, and a manual gradient descent update.

### Dataset

The exercise uses the built-in Diabetes dataset from scikit-learn.

Dataset shape:

- Samples: 442
- Features: 10
- Training samples: 353
- Test samples: 89

### Topics Practiced

- PyTorch Tensor creation
- Tensor shapes and data types
- `requires_grad=True`
- Forward pass
- Mean Squared Error (MSE)
- `loss.backward()`
- Gradient calculation
- Manual gradient descent
- Gradient reset
- Reproducibility with random seeds

### Run

```powershell
py pytorch_autograd_basics.py
```

### Main Result

The initial training loss was:

```text
29711.322265625
```

After one manual gradient descent update:

```text
28434.923828125
```

The loss decreased successfully, showing that the calculated gradients updated
the model parameters in the correct direction.

### Learning Question

**Why do we need to clear gradients before the next backward pass?**

PyTorch accumulates gradients by default. If the gradients are not cleared,
the next `backward()` call adds the newly calculated gradients to the previous
ones instead of replacing them.