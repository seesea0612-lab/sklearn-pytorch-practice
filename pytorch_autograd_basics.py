import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# PyTorch Tensor, Gradient and Autograd Practice
# PyTorch 张量、梯度和自动求导练习
#
# Data source:
# scikit-learn built-in Diabetes dataset
# 数据来源：scikit-learn 内置 Diabetes 糖尿病数据集
# ============================================================


# 1. Fix random seeds
# 固定随机种子，保证实验尽可能可复现
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)

print("=== Random Seed ===")
print(f"NumPy seed: {SEED}")
print(f"PyTorch seed: {SEED}")


# 2. Load dataset
# 加载 Diabetes 糖尿病数据集
X, y = load_diabetes(return_X_y=True)

print("\n=== Original Dataset ===")
print("X shape:", X.shape)
print("y shape:", y.shape)


# 3. Split training and test sets
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=SEED
)

print("\n=== Train / Test Split ===")
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# 4. Standardize input features
# 对输入特征进行标准化
#
# Important:
# Scaler is fitted only on the training set
# StandardScaler 只在训练集上进行 fit，避免数据泄漏
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 5. Convert NumPy arrays to PyTorch tensors
# 将 NumPy 数组转换为 PyTorch Tensor（张量）
X_train_tensor = torch.tensor(
    X_train_scaled,
    dtype=torch.float32
)

y_train_tensor = torch.tensor(
    y_train,
    dtype=torch.float32
).reshape(-1, 1)

X_test_tensor = torch.tensor(
    X_test_scaled,
    dtype=torch.float32
)

y_test_tensor = torch.tensor(
    y_test,
    dtype=torch.float32
).reshape(-1, 1)

print("\n=== PyTorch Tensors ===")
print("X_train_tensor shape:", X_train_tensor.shape)
print("y_train_tensor shape:", y_train_tensor.shape)
print("X_train_tensor dtype:", X_train_tensor.dtype)
print("y_train_tensor dtype:", y_train_tensor.dtype)


# 6. Create model parameters manually
# 手动创建模型参数
#
# requires_grad=True means PyTorch will track operations
# and automatically calculate gradients.
#
# requires_grad=True 表示 PyTorch 会追踪相关运算，
# 从而可以自动计算梯度。
num_features = X_train_tensor.shape[1]

W = torch.zeros(
    (num_features, 1),
    dtype=torch.float32,
    requires_grad=True
)

b = torch.zeros(
    1,
    dtype=torch.float32,
    requires_grad=True
)

print("\n=== Model Parameters ===")
print("W shape:", W.shape)
print("b shape:", b.shape)
print("W requires_grad:", W.requires_grad)
print("b requires_grad:", b.requires_grad)


# 7. Forward pass
# 前向传播
#
# Linear model:
# y_pred = XW + b
#
# 线性模型：
# 预测值 = 输入 × 权重 + 偏置
y_pred = X_train_tensor @ W + b

print("\n=== Forward Pass ===")
print("Prediction shape:", y_pred.shape)


# 8. Calculate MSE loss
# 计算 Mean Squared Error（均方误差）
loss = torch.mean(
    (y_pred - y_train_tensor) ** 2
)

print("\n=== Before Backward ===")
print("Initial loss:", loss.item())
print("W.grad:", W.grad)
print("b.grad:", b.grad)


# 9. Backward propagation
# 反向传播
#
# backward() triggers PyTorch Autograd.
# backward() 会启动 PyTorch 自动求导机制。
loss.backward()

print("\n=== After Backward ===")
print("W gradient shape:", W.grad.shape)
print("b gradient shape:", b.grad.shape)

print("First 3 gradients of W:")
print(W.grad[:3])

print("Gradient of b:")
print(b.grad)


# 10. Manually perform one Gradient Descent step
# 手动执行一次 Gradient Descent（梯度下降）
learning_rate = 0.01

with torch.no_grad():
    W -= learning_rate * W.grad
    b -= learning_rate * b.grad


# 11. Clear gradients
# 清空梯度
#
# PyTorch accumulates gradients by default.
# PyTorch 默认会累加梯度，因此需要手动清零。
W.grad.zero_()
b.grad.zero_()

print("\n=== After Gradient Reset ===")
print("W.grad:")
print(W.grad)

print("b.grad:")
print(b.grad)


# 12. Calculate the loss again
# 使用更新后的参数重新计算 Loss（损失）
new_prediction = X_train_tensor @ W + b

new_loss = torch.mean(
    (new_prediction - y_train_tensor) ** 2
)

print("\n=== Result ===")
print("Loss before update:", loss.item())
print("Loss after one update:", new_loss.item())

if new_loss.item() < loss.item():
    print("Result: Loss decreased successfully.")
else:
    print("Result: Loss did not decrease.")


# 13. Record one learning question
# 记录一个学习过程中遇到的问题
print("\n=== Learning Question ===")
print(
    "Why do we need to clear gradients before the next backward pass?"
)

print(
    "Answer: Because PyTorch accumulates gradients by default. "
    "If gradients are not cleared, the next backward() call will "
    "add new gradients to the old gradients."
)